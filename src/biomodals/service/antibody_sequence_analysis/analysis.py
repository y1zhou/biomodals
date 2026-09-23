"""Bounded FASTA parsing and per-request deduplicated local computation."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import Callable, Coroutine
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import BoundedSemaphore
from typing import Any

from biomodals.helper.antibody import (
    ChainAnalysis,
    analyze_chain,
    combined_pi,
    normalize_sequence,
)
from biomodals.service.antibody_sequence_analysis.contracts import (
    MAX_ENTRIES,
    AnalysisEntry,
    AnalysisGroup,
    AnalysisIssue,
    AnalysisRequest,
    AnalysisResponse,
    AnalyzedChain,
    FastaGroup,
    ReferenceInfo,
)
from biomodals.service.antibody_sequence_analysis.reference import (
    SOURCE_URL,
    ReferenceSnapshot,
    TherapeuticReference,
    present_germlines,
)
from biomodals.service.http_contract import CodedAPIError

LOGGER = logging.getLogger(__name__)


@dataclass
class ParsedEntry:
    """One input entry before native chain recognition."""

    id: str
    sequences: dict[str, str] = field(default_factory=dict)
    issues: list[AnalysisIssue] = field(default_factory=list)


def parse_fasta(group: FastaGroup) -> tuple[list[ParsedEntry], list[AnalysisIssue]]:
    """Accept colon pairs, explicit suffixed partners and standalone records."""
    records: list[tuple[str, list[str]]] = []
    for line in group.fasta.splitlines():
        if not line.strip():
            continue
        if line.startswith(">"):
            header = line[1:].split()
            if not header:
                return [], [
                    AnalysisIssue(
                        code="invalid_header",
                        detail="Every FASTA record needs an identifier",
                    )
                ]
            records.append((header[0], []))
            if len(records) > 2 * MAX_ENTRIES:
                return [], [
                    AnalysisIssue(
                        code="too_many_entries",
                        detail=f"Use at most {MAX_ENTRIES} entries per group",
                    )
                ]
        elif not records:
            return [], [
                AnalysisIssue(
                    code="invalid_fasta", detail="Start each record with >identifier"
                )
            ]
        else:
            records[-1][1].append(line)
    if not records:
        return [], [
            AnalysisIssue(code="empty_group", detail="Enter at least one FASTA record")
        ]
    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for identifier, lines in records:
        role = identifier[-3:].lower()
        paired_suffix = role in ("_vh", "_vl")
        entry_id = identifier[:-3] if paired_suffix else identifier
        grouped[entry_id].append((
            role[1:] if paired_suffix else "sequence",
            "".join(lines),
        ))
    if len(grouped) > MAX_ENTRIES:
        return [], [
            AnalysisIssue(
                code="too_many_entries",
                detail=f"Use at most {MAX_ENTRIES} entries per group",
            )
        ]
    entries = []
    for identifier, members in grouped.items():
        entry = ParsedEntry(identifier)
        entries.append(entry)
        roles = [role for role, _ in members]
        if (
            not identifier
            or len(roles) != len(set(roles))
            or ("sequence" in roles and len(roles) > 1)
        ):
            entry.issues.append(
                AnalysisIssue(
                    code="duplicate_id",
                    detail="Use unique nonempty entry IDs within each group",
                )
            )
            continue
        supplied = dict(members)
        if "sequence" in supplied:
            raw = supplied["sequence"]
            if ":" in raw:
                if raw.count(":") != 1:
                    entry.issues.append(
                        AnalysisIssue(
                            code="invalid_pair",
                            detail="Use exactly one colon between VH and VL",
                        )
                    )
                    continue
                vh, vl = raw.split(":")
                supplied = {"vh": vh, "vl": vl}
        elif set(supplied) != {"vh", "vl"}:
            entry.issues.append(
                AnalysisIssue(
                    code="missing_partner",
                    detail="Both _vh and _vl records are required for a suffixed pair",
                )
            )
            continue
        for role, sequence in supplied.items():
            try:
                entry.sequences[role] = normalize_sequence(sequence)
            except ValueError as error:
                entry.issues.append(
                    AnalysisIssue(code="invalid_sequence", detail=f"{role}: {error}")
                )
        if entry.issues:
            entry.sequences.clear()
    return entries, []


class AnalysisService:
    """Bounded local CPU work; inputs live only until admitted work finishes."""

    def __init__(
        self, reference: TherapeuticReference, *, max_requests: int = 8
    ) -> None:
        """Keep reference I/O independent of the result-archive worker."""
        if max_requests < 1:
            raise ValueError("Local analysis request capacity must be positive")
        self.reference = reference
        self._pool = ThreadPoolExecutor(
            max_workers=8, thread_name_prefix="antibody-analysis"
        )
        self._capacity = BoundedSemaphore(max_requests)
        self._requests: set[asyncio.Task] = set()
        self._closed = False
        self._reference_task: asyncio.Task[ReferenceSnapshot] | None = None

    async def _admit[T](
        self, operation: Callable[..., Coroutine[Any, Any, T]], *args: object
    ) -> T:
        if self._closed or not self._capacity.acquire(blocking=False):
            raise CodedAPIError(
                503,
                "local_analysis_busy",
                "Local sequence analysis is busy. Your input is unchanged; try again shortly.",
                headers={"Retry-After": "1"},
            )
        task = asyncio.create_task(operation(*args))
        self._requests.add(task)
        task.add_done_callback(self._finished)
        # An HTTP disconnect must not release capacity while native work is
        # still running. The bounded operation finishes without its waiter.
        return await asyncio.shield(task)

    def _finished(self, task: asyncio.Task) -> None:
        self._requests.discard(task)
        self._capacity.release()
        if not task.cancelled():
            task.exception()  # Observe failures even when the HTTP waiter left.

    async def _work[T](self, operation: Callable[..., T], *args: object) -> T:
        future = asyncio.get_running_loop().run_in_executor(
            self._pool, operation, *args
        )
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            # Loop shutdown can cancel the admitted Task too. Native threads
            # cannot be interrupted; drain them before releasing its slot.
            await asyncio.shield(future)
            raise

    async def run[T](self, operation: Callable[..., T], *args: object) -> T:
        """Admit one inspection or preparation through the shared local limit."""
        return await self._admit(self._work, operation, *args)

    async def reference_snapshot(
        self,
    ) -> tuple[ReferenceSnapshot | None, ReferenceInfo]:
        """Share in-flight initialization and report reference failure separately."""
        if self._reference_task is None:
            self._reference_task = asyncio.create_task(
                asyncio.to_thread(self.reference.get)
            )
        task = self._reference_task
        try:
            snapshot = await asyncio.shield(task)
        except Exception:
            LOGGER.warning(
                "Therapeutic germline reference is unavailable", exc_info=True
            )
            # A later explicit analysis can retry an absent cache; an invalid file
            # remains untouched and continues to report unavailable.
            if self._reference_task is task:
                self._reference_task = None
            return None, ReferenceInfo(
                status="unavailable",
                source_url=SOURCE_URL,
                detail="Therapeutic usage is unavailable. An administrator can inspect the reference cache; sequence metrics are unaffected.",
            )
        return snapshot, snapshot.info

    async def analyze(self, request: AnalysisRequest) -> AnalysisResponse:
        """Analyze valid entries once per exact chain, preserving group failures."""
        return await self._admit(self._analyze, request)

    async def _analyze(self, request: AnalysisRequest) -> AnalysisResponse:
        parsed = [parse_fasta(group) for group in request.groups]
        sequences = sorted({
            sequence
            for entries, _ in parsed
            for entry in entries
            for sequence in entry.sequences.values()
        })
        if not sequences:
            return self._assemble(
                request,
                parsed,
                {},
                None,
                ReferenceInfo(
                    status="unavailable",
                    source_url=SOURCE_URL,
                    detail="No valid sequences to analyze; reference data were not requested.",
                ),
            )
        reference_future = asyncio.create_task(self.reference_snapshot())
        remaining = iter(sequences)
        results = {}

        async def consume() -> None:
            for sequence in remaining:
                results[sequence] = await self._work(analyze_chain, sequence)

        # Four rolling calls per request preserve batch parallelism without
        # placing thousands of chains ahead of interactive work in the FIFO.
        outcomes = await asyncio.gather(
            *(consume() for _ in range(min(4, len(sequences)))),
            return_exceptions=True,
        )
        snapshot, info = await reference_future
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                raise outcome
        return await self._work(
            self._assemble, request, parsed, results, snapshot, info
        )

    @staticmethod
    def _assemble(
        request: AnalysisRequest,
        parsed: list[tuple[list[ParsedEntry], list[AnalysisIssue]]],
        results: dict[str, ChainAnalysis],
        snapshot: ReferenceSnapshot | None,
        info: ReferenceInfo,
    ) -> AnalysisResponse:
        """Reuse per-chain presentations and per-pair pI off the event loop."""
        chains = {}
        for sequence, analyzed in results.items():
            chain_type = analyzed["germlines"]["chain_type"]
            role = (
                "vh"
                if chain_type == "H"
                else "vl"
                if chain_type in ("K", "L")
                else "unassigned"
            )
            chains[sequence] = (
                role,
                AnalyzedChain(
                    sequence=sequence,
                    metrics=analyzed["metrics"],
                    germlines=present_germlines(analyzed["germlines"], role, snapshot),
                    germline_pi=analyzed["germline_pi"],
                ),
            )
        pair_pis = {}
        groups = []
        for group, (entries, issues) in zip(request.groups, parsed, strict=True):
            output = AnalysisGroup(id=group.id, entries=[], issues=issues)
            groups.append(output)
            for entry in entries:
                row = AnalysisEntry(id=entry.id, issues=entry.issues)
                output.entries.append(row)
                for role, sequence in entry.sequences.items():
                    analyzed = results[sequence]
                    chain_type = analyzed["germlines"]["chain_type"]
                    detected_role, chain = chains[sequence]
                    target = detected_role if role == "sequence" else role
                    setattr(row, target, chain)
                    if role != "sequence" and detected_role not in (role, "unassigned"):
                        row.issues.append(
                            AnalysisIssue(
                                code="wrong_chain_role",
                                detail=f"{role} input was recognized as {chain_type}; chains were not swapped",
                            )
                        )
                    if analyzed["germlines"]["error"]:
                        row.issues.append(
                            AnalysisIssue(
                                code="annotation_unavailable",
                                detail=f"{role}: antibody numbering/germline assignment unavailable; physical metrics retained",
                            )
                        )
                if (
                    row.vh is not None
                    and row.vl is not None
                    and not any(
                        issue.code == "wrong_chain_role" for issue in row.issues
                    )
                ):
                    pair = row.vh.sequence, row.vl.sequence
                    if pair not in pair_pis:
                        pair_pis[pair] = combined_pi(*pair)
                    row.vh_vl_pi = pair_pis[pair]
        return AnalysisResponse(groups=groups, reference=info)

    async def shutdown(self) -> None:
        """Finish active local work before closing the app."""
        self._closed = True
        await asyncio.gather(*self._requests, return_exceptions=True)
        if self._reference_task is not None:
            await asyncio.gather(self._reference_task, return_exceptions=True)
        await asyncio.to_thread(self._pool.shutdown, wait=True, cancel_futures=True)

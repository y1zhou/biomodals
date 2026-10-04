"""Validate downloaded model artifacts; optionally compare fresh OligoFormer runs.

An oracle directory must come from independently running the pinned upstream
with the evaluation-mode correction. This command does not generate an oracle.
"""

from __future__ import annotations

import argparse
import io
import math
import re
import tarfile
from pathlib import Path, PurePosixPath

import orjson
import polars as pl
import zstandard

from biomodals.helper.artifacts import sha256_bytes, sha256_file
from biomodals.helper.pdb import validate_pdb_content


def archive_files(path: Path) -> dict[str, bytes]:
    """Read regular members without extracting paths or following links."""
    files = {}
    total = 0
    with (
        path.open("rb") as compressed,
        zstandard.ZstdDecompressor().stream_reader(compressed) as stream,
    ):
        with tarfile.open(fileobj=stream, mode="r|") as archive:
            for member in archive:
                name = PurePosixPath(member.name)
                if (
                    name.is_absolute()
                    or ".." in name.parts
                    or member.issym()
                    or member.islnk()
                ):
                    raise ValueError(f"Unsafe archive member: {member.name}")
                if not member.isfile():
                    continue
                total += member.size
                if total > 512 * 1024 * 1024:
                    raise ValueError("Acceptance archive exceeds 512 MiB")
                if member.name in files:
                    raise ValueError("Duplicate archive member")
                files[member.name] = archive.extractfile(member).read()
    return files


def oligoformer_tables(files: dict[str, bytes]) -> dict[str, pl.DataFrame]:
    """Validate table identity, numeric scores, ranking, and filtering."""
    contents = {
        Path(name).name: content
        for name, content in files.items()
        if name.endswith(".txt")
    }
    base_names = [
        name
        for name in contents
        if not name.endswith(("_ranked.txt", "_ranked_filtered.txt"))
    ]
    if not base_names:
        raise ValueError("No OligoFormer candidate tables")
    tables = {}
    for name in base_names:
        base = tables[name] = pl.read_csv(io.BytesIO(contents[name]), separator="\t")
        ranked_name = name.removesuffix(".txt") + "_ranked.txt"
        filtered_name = name.removesuffix(".txt") + "_ranked_filtered.txt"
        for derived in (ranked_name, filtered_name):
            # A valid filtered table may contain only a header. Preserve the
            # source schema instead of inferring every empty column as String.
            tables[derived] = pl.read_csv(
                io.BytesIO(contents[derived]),
                separator="\t",
                schema_overrides=base.schema,
            )
        ranked = tables[ranked_name]
        filtered = tables[filtered_name]
        if base.is_empty() or base["pos"].n_unique() != base.height:
            raise ValueError("Missing or duplicate candidate identities")
        for table in (base, ranked, filtered):
            if not table["efficacy"].is_finite().fill_null(False).all():
                raise ValueError("Nonfinite efficacy score")
        if not ranked.sort("pos").equals(base.sort("pos")):
            raise ValueError("Ranked table changed candidate rows")
        if not ranked["efficacy"].is_sorted(descending=True):
            raise ValueError("Efficacy ranking is not descending")
        if not ranked.filter(pl.col("filter") == 0).equals(filtered):
            raise ValueError("Filtered table does not match passing ranked candidates")
    return tables


def compare_tables(left, right, *, efficacy_only: bool, atol: float) -> float:
    """Compare candidate order and scores against independent retained evidence."""
    from polars.testing import assert_frame_equal

    if set(left) != set(right):
        raise ValueError("Candidate table sets differ")
    maximum = 0.0
    for name, table in left.items():
        other = right[name]
        if efficacy_only:
            # This oracle establishes the efficacy correction, not equivalence
            # of Biomodals' separately versioned off-target/filter policies.
            if name.endswith(("_ranked.txt", "_ranked_filtered.txt")):
                continue
            table = table.select("pos", "siRNA", "efficacy").sort("pos")
            other = other.select("pos", "siRNA", "efficacy").sort("pos")
        assert_frame_equal(table, other, check_dtypes=False, rel_tol=0, abs_tol=atol)
        if table.height:
            maximum = max(maximum, (table["efficacy"] - other["efficacy"]).abs().max())
    return maximum


def oligoformer_provenance(files: dict[str, bytes]) -> dict:
    """Verify table digests and require content identities for custom references."""
    provenance = orjson.loads(
        next(
            content
            for name, content in files.items()
            if name.endswith("/provenance.json")
        )
    )
    if provenance["tables"] != {
        Path(name).name: sha256_bytes(content)
        for name, content in files.items()
        if name.endswith(".txt")
    }:
        raise ValueError("Table digest mismatch")
    config = provenance["config"]
    if config["off_target"] and not config["all_human"]:
        for name in ("utr.txt", "orf.txt"):
            digest = provenance["inputs"].get(name)
            if (
                not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            ):
                raise ValueError(f"Missing or invalid custom-reference digest: {name}")
    return provenance


def main() -> None:
    """Write an acceptance report even when validation fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("app", choices=("flowpacker", "oligoformer"))
    parser.add_argument("archive", type=Path)
    parser.add_argument("--repeat", type=Path)
    parser.add_argument("--oracle", type=Path)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.atol) or args.atol < 0:
        parser.error("atol must be finite and nonnegative")
    report = {"app": args.app, "status": "failed", "atol": args.atol}
    try:
        report["archive_sha256"] = sha256_file(args.archive)
        files = archive_files(args.archive)
        if args.app == "flowpacker":
            manifest_name, content = next(
                (name, content)
                for name, content in files.items()
                if name.endswith("/validation.json")
            )
            manifest = orjson.loads(content)
            root = str(PurePosixPath(manifest_name).parent)
            expected = {
                f"run_{index + 1}/{Path(item['name']).stem}.pdb"
                for item in manifest["inputs"]
                for index in range(manifest["n_samples"])
            }
            if manifest["use_confidence"]:
                expected.update(
                    f"best_run/{Path(item['name']).stem}.pdb"
                    for item in manifest["inputs"]
                )
            if not expected or expected != {
                item["path"] for item in manifest["structures"]
            }:
                raise ValueError("Structure coverage does not match the request")
            for item in manifest["structures"]:
                structure = files[f"{root}/{item['path']}"]
                validate_pdb_content(structure, max_bytes=128 * 1024 * 1024)
                if (
                    sha256_bytes(structure) != item["sha256"]
                    or len(structure) != item["size_bytes"]
                ):
                    raise ValueError("Structure digest mismatch")
            report["provenance"] = manifest
        else:
            tables = oligoformer_tables(files)
            provenance = oligoformer_provenance(files)
            report["provenance"] = provenance
            if args.repeat:
                repeated = archive_files(args.repeat)
                second_provenance = oligoformer_provenance(repeated)
                if provenance["efficacy_key"] == second_provenance["efficacy_key"]:
                    raise ValueError(
                        "Repeat reused the efficacy cache; run both with --force"
                    )
                for field in (
                    "inputs",
                    "model_identity",
                    "reference_identity",
                    "efficacy_checkpoint_sha256",
                    "efficacy_policy",
                    "upstream_commit",
                    "config",
                    "seed",
                ):
                    if provenance[field] != second_provenance[field]:
                        raise ValueError(f"Repeat changed {field}")
                report["repeat_max_error"] = compare_tables(
                    tables,
                    oligoformer_tables(repeated),
                    efficacy_only=False,
                    atol=args.atol,
                )
                report["repeat_sha256"] = sha256_file(args.repeat)
            if args.oracle:
                oracle_files = {
                    path.name: path.read_bytes() for path in args.oracle.glob("*.txt")
                }
                oracle = {
                    name: pl.read_csv(io.BytesIO(content), separator="\t")
                    for name, content in oracle_files.items()
                }
                report["oracle_max_error"] = compare_tables(
                    tables, oracle, efficacy_only=True, atol=args.atol
                )
                report["oracle_digests"] = {
                    name: sha256_bytes(content)
                    for name, content in oracle_files.items()
                }
            report["scientific_comparison"] = (
                "passed" if args.repeat and args.oracle else "not_run"
            )
        report["status"] = "passed"
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_bytes(orjson.dumps(report, option=orjson.OPT_INDENT_2))


if __name__ == "__main__":
    main()

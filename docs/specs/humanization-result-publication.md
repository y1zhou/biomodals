# Humanization result publication

Date: 2026-09-09. Status: accepted implementation contract.
The baseline findings below describe schema 2; schema 3 implements the lean
publication contract. No existing archive is rewritten.
The archive inventory below additionally uses read-only inspection of the cached
result for job `4a0e44bd-9fbb-49ae-8fc6-84b72619aa1b`.

## Baseline: schema 2 archive

This job's ZIP is 26,237,632 bytes: 29 selection rows, six consolidated CSV/Parquet
files totaling 1,238,516 bytes, a 180,768-byte manifest, and 26 native generation
publications totaling 24,810,234 bytes. ZIP packaging uses `ZIP_STORED`, so JSON
and base64 structures are not compressed by the outer archive.
[Packaging](../../src/biomodals/service/humanization/results.py#L201).

| Native generation content | Count / bytes | Overlap and information not consolidated |
| --- | --- | --- |
| Sapiens compressed bundles | 2 / 83,155 | Inputs, final/intermediate sequences overlap candidates and provenance. Native numbering, generation input/final profiles, and mutation history are not the common IMGT annotation table. |
| Humatch compressed bundles | 2 / 9,782 | Sequences and classifier distributions overlap union evaluation. Generation edit counts/stopping evidence, native alignment and mutation records are additional diagnostics. |
| p-AbNatiV2 JSON | 20 / 24,613,406 | Endpoint sequences, seed provenance, and score fields overlap consolidated outputs. Native AHo mutation annotations, generation-time profiles, predicted structures, and structural RMSD/CDR displacement metrics are additional evidence. |
| HuDiff JSON | 2 / 103,891 | Accepted sequences and mutations overlap candidates/IMGT mutations. The complete attempt ledger also records rejected/duplicate attempts, rejection reasons, and aligned sequences; these are not fully retained in the main table. |

The p-AbNatiV2 JSON contains base64-encoded input and final PDBs totaling
14,977,544 bytes in this job. Ten seed runs can converge to the same candidate,
but each JSON repeats its generation record and structure payload. Existing
evaluation Parquets are scoring-only: their sequence table has no structural
RMSD/displacement columns. Removing native files is therefore not lossless
unless these additional diagnostics are intentionally outside the download
contract or extracted elsewhere.
[p-AbNatiV2 publication](../../src/biomodals/app/design/pabnativ2/app.py#L575),
[Sapiens generation](../../src/biomodals/app/design/sapiens/app.py#L311),
[native copying and score consolidation](../../src/biomodals/workflow/humanization/export.py#L60).

## Baseline: schema 2 manifest duplication

The schema 2 manifest stored schema/run/status, parameters, derived root seeds, scientific
versions/protocols, candidate count, candidate origins, evaluation/generation
errors, native publication references, per-candidate scoring publication
manifests, ranking policy/interpretation, CDR definition, and file size/hash
inventory. [Definition](../../src/biomodals/workflow/humanization/export.py#L173).

In the inspected job, `scoring_publications` contains 87 nested manifests and is
about 139 KB before outer indentation: repeated runtime/model identities,
references, input/summary metadata, and hashes for per-call files that are not
the consolidated files delivered in the ZIP. The final `files` inventory already
hashes the delivered files, while `scientific_versions` and parameters capture
shared configuration once. Candidate provenance (about 11 KB here) is different:
it preserves all origins when multiple seeds or passes collapse to one candidate,
including parental no-ops hidden by the main table's null generation method.
[Exporter](../../src/biomodals/workflow/humanization/export.py#L127),
[candidate-origin construction](../../src/biomodals/workflow/humanization/artifacts.py#L102).

## Accepted schema 3 standard-download contract

- Keep `selection.csv`, `imgt_mutations.parquet`, and all four consolidated
  `scores/*.parquet` files. They answer final candidate selection and detailed
  evaluation questions without rerunning models.
- Keep a compact manifest containing run/schema/status, settings, scientific
  identity once, ranking policy, concise generation failures, and delivered-file
  sizes/hashes. Remove nested scoring manifests and native-copy bookkeeping from
  the user archive. Do not weaken validation of upstream publications before
  their values enter consolidated tables.
- Consolidate generation accounting into one compact `generation.parquet`:
  parent, method, seed/iteration/attempt, outcome/rejection reason, and retained
  candidate ID (including parent/no-op and deduplicated origins). This replaces
  duplicated manifest candidate provenance rather than creating another copy.
- Omit `native/` from the standard download. Do not delete original app/Task
  publications in Modal: leave them as operational diagnostic evidence under
  the existing retention policy. No new debug-download endpoint is needed now.
- Deliberately omit predicted PDBs, detailed generation profiles/alignments and
  structural diagnostics from the standard download. If users need those for
  selection, preserve the requested metrics in a dedicated consolidated table
  or structure artifact rather than copying every opaque native result.

These changes deliberately narrow the scientific download boundary; they do not
claim all native data are present in the Parquets. Result schema/scientific
identity becomes 3. The service continues to accept schema 2 and serve immutable
old archives. The ZIP transport remains humanization/1.

`generation.parquet` contains parent_id, method, source_id, root_seed, seed, iteration,
attempt_index, outcome, reason, and candidate_id. Outcomes are generated, no_op,
duplicate (a native HuDiff duplicate), rejected, no_candidates, or failed.
Multiple generated rows can reference the same exact union candidate; no origin
is arbitrarily selected as its owner. Rejected/failed/zero-yield outcomes have
null candidate IDs. Sequences live only in selection.csv. Failure rows describe
failed generator calls, not fictional sampling attempts. Manifest generation
failure counts summarize this ledger; evaluator errors remain in selection.csv.
`root_seed` is the seed passed to the app; `seed` is its reported derived
per-pair seed and remains null if the call failed without a publication.

## Download and cancellation contract

Use the existing shared helpers; no workflow-local transfer implementation:

- `read_modal_volume_file` reads the bounded manifest (16 MiB maximum).
- `download_modal_volume_files` downloads the validated membership list using
  the configured per-invocation concurrency.
- `run_blocking_io` runs that bulk transfer independently of the cache worker.
  It drains the worker before propagating cancellation, preserving temporary
  directory lifetimes. It does not stop an in-progress network transfer.
- `cache.run_bounded` retains local archive construction, hash verification,
  and atomic cache publication. Readiness and cached table queries must remain
  runnable while a network transfer is waiting.

The service owns a fresh temporary directory around the awaited transfer.
Cancellation during download waits for the transfer to finish, cleans that
directory, and does not proceed to archive publication. Existing Result
streaming and lease helpers remain unchanged. Do not copy AlphaFold3's plain
`asyncio.to_thread` wrapper around a caller-owned temporary directory: it does
not provide the same cancellation-draining lifetime guarantee.

Sources: [shared Volume helpers](../../src/biomodals/helper/modal_volume.py),
[executor/cache helpers](../../src/biomodals/service/artifacts.py),
[humanization adapter](../../src/biomodals/service/humanization/modal.py),
[archive validation](../../src/biomodals/service/humanization/results.py).

## Verification

Offline regressions cover:

- Exact union references for Sapiens intermediate designs, converged origins,
  parental no-ops, and independently seeded p-AbNatiV2 runs.
- HuDiff accepted, duplicate and rejected attempt accounting.
- Failed generator calls with root seeds, null unreported pair seeds and null
  retained candidate IDs.
- 200 parents with 5 Sapiens passes, 1 Humatch endpoint, 25 p-AbNatiV2 runs and
  25 HuDiff attempts: 11,400 possible selection rows, 11,200 generated origins.
  The exported manifest stays below 64 KiB and passes archive verification.
  Maximum-length multibyte IDs, maximum chain lengths and populated numeric
  scores fit the 32 MiB selection limit and support reading the final page.
  This is a synthetic publication-envelope check, not a paid inference run or
  a benchmark of maximum-size detailed score tables.
- Schema 2 and schema 3 archive packaging/restoration with identity, path and
  digest verification.
- A held fake download while cache readiness runs, followed by cancellation:
  its temporary directory survives until the worker exits, then is removed.

[Generation/export tests](../../tests/workflow/test_humanization_export.py),
[graph tests](../../tests/workflow/test_humanization_graph.py),
[service download tests](../../tests/service/test_humanization_modal.py).

Deployment of the new workflow and compatible service is required for new
schema 3 results; these local changes neither rewrite existing archives nor
alter already submitted scientific jobs.

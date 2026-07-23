# AlphaFold 3 small-BFD sharding benchmark plan

Status: executed with an approved focused-sweep scope; production promotion pending
Plan date: 2026-07-22
Results date: 2026-07-23
Original campaign: `small-bfd-phase1-v1`
Completed campaign: `small-bfd-phase1-v2`
Implementation target: `src/biomodals/app/fold/alphafold3_msa_app.py`

## Outcome

Build an isolated, one-off benchmark app that determines whether AlphaFold 3
small-BFD searches become materially faster when 64 FASTA shards are read
directly from a Modal Volume. The benchmark must establish both scientific
equivalence and production-shaped performance before it recommends a sharding
strategy.

The benchmark produces evidence and a ranked recommendation only. It does not
modify `alphafold3_app.py`, its volumes, its cache, or any workflow. Promotion
into the production app is a later, explicitly approved change.

The supporting research is in:

- [AlphaFold 3 MSA database sharding](alphafold3-msa-sharding.md)
- [AlphaFold 3 MSA storage scenarios](alphafold3-msa-storage-scenarios.md)
- [Scientifically faithful MSA sharding](alphafold3-msa-scientific-sharding.md)
- [Phase 1 benchmark results](alphafold3-msa-phase1-results.md)
- [AlphaFold 3 performance guide](https://github.com/google-deepmind/alphafold3/blob/main/docs/performance.md)

The original repeated matrix and GroEL stress phase below were not submitted.
After the smoke gate passed, the user approved a smaller one-shot topology sweep
to choose a production candidate without paying for repeated benchmark samples.
The results document records the exact executed scope, durable evidence, and
limitations; the remaining sections preserve the plan that guided the
implementation.

## Hard boundaries

- Create the temporary app file `alphafold3_msa_app.py`; do not edit or import
  `alphafold3_app.py`.
- Give the temporary app the catalog name `alphafold3_msa` and Modal App name
  `AlphaFold3-MSA-Benchmark`.
- Copy the proven AlphaFold image/environment setup where useful. Keep the same
  fork commit `987ad1cb7d7028b6d35908cf63fe7d951d98d6b6`, HMMER 3.4, and
  `jackhmmer_seq_limit.patch`.
- Do not mount model weights or the production MSA cache.
- Do not run inference, template search, the full data pipeline, or workflow
  integration.
- Phase 1 implements small BFD only. It does not implement four-database or
  per-sequence container fanout.
- Database FASTAs remain on Modal Volumes. Never stage a complete monolith or
  shard set on ephemeral SSD.
- Local commands are plan-only unless explicitly given `--submit`.
- Before any cost-incurring Modal submission, ask the user for permission and
  show the exact cases and maximum work about to be submitted.
- Do not run the smoke or measured matrix more than once for this campaign.
  Valid completion markers turn accidental reruns into no-ops.
- The separately authorized test file
  `tests/app/fold/test_alphafold3_msa_app.py` must remain uncommitted. No commit
  is authorized by this plan.

## Modal resources and mounts

Create or look up two isolated Volume v2 resources:

| Purpose | Modal Volume name | Benchmark mount mode |
| --- | --- | --- |
| Self-contained database profile | `AlphaFold3-msa-db-sharded` | read-only after preparation |
| Durable evidence | `AlphaFold3-MSA-Benchmark-outputs` | writable |

Only the explicit preparation function mounts the production
`AlphaFold3-msa-db` Volume, read-only. It also mounts the new sharded Volume
writable. Search and scan workers mount only the published sharded Volume
read-only and the evidence Volume writable.

Use `cpu=(0.125, 32.125)` for benchmark containers. The request is the
guaranteed floor and the limit is where throttling begins. Billing is based on
the greater of requested and actual use, so the low request can reduce setup
and tail cost while still permitting a 32-core search burst. High observed CPU
use alone is not a reason to increase the request: it means bursting worked.

Modal's current resource documentation does not publish one universal numeric
maximum CPU request. It says oversized requests are rejected at creation time.
The proposed tuple is already declared by the existing production app; any
declaration-time `InvalidError` is a preflight failure and must submit no
benchmark work. See [Modal resource requests and limits](https://modal.com/docs/guide/resources).

Use the existing app's conservative `memory=(1024, 131072)` and six-hour
timeout unless local validation reveals an incompatibility. Record actual
memory use rather than treating the limit as expected consumption.

Modal describes 2.5 GB/s as a theoretical maximum for a Volume, not a
guarantee. The scan matrix therefore measures both reader scaling inside one
container and aggregate scaling across containers; it does not assume each
container receives 2.5 GB/s. See [Modal Volumes](https://modal.com/docs/guide/volumes).

## Operation 1: prepare the database profile

Preparation is an explicit, resumable operation and never happens lazily in a
search worker.

### Published layout

The new Volume contains one immutable profile with both physical layouts:

```text
/profiles/small-bfd-64-v1/
  source/
    bfd-first_non_consensus_sequences.fasta
  shards/
    bfd-first_non_consensus_sequences.fasta-00000-of-00064
    ...
    bfd-first_non_consensus_sequences.fasta-00063-of-00064
  validation/
    source-stats.tsv
    shard-stats.tsv
    shard-summary.parquet
    seqkit-sum.json
  manifest.json
```

Build into a unique staging generation. Stream the source from the production
Volume to the new Volume without writing the full payload to local disk. Move
the validated generation into the immutable profile path and publish
`manifest.json` last. A valid existing profile is verified and reused.

### Shard recipe

Pin SeqKit 2.13.0 in the image. The sharding function accepts a validated
`seqkit_threads` argument with default `8`, and passes it using `-j` wherever
the SeqKit subcommand supports threads.

The version-1 recipe is:

1. Shuffle the exact small-BFD source with `seqkit shuffle --two-pass
   --update-faidx --rand-seed 23`.
2. Stream the shuffled records into `seqkit split2 --by-part 64`.
3. Rename outputs to AlphaFold's zero-based, five-digit
   `prefix-00000-of-00064` convention.
4. Record the exact command, SeqKit version, thread count, source identity, and
   every output digest in the manifest.

Thread count is operational provenance, not scientific profile identity, if
the resulting shard digests are unchanged.

### Publication gate

Stop without publishing unless all checks pass:

- the copied monolith SHA-256 equals the production source SHA-256;
- there are exactly 64 correctly named, nonempty shards;
- source and aggregate shard record counts and residue counts match;
- an order-independent `seqkit sum` matches between source and shards;
- `seqkit stats --all --tabular -j <threads>` succeeds for source and shards;
- measured `num_seqs` equals the published small-BFD Z value `65,984,053`;
- every shard's residue count is within 5% of the mean;
- byte sizes and SHA-256 digests are recorded for the monolith and every shard;
- `manifest.json` is written only after every artifact and validation report is
  durable.

Use Polars to aggregate the SeqKit tables and write
`shard-summary.parquet`. A mismatch with the published Z value is a review
condition, not something to silently normalize.

## Operation 2: Volume scan matrix

Each case reads the entire small-BFD dataset exactly once per pass. Run a first
pass and an immediate same-container repeat. Label them literally `first-pass`
and `immediate-repeat`; do not call them cold and warm because Modal's
distributed cache cannot be reset or proven cold.

`C×R` means containers × readers per container. Multi-container cases receive
disjoint shard assignments.

| Case | Physical layout | Topology | Purpose |
| --- | --- | ---: | --- |
| V0 | monolith | 1×1 | sequential reference |
| V1 | 64 shards | 1×1 | shard-open overhead |
| V2 | 64 shards | 1×2 | one-container reader scaling |
| V3 | 64 shards | 1×4 | one-container reader scaling |
| V4 | 64 shards | 1×8 | one-container reader scaling |
| V5 | 64 shards | 1×16 | one-container and fixed-16 reference |
| V6 | 64 shards | 2×8 | fixed 16 aggregate readers |
| V7 | 64 shards | 4×4 | fixed 16 aggregate readers |
| V8 | 64 shards | 4×16 | production-shaped 64-reader stress |

V5 serves both requested roles and is not run twice. Use bounded buffered
reads and report exact bytes, per-file start/finish/duration, per-container
throughput, aggregate throughput, and container placement metadata. Do not use
local `io.stat` as a proxy for Volume throughput.

Store scan evidence under:

```text
/benchmarks/small-bfd-phase1-v1/storage-scans/{case-id}/
```

## Operation 3: small-BFD search benchmark

Use the pinned AlphaFold Jackhmmer wrapper and unchanged scientific search
defaults. Add a minimal instrumented adapter in the temporary app because the
upstream public result omits per-shard timings and merged sharded `tblout`:

- invoke the pinned private `_query_db_shard`;
- request and retain `tblout` for monolith and every shard;
- retain per-shard timings;
- pass the shard results to the unchanged upstream
  `_merge_jackhmmer_results` for final A3M construction;
- do not change HMMER flags, filtering, sorting, truncation, or merge behavior.

The adapter is deliberately version-coupled and belongs only to the temporary
app.

### Queries

The screening query is the 120-residue pembrolizumab VH chain:

```text
QVQLVQSGVEVKKPGASVKVSCKASGYTFTNYYMYWVRQAPGQGLEWMGGINPSNGGTNFNEKFKNRVTLTTDSSTTTAYMELKSLQFDDTAVYYCARRDYRFDMGFDYWGQGTTVTVSS
```

- ID: `pembrolizumab-vh`
- sequence SHA-256:
  `5d92fab232244fa55131fc3b8d31b34990aa778623cdd906d58cf920dbdaf28f`
- expected shallow-MSA behavior is a hypothesis to measure, not fixed metadata.

The stress query is reviewed UniProt P0A6F5, *E. coli* K-12 GroEL, sequence
version 2:

- ID: `ecoli-k12-groel`
- length: 548
- sequence SHA-256:
  `40544c6fee0f15b6fe78d6ab7e5e27d8080224fe28dc0d6ca6f2e9a790dd24d4`
- source: [UniProt P0A6F5](https://www.uniprot.org/uniprotkb/P0A6F5/entry)

Embed and validate the exact GroEL sequence in the implementation; do not
fetch it at benchmark runtime.

### Controls and candidate layouts

All shard candidates use 64 shards and explicit
`Z=domZ=65,984,053`. `HMMER CPUs × active shards` is the peak runnable slot
count inside the one search container.

| Case | Layout | HMMER CPUs | Active shards | Peak slots | Scientific role |
| --- | --- | ---: | ---: | ---: | --- |
| B0 | monolith | 8 | 1 | 8 | operational baseline; Z/domZ unset |
| B1 | monolith | 8 | 1 | 8 | oracle; explicit Z/domZ |
| S0 | 64 shards | 8 | 1 | 8 | sharding overhead at baseline slots |
| S1 | 64 shards | 2 | 4 | 8 | more processes at baseline slots |
| S2 | 64 shards | 2 | 8 | 16 | moderate parallelism |
| S3 | 64 shards | 2 | 16 | 32 | exact upstream example shape |
| S4 | 64 shards | 4 | 8 | 32 | balanced 32-slot alternative |
| S5 | 64 shards | 8 | 4 | 32 | coarse 32-slot alternative |

B0 is descriptive and cannot be the correctness oracle because its database
scale behavior differs. B1 is the monolithic scientific oracle. S0 through S5
share the same scientific Search Identity but have distinct Benchmark Sample
IDs because CPU and active-shard settings are operational variables.

### Smoke mode

Run B0, B1, and S3 once with the pembrolizumab query. The smoke is uncounted.
It must pass artifact validation and the scientific gate before the measured
matrix is eligible.

Start with `cpu=(0.125, 32.125)`. Inspect the Resource Trace for runnable work,
CPU use, and cgroup throttling. High CPU utilization means the burst is being
used successfully. If demand exists but the container cannot burst reliably,
stop the campaign before the matrix. Do not automatically submit more jobs;
propose a separately approved CPU-floor diagnostic using request floors 4, 8,
and 16 with the same 32.125 limit. The eventual primary matrix must use one
consistent, lowest-stable floor.

### Matrix mode

The matrix is one invocation, not three invocations:

1. Run three measured screening blocks. Each block contains B0, B1, and S0-S5
   sequentially, never concurrently. Use a deterministic, different shuffled
   case order in each block.
2. Record whether every sample used a new or reused container.
3. Calculate median, range, and median absolute deviation. Do not report p95
   from three samples.
4. If a scientifically valid candidate varies by more than 10%, stop and ask
   before any extra samples. The previously discussed two interleaved samples
   are optional diagnostics, not pre-authorized work.
5. Select the two layouts that satisfy the scientific gate and the performance
   ranking rules below.
6. Run B0, B1, and those two layouts for three sequential, deterministically
   shuffled GroEL blocks. If screening results need human scientific review,
   stop before submitting the GroEL work.

At most 24 measured screening searches and 12 measured stress searches are
planned. Before matrix submission, present these counts and obtain explicit
permission. A conditional stop submits less work, never more.

## Timing and telemetry

Use three non-overlappingly named elapsed-time concepts:

- **Search Wall Time**: immediately before the pinned database query through
  completion of the merged A3M. This is the primary 20% speed gate.
- **Sample Wall Time**: remote function entry through durable evidence commit.
- **Remote Call Wall Time**: local submission through observed completion,
  including queueing and container startup.

Persist a one-second `trace.jsonl` beside each sample with feature-detected,
standard-library collection of:

- cgroup CPU usage and throttling;
- current and peak memory;
- runnable/load indicators when available;
- query, merge, and publish phase boundaries;
- per-shard start, finish, and duration;
- active shard count;
- child-process CPU time and affinity;
- task, region, provider, container, and new/reused-container metadata when
  available.

Also write `run.log` to the evidence Volume while continuing to emit useful
stdout/stderr to Modal logs.

## Scientific promotion gate

Retain the monolithic `tblout`, every raw shard `tblout`, the final merged A3M,
and a normalized hit table. Compare each sharded result with B1 using target
IDs, normalized aligned sequences, score, E-value, and deterministic ordering.

A candidate is scientifically valid only if:

- the first 100 unique hits are exact, or all hits are exact when the result is
  shallower than 100;
- every shared hit has the same score and E-value;
- at least 99% of the full truncated unique-MSA hit set overlaps;
- every remaining difference is characterized as a cross-shard duplicate and
  truncation-tail effect;
- no unexplained or top-100 difference exists.

Any other difference blocks promotion regardless of speed. B0 is reported but
does not weaken this gate.

## Performance ranking

Among scientifically valid layouts, advance:

1. the lowest median Search Wall Time; and
2. the lowest-cost layout whose median Search Wall Time is within 15% of the
   fastest.

A sharded layout counts as a meaningful success only if its median Search Wall
Time is at least 20% faster than B1, including HMMER process-launch and merge
overhead. Also report Sample and Remote Call Wall Time; reject a layout if
operational overhead erases the query-level improvement.

If no candidate clears the 20% gate, retain the evidence and stop. Do not
expand sharding to the other databases.

## Identity, paths, and resumability

Hash the canonical uppercase sequence. The first two hexadecimal characters
are only a directory-fanout prefix; the following component is the full hash:

```text
/{sequence-hash-prefix}/{sequence-hash}/raw-msa/
  small-bfd/{search-identity}/
    samples/{sample-id}/
      result.a3m
      result.tblout                 # monolith, when applicable
      shards/*.tblout               # sharded cases
      hits.parquet
      metrics.json
      trace.jsonl
      run.log
      done.json                     # written last
```

`search_identity` is a digest of result-affecting inputs: sequence, database
profile manifest, physical scientific layout, AlphaFold commit, HMMER build,
Z/domZ, iterations, E-value thresholds, sequence limit, and filtering. It does
not include CPU request/limit, HMMER thread count, active-shard count, Modal
placement, or repetition number.

`sample_id` is a human-readable measurement ID such as
`screen-S3-block-01`; it is not a cache identity.

A sample is reusable only when `done.json` matches the expected identity and
artifact sizes/digests. A missing or invalid marker causes recomputation. A
valid marker causes an accidental rerun to validate and skip the sample without
submitting work.

Benchmark samples never write canonical artifacts at the Search Identity root.
The future production cache may use:

```text
/{prefix}/{hash}/raw-msa/{database-id}/{search-identity}/result.a3m
/{prefix}/{hash}/raw-msa/{database-id}/{search-identity}/done.json
```

only after a strategy is promoted in a separate implementation.

Campaign-wide data has no synthetic sequence hash:

```text
/benchmarks/small-bfd-phase1-v1/
  plan.json
  results.parquet
  summary.md
  storage-scans/...
```

`plan.json` is immutable. `results.parquet`, written with Polars, has one row
per scan/search sample and references raw sample paths. `summary.md` reports
medians, variability, scientific and performance gates, and candidate ranks.
Large artifacts are never copied into the campaign directory.

## Failure and publication behavior

- Each worker owns one exclusive output directory.
- Write temporary artifacts first, validate them, then publish `done.json`
  last.
- Preserve successful completed samples when another sample fails.
- Do not publish a partial campaign summary as complete.
- Do not add retries that can silently create extra paid executions. Report the
  failed case and ask before another submission.
- Reject reuse when the immutable campaign plan or profile manifest differs.
- Do not build a scheduler, lease system, ledger, generic cache service, or
  attempt-history database for this one-off campaign.

## Implementation shape

Keep all benchmark-specific code in `alphafold3_msa_app.py`. Prefer plain
dataclasses or typed dictionaries for the small fixed case definitions; avoid
a generic benchmark framework.

Expose three separately invoked, plan-first operations:

1. prepare and validate the small-BFD profile;
2. execute the Volume scan matrix;
3. execute search mode `smoke` or `matrix`.

Remote helpers may implement one shard-build, scan worker, or isolated search
sample. Orchestration remains local so it can list the exact work before
submission and measure Remote Call Wall Time. Search cases run sequentially;
only the explicitly defined multi-container Volume scans fan out.

## Local verification before any submission

The uncommitted focused test file should exercise without Modal compute:

- sequence canonicalization and fixed SHA-256 values;
- Search Identity inclusion/exclusion rules;
- safe path construction and sample IDs;
- exact case tables and deterministic block order;
- shard filename expansion;
- manifest publication and validation using a tiny FASTA fixture;
- record/residue conservation and balance calculations;
- `done.json` validation and no-op resume behavior;
- normalized hit comparison, top-100 gate, and duplicate-tail classification;
- summary/ranking logic;
- cgroup telemetry feature detection.

Then run formatting/static checks and app-discovery/help smoke tests required by
the Biomodals app standards. These checks may build local environments but may
not submit Modal functions. Show the final dry-run plans and exact expected job
counts before requesting permission for preparation, scans, smoke, or matrix.

## Deferred Phase 2 design

Only after Phase 1 succeeds, use its data to choose per-database shard and CPU
layouts. The intended comparison is one container versus database-per-container
fanout at equal measured resource budget, with an optional scaled-out case.

Fanout is per unique protein sequence, not blindly per chain: homomeric chains
share one search result and caller-provided MSAs bypass search. Each database
worker writes A3M, `tblout`, metrics, and a completion marker to an exclusive
Volume path and returns only identifiers, paths, and digests. A coordinator
waits for all branches and combines results in AlphaFold's fixed database order,
never completion order. Failed branches preserve successful evidence but never
produce a partial combined MSA.

The exact worker counts, global sequence concurrency, merge/template
integration, and production cache publication are decisions for the Phase 1
review. No Phase 2 code is part of this implementation.

## Retirement

After a separately approved production migration, the temporary app can be
deleted. Keep the immutable database profile and campaign evidence until the
production change is verified and its rollback window has closed. Deleting or
mutating either Modal Volume is never implied by retiring the app.

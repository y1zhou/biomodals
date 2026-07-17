# AlphaFold 3 MSA database storage and staging scenarios

Status: research and benchmark design, 2026-07-16

Scope: storage, compression, sharding, staging, and reuse options for the
AlphaFold 3 data pipeline in
[`alphafold3_app.py`](../../src/biomodals/app/fold/alphafold3_app.py). This note
does not change production code or recommend a production default before
measurement.

Source-backed behavior in “Evidence and constraints” is treated as verified.
Scenario rankings, bottleneck predictions, and statements about the likely
winner are hypotheses to test. The catalogue marks modes that need an upstream
or wrapper patch instead of presenting them as currently supported.

## Direct answer

Storing the source databases compressed **can** improve cold-start transfer
time, but asking SeqKit to decompress, shuffle, and split a monolithic archive
inside every prediction worker is almost certainly slower end to end. Every MSA
search scans every database shard. Runtime splitting therefore adds a complete
database read, decompression, rewrite, and shard-validation pass before HMMER
can begin.

SeqKit belongs in a one-time ingestion job. The strongest compressed design to
test is:

1. Shuffle and split each database once, offline, into the exact AF3 shard
   layout.
2. Store every shard as an independent compressed object, with `.zst` and `.gz`
   as the first two codec candidates.
3. Fetch and decompress shards concurrently to a worker's local SSD, or to RAM
   only when the memory budget is demonstrably safe.
4. Reuse that staged layout for several MSA queries in the same warm worker so
   the transfer/decompression cost is amortized.

Stock Jackhmmer cannot search a gzipped target database directly, even when AF3
sets `-N 1`; it requires a normal rewindable sequence file. HMMER has no native
zstd target support. Consequently, compression is initially a **transport and
persistent-storage format**, while HMMER consumes decompressed FASTA shards on
SSD or RAM. Nhmmer has a narrower one-query streaming path that accepts gzip,
so direct gzip is a real RNA-only experiment. It is not a shared solution for
all seven databases.

The first benchmark set should be:

1. uncompressed sharded FASTA read directly from the existing Modal Volume;
2. uncompressed sharded FASTA read directly from a read-only CloudBucketMount;
3. uncompressed sharded FASTA copied once to local SSD;
4. independently zstd-compressed shards fetched and decompressed to local SSD;
5. independently gzip-compressed shards fetched and decompressed to local SSD;
6. the winning SSD layout reused for batches of 1, 2, 4, 8, and 16 queries;
7. per-database RAM-backed workers only if the simpler cases leave search I/O
   bound.

This ordering tests the claim that the Volume is too slow without assuming it.
Modal describes Volumes as cached, chunked, write-once/read-many distributed
filesystems, while CloudBucketMount is explicitly optimized for large
sequential reads. Neither statement predicts throughput for seven concurrent
HMMER scans, so the decision needs measurements on the actual database and
worker shape ([Volume documentation](https://modal.com/docs/guide/volumes),
[CloudBucketMount documentation](https://modal.com/docs/guide/cloud-bucket-mounts)).

## Evidence and constraints

### AF3 always searches all shards

The upstream performance guide says to randomly distribute records into
roughly balanced shards, names every shard using the zero-based, five-digit
`prefix-<index>-of-<total>` convention, and points AF3 at `prefix@N`. It also
says to place shards on fast SSD or a RAM-backed filesystem and tune HMMER CPUs,
active shards, and shard size together. AF3's example searches four protein
databases concurrently and three RNA databases concurrently
([performance guide](https://github.com/google-deepmind/alphafold3/blob/main/docs/performance.md#sharded-genetic-databases)).

The implementation expands an `@N` specification to a list containing **all**
`N` paths, starts one HMMER subprocess for each shard subject to the
`max_parallel_shards` bound, waits for every result, and then merges them
([shard expansion](https://github.com/google-deepmind/alphafold3/blob/main/src/alphafold3/data/tools/shards.py),
[Jackhmmer wrapper](https://github.com/google-deepmind/alphafold3/blob/main/src/alphafold3/data/tools/jackhmmer.py),
[Nhmmer wrapper](https://github.com/google-deepmind/alphafold3/blob/main/src/alphafold3/data/tools/nhmmer.py)).
More shards increase scheduling and merge overhead; they do not reduce total
database bytes scanned for one query.

The official database installer reports about 252 GB downloaded and about
630 GB after decompression for the complete AF3 reference set, including
template assets. It recommends SSD storage
([installation guide](https://github.com/google-deepmind/alphafold3/blob/main/docs/installation.md#obtaining-genetic-databases)).
The benchmark manifest must measure the seven MSA FASTAs separately rather than
using 630 GB as their assumed footprint.

### What HMMER can search directly

The following table is verified against the official HMMER 3.4 source and AF3
wrappers. “Wrapper change” means the current Biomodals image or upstream Python
wrapper cannot use the mode as-is.

| Target representation | Jackhmmer | Nhmmer | Practical conclusion |
| --- | --- | --- | --- |
| Plain FASTA file | Supported | Supported | Production-compatible AF3 input. |
| Plain FASTA shards | Supported by AF3 | Supported by AF3 | Production-compatible sharding input. |
| `.gz` FASTA | **Rejected** as target DB | Supported for the AF3 one-query invocation | Do not point protein paths at `.gz`; direct gzip is an RNA-only test. |
| `.zst`, `.xz`, `.bz2`, `.lz4` FASTA | Not natively supported | Not natively supported | Decompress before HMMER or add a supervised pipe/FIFO wrapper. |
| `.zip` archive/member | Not supported | Not supported | Extract members to normal files first; ZIP is not an AF3 database format. |
| BGZF plus index | Not used | Not used | Easel treats gzip as a pipe, not as a seekable indexed BGZF database. |
| Nhmmer `HMMERDB`/FM index | Not applicable | Supported only unsharded by the AF3 wrapper | Separate RNA-only experiment; not a sharded transport format. |

HMMER's Jackhmmer manual says the target `seqdb` cannot be stdin or gzip because
Jackhmmer needs multiple passes and treats gzip as an external decompression
pipe. The general HMMER guide repeats this exception
([Jackhmmer manual source](https://github.com/EddyRivasLab/hmmer/blob/hmmer-3.4/documentation/man/jackhmmer.man.in),
[compressed-input guide](https://github.com/EddyRivasLab/hmmer/blob/hmmer-3.4/documentation/userguide/formats.tex)).
The HMMER 3.4 code still rewinds the target after an iteration, and the
restriction is not conditional on `-N 1`
([Jackhmmer source](https://github.com/EddyRivasLab/hmmer/blob/hmmer-3.4/src/jackhmmer.c)).
AF3 requires `n_iter=1` for sharded Jackhmmer, but that AF3 constraint does not
remove HMMER's input restriction.

Nhmmer permits either the query or target database, but not both, to come from
stdin and supports gzip through Easel's external `gzip -dc` path. It cannot
rewind a streamed target when a query file contains more than one query
([Nhmmer manual source](https://github.com/EddyRivasLab/hmmer/blob/hmmer-3.4/documentation/man/nhmmer.man.in)).
The AF3 wrapper creates one query file per chain/shard invocation, so an
independently gzipped RNA shard layout is compatible with that one-query
constraint. It still needs an integration and scientific-equivalence test
before use.

AF3's shard grammar does support a suffix after `@N`, so a spec such as
`prefix@64.fa.gz` expands to `prefix-00000-of-00064.fa.gz` and so on. This only
names the files; it does not make Jackhmmer accept them
([`shards.py`](https://github.com/google-deepmind/alphafold3/blob/main/src/alphafold3/data/tools/shards.py)).

### What SeqKit contributes

SeqKit currently reads and writes gzip, xz, zstd, bzip2, and LZ4. Its gzip
reader/writer uses parallel gzip, and the global thread count defaults to four.
`split2 --by-part N` distributes records round-robin with low memory use and
can write compressed outputs. `shuffle --two-pass` is deterministic by default
and reduces memory consumption, but when its input is compressed it first
writes an uncompressed temporary FASTA and builds an index
([SeqKit input/compression documentation](https://bioinf.shenwei.me/seqkit/usage/#input-and-output-files),
[`split2` documentation](https://bioinf.shenwei.me/seqkit/usage/#split2),
[`shuffle` documentation](https://bioinf.shenwei.me/seqkit/usage/#shuffle)).

Consequences:

- SeqKit is well suited to the reproducible, one-time shard builder.
- A runtime `shuffle --two-pass` over compressed input needs scratch for an
  uncompressed temporary database, its index, the shuffled stream, and the
  final shards.
- `split2` output names do not match AF3's required zero-based padded convention
  by default, so ingestion must rename and validate them.
- SeqKit is not a machine scheduler. Modal decides which SSD/RAM worker receives
  data; SeqKit only transforms sequence files inside a worker.
- For lossless staging of already-created shards, `zstd -d` or `pigz -d` can
  preserve the exact decompressed bytes without parsing and rewriting FASTA.
  SeqKit is unnecessary in that hot path.

### Modal storage and lifecycle facts

- A Volume is persistent distributed storage with built-in caching and
  chunking. Modal says Volumes are designed for up to 2.5 GB/s, while warning
  that actual throughput is not guaranteed and may be lower with network
  conditions. Volume v2 supports more files and more irregular accesses, but
  is currently beta. The exact cache state is not user-controlled
  ([Volumes](https://modal.com/docs/guide/volumes)).
- CloudBucketMount supports S3, R2, and GCS and is based on Mountpoint. Modal
  says S3 mounts are optimized for reading large files sequentially, which
  matches HMMER's scan pattern, but they retain object-filesystem limitations
  ([Cloud bucket mounts](https://modal.com/docs/guide/cloud-bucket-mounts)).
- Every container has local SSD. Its default quota is 512 GiB, configurable up
  to 3 TiB through `ephemeral_disk`. Disk requests are billed by raising the
  memory request at a 20:1 ratio
  ([CPU, memory, and disk](https://modal.com/docs/guide/resources#disk-limits)).
- Modal's large-dataset guide estimates gzip decompression at about 80 MiB/s
  and recommends transforming compressed data once on local SSD before storing
  the runtime layout, rather than decompressing on every read. That is strong
  evidence against runtime monolith splitting. Compressed-per-shard staging is
  still worth measuring here only because it may reduce bytes transferred from
  a slow shared source and can be amortized across queries
  ([dataset ingestion](https://modal.com/docs/guide/dataset-ingestion#transforming)).
- Modal may reuse one Function container for multiple inputs. A cold container
  has new local state; subsequent calls may see earlier local files, but code
  must not depend on that reuse for correctness. `min_containers` and
  `scaledown_window` trade idle cost for warmer capacity
  ([cold starts](https://modal.com/docs/guide/cold-start),
  [troubleshooting local side effects](https://modal.com/docs/guide/troubleshooting)).
- A class `@modal.enter` hook can perform one staging operation when a container
  starts, before it accepts inputs
  ([lifecycle hooks](https://modal.com/docs/guide/lifecycle-functions)).
- Modal's configurable `scaledown_window` is bounded between two seconds and
  twenty minutes, and an overprovisioned container may still terminate sooner.
  Warm SSD reuse must therefore be observed, not assumed
  ([cold starts](https://modal.com/docs/guide/cold-start)).
- Function CPU Memory Snapshots save RAM state to disk, but Modal explicitly
  says snapshots generally do not improve initialization dominated by storage
  bandwidth. A Volume mutation also does not invalidate a snapshot
  ([Memory Snapshots](https://modal.com/docs/guide/memory-snapshots)).
- Sandbox filesystem/directory snapshots are Images with a default 30-day TTL
  and can make immutable local filesystem state reusable by other Sandboxes.
  This is a separate Sandbox architecture, not a drop-in Function Volume mount
  ([Sandbox snapshots](https://modal.com/docs/guide/sandbox-snapshots)).
- Modal says Image and Volume read performance is similar for model weights and
  recommends Volumes for flexibility. This does not prove equivalence for a
  630 GB database scan, but it makes an image-baked database a low-priority
  experiment rather than a presumed faster path
  ([model-weight storage](https://modal.com/docs/guide/model-weights#storing-weights-in-the-modal-image)).
- Modal runs containers across clouds and regions. A worker can be pinned near
  an external bucket, at an additional documented price multiplier. Bucket
  benchmarks must record worker location and compare unpinned with same-region
  placement rather than silently mixing regions
  ([region selection](https://modal.com/docs/guide/region-selection)).

The present app mounts the MSA Volume, optionally copies five monolithic files
to a unique local directory, requests only 0.125 CPU with a 32.125-core soft
limit, and has no active `ephemeral_disk` request. Its chain-parallel path reads
directly from the Volume. These are implementation facts, not measurements of
either backend's speed
([`run_data_pipeline`](../../src/biomodals/app/fold/alphafold3_app.py)).

## Scenario dimensions

Every realistic design is a point in this four-axis space. Keeping the axes
separate avoids conflating a compression win with a storage or warm-reuse win.

| Axis | Options to test |
| --- | --- |
| Persistent source | Volume v2; S3/GCS/R2 via CloudBucketMount; object API download; immutable Sandbox/Image snapshot; external service |
| Persistent representation | Uncompressed monolith; uncompressed shards; monolithic gzip/zstd archive; individually compressed gzip/zstd/LZ4 shards; tar/ZIP of shards; RNA HMMERDB |
| Search target | Source mount directly; local ephemeral SSD; tmpfs/RAM; remote per-database worker |
| Reuse policy | New stage per query; reuse in a warm container; fixed warm pool; persistent external database service |

Compression can save persistent bytes and source/network bytes. It does not
save the uncompressed local capacity required by stock HMMER. Sharding adds
parallel search opportunities. It does not reduce the total logical database
scanned. Warm reuse amortizes transfer and decompression. It does not guarantee
that every autoscaled container is already staged.

## Scenario catalogue

“Value” is a hypothesis to prioritize tests, not a measured result. “Direct”
means HMMER reads the persistent representation without a full local copy.

### Baselines and direct shared-storage searches

| ID | Dataflow | Directly feasible now? | Likely bottleneck | Persistence/reuse | Expected value |
| --- | --- | --- | --- | --- | --- |
| B0 | Volume, uncompressed monolith → HMMER | Yes; current chain-parallel path | Single-file remote scan and limited HMMER parallelism | Persistent source; reread per query | Required baseline, low expected performance |
| B1 | Volume, uncompressed monolith → copy to SSD → HMMER | Yes; partial current path | Full copy before search; 512 GiB default quota | SSD may survive warm calls but is not durable | Required current staging baseline |
| D1 | Volume v2, uncompressed shards → sharded HMMER | Yes after AF3 flag/profile work | Volume throughput under many sequential readers; merge overhead | Persistent source; Volume cache may help warm reads | **Highest-priority test** |
| D2 | CloudBucketMount, uncompressed shards → sharded HMMER | Yes | Bucket/mount throughput, request concurrency, region placement | Persistent object source; mount cache unspecified | **High-priority test** |
| D3 | Object API range/whole-object reads exposed through custom FUSE → HMMER | Not without custom filesystem | FUSE/object request overhead and cache correctness | Depends on custom cache | Low priority; bucket mount already supplies this class of behavior |
| D4 | Volume/bucket, gzip RNA shards → Nhmmer | Feasible with AF3's suffix grammar and one-query invocation | External gzip decode CPU plus remote read | No local stage; decompressed every query | Medium, RNA-only exploration |
| D5 | Volume/bucket, gzip protein shards → patched one-iteration Jackhmmer | HMMER and wrapper patch required | Decode CPU, pipe behavior, loss of upstream support | Decompressed every query | Low priority despite possible cold-byte savings |
| D6 | Volume/bucket, zstd/gzip shards → FIFO/stdin → Nhmmer | Custom decompressor and wrapper; Jackhmmer cannot use stock stdin target | Pipe backpressure, decompression CPU, failure cleanup | Decompressed every query | Low-priority RNA prototype |
| D7 | Volume/bucket, `.zip`/BGZF shards → HMMER | No native support | Extraction/custom random access | None natively | Do not benchmark until higher-value options fail |
| D8 | One read-only Volume per database → sharded HMMER | Feasible mount layout | Whether bandwidth/cache isolation actually scales across Volumes is undocumented | Persistent source | Medium diagnostic if one Volume saturates |

Direct D1 and D2 are important because they have zero explicit staging tax.
Even if their raw scan bandwidth is lower than SSD, they can still win for
one-off, cold, or heavily autoscaled calls.

### Local SSD staging

| ID | Dataflow | Directly feasible now? | Likely bottleneck | Persistence/reuse | Expected value |
| --- | --- | --- | --- | --- | --- |
| S1 | Volume uncompressed shards → concurrent copy → SSD → HMMER | Yes with a staging plan | Source read plus local write; many independent copies | Reusable while container remains warm | **High** |
| S2 | Object API uncompressed shards → concurrent download → SSD → HMMER | Yes with credentials/downloader | Network and object request concurrency | Warm-container reuse | **High** |
| S3 | Volume gzip shards → concurrent decompress-to-SSD → HMMER | Yes; HMMER sees plain outputs | Volume compressed-byte read, gzip CPU, SSD write | Warm-container reuse | **High enough to test** |
| S4 | Object storage gzip shards → concurrent fetch/decompress → SSD → HMMER | Yes | Network, gzip CPU, SSD write | Warm-container reuse | **High enough to test** |
| S5 | Volume zstd shards → concurrent decompress-to-SSD → HMMER | Yes; `zstd` is already in the image | Volume read, zstd decode, SSD write | Warm-container reuse | **Strong candidate** |
| S6 | Object storage zstd shards → concurrent fetch/decompress → SSD → HMMER | Yes | Network, zstd decode, SSD write | Warm-container reuse | **Strong candidate** |
| S7 | Object storage LZ4 shards → concurrent fetch/decompress → SSD → HMMER | Requires installing decoder | More source bytes but lower decode work is plausible | Warm-container reuse | Medium; test after zstd/gzip |
| S8 | One gzip/zstd monolith → SSD → SeqKit shuffle/split → HMMER | Technically feasible with enough disk | Entire database transform and extra reads/writes before every cold search | Only warm-container reuse | **Low; ingestion-only design** |
| S9 | One tar.gz/tar.zst containing prebuilt shards → extract all to SSD → HMMER | Yes | One large transfer and mostly serial archive extraction; no per-shard fetch | Warm-container reuse | Medium-low; simple but weak parallelism |
| S10 | One ZIP containing prebuilt shards → parallel member extraction → SSD → HMMER | Custom extraction orchestration | Central-directory/range reads, decompression, SSD write | Warm-container reuse | Low; independent objects are simpler |
| S11 | Independently compressed shard batches → double-buffer SSD staging and search | AF3 wrapper/orchestrator change | Coordinating readiness while preserving all-shard merge | Reuses only current batch; lowers peak disk | Medium future option |

S3–S6 are the main answer to “parallel-friendly zipped files.” The unit of
parallelism should be an independently checksummed compressed shard, not one
huge archive. A single tar stream makes extraction and recovery coarser. ZIP
members are independently compressed and seekable in principle, but HMMER
cannot open them, all shards are needed anyway, and an object-per-shard layout
already gives independent fetch/decode without a custom archive reader.

Zstandard data consists of independent frames and supports streaming decode;
separate shard files make frame decompression independently schedulable
([Zstandard format](https://github.com/facebook/zstd/blob/dev/doc/zstd_compression_format.md),
[CLI documentation](https://github.com/facebook/zstd/blob/dev/programs/zstd.1.md)).
Whether zstd beats gzip here depends on compression ratio, available CPU,
source bandwidth, and SSD write speed; it is a benchmark hypothesis, not an
assumed fact.

### RAM and distributed-worker scenarios

| ID | Dataflow | Feasibility | Likely bottleneck | Persistence/reuse | Expected value |
| --- | --- | --- | --- | --- | --- |
| R1 | All databases → one container tmpfs → HMMER | Not with the app's current 128 GiB hard limit; full data plus HMMER exceeds it | RAM capacity and cold load | Warm-container only | Exclude from first round |
| R2 | Compressed active shard batch → decompress to tmpfs → search → evict | Requires AF3 staging/readiness changes | Repeated source transfer/decode, RAM pressure, pipeline coordination | Per-query/batch | Medium-low |
| R3 | One warm worker per database, database decompressed to RAM, coordinator merges | Major orchestration/upstream seam; resource availability must be confirmed | Cold fill, idle RAM cost, result transport | Strong reuse while workers live | **High upside, high complexity** |
| R4 | One warm worker per database, database on local SSD | Similar orchestration, lower RAM requirement | Cold SSD fill and worker churn | Warm reuse | High-upside phase two |
| R5 | Worker per shard batch, stage batch to RAM/SSD, return raw HMMER result | Requires horizontal AF3 fanout and deterministic merge | Function count, scheduling, transfer duplication, merge RAM | Optional warm shard affinity | Medium future option |
| R6 | One function per shard | Technically possible but potentially hundreds/thousands of calls per chain | Scheduler and RPC overhead, output fan-in | Little reuse without affinity | Low; batch shards per worker instead |

R3 resembles the topology described by an AF3 maintainer for AlphaFold Server:
each database search runs on a separate machine and the databases are RAM-backed
([issue #618](https://github.com/google-deepmind/alphafold3/issues/618)). It is
therefore a useful upper-bound architecture, not evidence that the same latency
will occur on Modal. It also prevents four protein scans from competing for one
container's memory bandwidth.

Modal's documented multi-node cluster hardware has at least 1 TB RAM and 4 TB
local NVMe per node, but CPU-only clustered Functions are not supported; that
GPU-oriented beta product is not an appropriate shortcut for this data pipeline
([multi-node clusters](https://modal.com/docs/guide/multi-node-training)).
Standard CPU memory availability for R3 must be confirmed with Modal before
designing around it.

Ordinary i6pn-enabled Functions can communicate over Modal's region-scoped,
workspace-private network at a documented 50 Gbps or more, but peer discovery
is application-managed. This could support R3–R5, while adding a service
protocol and failure surface that the initial in-container sharding design does
not have
([cluster networking](https://modal.com/docs/guide/private-networking)).

Before any RAM trial, measure available `/dev/shm`, requested/observed memory,
and HMMER peak RSS on the chosen worker. Modal does not document a fixed maximum
memory size for standard CPU Functions; schedulability is checked when the
Function is created. A tmpfs allocation also subtracts memory available for
deep-MSA outputs and shard-result merging.

### Persistence and snapshot variants

| ID | Dataflow | Feasibility | Main caveat | Expected value |
| --- | --- | --- | --- | --- |
| W1 | `@modal.enter` fetch/decompress to SSD, then serve many method calls | Supported class lifecycle | Replacement/autoscaling creates new cold stages; must be idempotent | **High if query reuse exists** |
| W2 | W1 plus `min_containers=1+` | Supported | Pays for idle worker resources and each replica needs its own copy | High for steady workloads; poor for sporadic ones |
| W3 | Longer `scaledown_window` to retain staged SSD | Supported up to documented limit | No guarantee a container survives; billed while idle | Cheap benchmark of amortization |
| W4 | Bake database into a normal Modal Image | Conceptually possible but a roughly hundreds-of-GB image is operationally unsuitable without Modal validation | Image build, distribution, invalidation, duplicated storage | Very low |
| W5 | Build a Sandbox filesystem/directory snapshot containing shards | Supported for Sandboxes; snapshot is an Image | New Sandbox execution architecture, TTL/refresh and huge snapshot behavior | Low experimental |
| W6 | CPU Memory Snapshot after loading databases into RAM | Function feature exists | Current 128 GiB cap, huge snapshot, and storage-bound restore does not improve | Exclude |
| W7 | Long-lived external VM/service with NVMe/RAM DB | Outside current serverless app but feasible | Operations, networking, availability, security, cost | Valid reference/upper-bound test |

Local SSD and RAM are accelerators, not durable sources. A manifest-validated
Volume or object store remains the rebuild authority. Warm reuse is a latency
optimization; correctness must tolerate a fresh container at any time.

### Alternative database encodings

| ID | Design | Feasibility | Scientific/operational risk | Priority |
| --- | --- | --- | --- | --- |
| A1 | Build Nhmmer HMMERDB/FM-index databases and search them unsharded | AF3 wrapper calls this experimental; HMMER reports roughly 10× acceleration with some sensitivity loss; sharded HMMERDB is rejected | RNA-only scientific change needing oracle comparison; build/storage lifecycle | Medium RNA experiment |
| A2 | Replace `jackhmmer -N 1` with `phmmer` so compressed protein streams are accepted | HMMER manual says `-N 1` is equivalent at the search level, but AF3 depends on patched Jackhmmer flags and exact outputs | Upstream patch, `--seq_limit`, command/output equivalence and cache identity | Low research prototype |
| A3 | Patch Jackhmmer to permit a non-rewindable target only for one query and `-N 1` | Narrow C patch is possible | Maintaining HMMER fork; error-prone guard and oracle requirement | Low research prototype |
| A4 | Custom seekable compressed FASTA/FUSE layer | Substantial new filesystem code | Correctness, latency, page cache, failure recovery | Last resort |
| A5 | Add an SSI index beside plain FASTA | HMMER/Easel supports indexed sequence fetches, but AF3 performs full scans | No expected scan reduction unless the search algorithm changes | Negative/low-priority control |
| A6 | Run `hmmpress` on the MSA databases | Not applicable: `hmmpress` indexes profile-HMM databases, not FASTA sequence targets | Wrong database type for Jackhmmer/Nhmmer target scans | Exclude |

AF3's Nhmmer wrapper documents FASTA as slow and HMMERDB as experimental and
approximately 10× faster, while explicitly rejecting HMMERDB in sharded mode
([`nhmmer.py`](https://github.com/google-deepmind/alphafold3/blob/main/src/alphafold3/data/tools/nhmmer.py)).
The HMMER 3.4 `makehmmerdb` manual reports roughly 10-fold acceleration with a
small loss of sensitivity on its benchmarks
([manual source](https://github.com/EddyRivasLab/hmmer/blob/hmmer-3.4/documentation/man/makehmmerdb.man.in)).
A1 should therefore be measured separately from FASTA sharding because it
changes database representation, search execution, and scientific sensitivity.

## Recommended stored layouts

Maintain one immutable manifest per layout and keep legacy monoliths until
rollout is complete:

```text
/layouts/<profile-id>/
  manifest.json
  plain/
    uniref90_2022_05.fa-00000-of-00128
    ...
  zstd/
    uniref90_2022_05.fa-00000-of-00128.zst
    ...
  gzip/
    uniref90_2022_05.fa-00000-of-00128.gz
    ...
```

The benchmark does not need all three persistent copies indefinitely. Create a
representative subset first, select the winner, then materialize the full
layout. The manifest should record:

- logical database and source snapshot;
- decompressed source digest, byte count, sequence count, and residue count;
- AF3 shard prefix/count and exact filename list;
- decompressed digest and size for every shard;
- compressed digest, size, codec, codec version, and codec options;
- SeqKit version, random seed, command, and temporary-space policy;
- full-database Z value and unit;
- exact AF3 and HMMER commits/patch identities;
- manifest schema version and publication time.

Publish compressed/plain shard files first and `manifest.json` last. A staging
worker should fetch into a unique temporary directory, verify compressed and
decompressed digests, close files, then atomically publish a local readiness
marker. Never infer readiness merely from a directory's existence.

Do not store only a single monolithic gzip as the long-term “optimized” layout.
It is useful as a compact rebuild source, but it prevents independent shard
fetches and forces runtime transformation unless a pre-sharded layout also
exists.

## Break-even model

For `N` searches handled by one staged worker, define:

- `T_stage`: read compressed/plain source, decode if needed, write local files,
  and verify the layout;
- `T_local_search`: one complete sharded search from local SSD/RAM;
- `T_remote_search`: one complete sharded search directly from Volume or bucket.

Staging wins on latency when:

```text
T_stage + N × T_local_search < N × T_remote_search
```

or equivalently:

```text
N > T_stage / (T_remote_search - T_local_search)
```

This formula is valid only when local search is faster. It must be evaluated per
database profile, worker shape, chain type, and cold/warm source state. Report
the measured break-even `N`, not only the fastest warm search. Benchmark
`N = 1, 2, 4, 8, 16` for S1–S6 and W1.

For direct compressed streaming, use a different decomposition:

```text
T_stream_search = max(T_compressed_read, T_decode, T_hmmer_scan)
                  + pipeline/merge overhead
```

The terms overlap through a pipe, so adding their isolated times overestimates
end-to-end latency. Measure both isolated ceilings and the integrated pipeline.

## Benchmark matrix

### Stage 0: validate the benchmark, not performance

Before paid full-database tests:

1. Build a small deterministic shard fixture using the same manifest schema.
2. Verify exact AF3 shard names, record/residue conservation, Z units, and
   compressed/decompressed digests.
3. Verify that stock Jackhmmer rejects gzip and accepts the decompressed file.
4. For D4, verify one-query Nhmmer gzip behavior and compare raw tblout/A3M
   against the decompressed oracle.
5. Ensure benchmark runs bypass both reads and writes in the existing
   sequence-only MSA cache.

### Stage 1: storage and codec microbenchmarks

Run each row in a new container and a repeated invocation of the same container.
Use one representative shard from small BFD, MGnify/UniProt, NT-RNA, and Rfam.

| Experiment | Concurrency | Measure independently |
| --- | --- | --- |
| Volume plain read → `/dev/null` | 1, 2, 4, 8, 16 files | source bytes/s, CPU, wall time |
| CloudBucketMount plain read → `/dev/null` | 1, 2, 4, 8, 16 files | source bytes/s, first-byte latency, wall time |
| Object API plain download → `/dev/null` | 1, 2, 4, 8, 16 files | network bytes/s, request latency/errors |
| Local SSD plain read → `/dev/null` | 1, 2, 4, 8, 16 files | local read ceiling |
| Volume/bucket copy → SSD | 1, 2, 4, 8, 16 files | source read, local write, end-to-end copy |
| Native object SDK multipart download → SSD | 1, 2, 4, 8, 16 files | compare against CloudBucketMount/FUSE copy |
| Local gzip → `/dev/null` | per-file workers 1, 2, 4, 8 | decode bytes/s and CPU |
| Local zstd → `/dev/null` | per-file workers 1, 2, 4, 8 | decode bytes/s and CPU |
| Local LZ4 → `/dev/null` | per-file workers 1, 2, 4, 8 | decode bytes/s and CPU |
| Volume/bucket compressed → decode → SSD | per-file workers 1, 2, 4, 8 | read, decode, write, and overlap |
| SeqKit monolith shuffle/split | one representative full DB | read, temp write, shuffle, split, output write, peak disk/RAM |

Use a fresh output directory for every SSD write test. Do not claim a “cold
Volume cache”: a new Modal container is observable, but the distributed
backend's cache state is not directly controlled. Record container identity and
label only `new-container` versus `same-container`.

### Stage 2: HMMER-only search

Hold decompressed shard bytes, Z/domZ, total shard counts, HMMER CPU count,
active shard count, query, and result limits constant. Compare only the source
path:

| Case | Source | Stage time included? | Search timing |
| --- | --- | --- | --- |
| H0 | Volume plain shards | No explicit stage | Per DB, per shard, merge, total |
| H1 | CloudBucketMount plain shards | No explicit stage | Same |
| H2 | SSD plain shards copied beforehand | Report separately | Same |
| H3 | RAM plain shards loaded beforehand | Report separately | Same |

Start with the same 32-core grid proposed for sharding:

| HMMER CPUs per shard | Active shards per database | Peak protein slots |
| ---: | ---: | ---: |
| 8 | 1 | 32 |
| 4 | 2 | 32 |
| 2 | 4 | 32 |
| 1 | 8 | 32 |

This separates storage throughput from CPU-allocation effects. Repeat plausible
configurations at least three times, interleaved rather than running all samples
from one backend consecutively.

### Stage 3: end-to-end one-query and warm-batch tests

| Scenario | Cold stage | Searches per staged worker | Required result |
| --- | --- | ---: | --- |
| D1 Volume plain direct | None | 1, 2, 4, 8, 16 | End-to-end latency and repeated-read curve |
| D2 bucket mount plain direct | None | 1, 2, 4, 8, 16 | Same |
| S1 Volume plain → SSD | Copy once | 1, 2, 4, 8, 16 | Stage/search split and break-even |
| S2 object plain → SSD | Download once | 1, 2, 4, 8, 16 | Same |
| S3/S4 gzip shards → SSD | Fetch/decode once | 1, 2, 4, 8, 16 | Same plus codec CPU |
| S5/S6 zstd shards → SSD | Fetch/decode once | 1, 2, 4, 8, 16 | Same plus codec CPU |
| W1 winning format in warm class | `@modal.enter` | realistic arrival sequence | Cold fraction, replacement frequency, cost |

Run the ladder first on small BFD with the other protein databases held at the
same control layout, then all four protein databases, then one RNA database,
then all three RNA databases. Finally use a protein monomer, deep/repetitive
protein, two-chain heteromer, RNA, and mixed protein–RNA fixture.

### Metrics and provenance

Each sample should record:

- manifest digest, decompressed shard digests, source backend/region, codec and
  compressed bytes;
- worker cloud/provider/region, bucket region, Volume identity, and whether one
  or separate per-database Volumes were mounted;
- exact AF3 commit, HMMER build/patch, SeqKit/codec versions, and complete argv;
- Modal call/container identity and new-container/same-container label;
- requested CPU, observed CPU, peak RSS, local disk bytes, and local read/write
  throughput;
- source transfer first-byte time, bytes, duration, retry/error counts, and
  aggregate throughput;
- decompression CPU time, wall time, input/output bytes, and worker count;
- verification and local publication time;
- per-database and per-shard HMMER durations, merge duration, MSA depths, and
  total data-pipeline time;
- stage-only, search-only, and uncached end-to-end latency;
- cost per query for `N = 1, 2, 4, 8, 16`, including warm idle cost.

Do not compare a cached MSA result with an uncached database search. Do not omit
staging from the cold end-to-end number. Report warm search separately rather
than using it as the headline latency.

## Scientific implications

Lossless compression and decompression should preserve the decompressed shard
bytes. Raw-codec staging can therefore be tested as result-neutral by comparing
digests before HMMER. SeqKit may rewrap FASTA records; in that case compare
record IDs and sequences as well as downstream evidence.

Sharding itself is not byte-identical to the monolithic search. AF3 warns that
HMMER deduplication cannot detect duplicates across shards, usually producing a
small number of extra low-ranked hits, especially for deep MSAs. Full-database
Z values—and Jackhmmer domZ—must be the same across storage scenarios
([Jackhmmer warning](https://github.com/google-deepmind/alphafold3/blob/main/src/alphafold3/data/tools/jackhmmer.py),
[issue #492](https://github.com/google-deepmind/alphafold3/issues/492)).

For D1/D2/S1–S7, use the **same decompressed shard digests and identical AF3
flags**. Compare:

- raw per-database tblout hit IDs, coordinates, scores, and E-values;
- normalized unpaired/paired A3M records and ordering;
- MSA depths and templates;
- semantic data-pipeline JSON;
- failures, truncation, and tie behavior.

Direct compressed paths that require HMMER or wrapper patches—D5–D6 and
A2–A3—need a separately pinned external oracle and a new cache identity. D4
should preserve the decompressed RNA FASTA stream, but still needs raw-evidence
comparison against plain shards. Do not infer equivalence solely from final
prediction scores.

The existing sequence-only MSA cache must be bypassed for every benchmark and
must not receive benchmark results. A future production cache identity needs at
least the AF3/HMMER behavior identity, database manifest digest, decompressed
shard layout, and Z/domZ mode. Storage backend, codec, and staging concurrency
may be excluded only after tests prove they do not alter decompressed inputs or
search behavior.

## Recommended testing order and stop rules

1. Create one immutable sharded small-BFD subset in plain, gzip-per-shard, and
   zstd-per-shard form.
2. Run Stage 1 microbenchmarks against Volume and one colocated object bucket.
3. Run D1, D2, S1/S2, and S3–S6 for one protein query with cache bypass.
4. Calculate the observed warm-batch break-even for each SSD candidate.
5. Materialize all four protein databases only for candidates that beat direct
   Volume end to end or have a realistic warm-batch break-even.
6. Add RNA and mixed fixtures; test D4/HMMERDB only as separate RNA experiments.
7. Test W1/W2 only when expected query arrival rate justifies warm-stage cost.
8. Consider R3/R4 database-per-worker fanout only if SSD search remains I/O
   bound and the simpler architecture cannot meet the latency target.

Stop expanding a compression candidate when any of these holds:

- `T_local_search >= T_remote_search`, so staging has no positive break-even;
- its measured break-even batch is larger than realistic warm-container reuse;
- decode consumes CPUs needed by concurrent HMMER and increases end-to-end
  latency;
- local write bandwidth, not source transfer, dominates staging;
- peak disk or RAM leaves insufficient headroom for deep-MSA merge memory;
- cost per uncached query is worse without a compensating latency target;
- decompressed digests or scientific evidence differ unexpectedly.

## Recommended interpretation

The most plausible winner for sporadic, one-off requests is uncompressed
sharded direct search from whichever shared backend—Volume v2 or
CloudBucketMount—delivers the higher concurrent sequential throughput. The most
plausible winner for repeated queries is independently zstd- or gzip-compressed
shards staged once to SSD and reused by a warm worker. Per-database RAM workers
offer the highest theoretical search throughput but also the largest
orchestration and idle-resource cost.

Those are hypotheses. The benchmark must keep shard bytes, AF3 flags, CPU
budget, and query fixed and separately report source transfer, decompression,
local write, HMMER search, merge, and full end-to-end time. That separation is
what will tell us whether the Volume, decompressor, SSD, CPU budget, memory
bandwidth, or HMMER merge is the actual limiting resource.

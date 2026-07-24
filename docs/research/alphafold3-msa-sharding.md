# AlphaFold 3 MSA database sharding for Biomodals

Status: research and design recommendation
Research date: 2026-07-14
Target: `src/biomodals/app/fold/alphafold3_app.py`

## Decision summary

MSA database sharding is usable with the AlphaFold 3 source commit already
pinned by Biomodals. The safest first implementation is to add a versioned,
manifest-validated sharded database layout beside the existing monolithic
layout, pass explicit absolute shard specifications and database Z values to
`run_alphafold.py`, and bound shard parallelism to the CPU budget of one Modal
container.

Do not change the AlphaFold source pin merely because `CONF.version` says
`3.0.2`. Stock AlphaFold 3 v3.0.2 has incomplete sharding support, but the
Biomodals fork commit `987ad1c` is based on later upstream commit `5a3d6b6` and
contains the fixes for path validation, more robust merge ordering, RNA Z
types, and propagation of Z/domZ and maximum shard concurrency.

The first rollout should read shards directly from the existing Modal Volume.
Copying all shards to ephemeral SSD should be a separate measured experiment:
the current copy step is hundreds of GiB of cold-start work and could erase the
search-time gain. Keep the monolithic layout and legacy cache namespace intact
for a one-switch rollback.

The upstream “10–30x” figure is a commit claim, not a benchmark that can be
transferred to this app. AlphaFold Server also searches each database on a
separate machine with the database in RAM, a topology that the current
single-container pipeline does not reproduce. Biomodals therefore needs its
own cold/warm, end-to-end benchmark before selecting shard counts or promising
a speedup.

## What the upstream feature does

AlphaFold 3 accepts a shard specification such as `uniprot.fasta@64` and
expands it to files named:

```text
uniprot.fasta-00000-of-00064
uniprot.fasta-00001-of-00064
...
uniprot.fasta-00063-of-00064
```

The implementation supports `@N` and `@*`, with at most 99,999 shards. The
official preparation procedure is to randomize the FASTA records, split them
into roughly equal parts, and rename the outputs to zero-based, five-digit
indices. Randomization matters because equal record counts do not otherwise
imply equal residue counts or equal search time. See the
[performance guide](https://github.com/google-deepmind/alphafold3/blob/5a3d6b63656038fbb5285d405cd3389b190a5774/docs/performance.md#sharded-genetic-databases)
and the pinned-base
[shard parser](https://github.com/google-deepmind/alphafold3/blob/5a3d6b63656038fbb5285d405cd3389b190a5774/src/alphafold3/data/tools/shards.py#L15-L94).

For each database, AlphaFold creates an in-process Python thread pool. Each
worker launches one HMMER subprocess for one shard. After all shard searches
finish, AlphaFold merges the per-shard A3M and table outputs, sorts the hits,
and truncates the merged result to the configured sequence limit. This is
horizontal parallelism across HMMER processes, in addition to the threads used
inside each HMMER process. See the
[Jackhmmer query implementation](https://github.com/google-deepmind/alphafold3/blob/5a3d6b63656038fbb5285d405cd3389b190a5774/src/alphafold3/data/tools/jackhmmer.py#L90-L173)
and
[Nhmmer query implementation](https://github.com/google-deepmind/alphafold3/blob/5a3d6b63656038fbb5285d405cd3389b190a5774/src/alphafold3/data/tools/nhmmer.py#L94-L175).

AlphaFold already runs the four protein database searches concurrently. For
one protein chain, the approximate peak runnable CPU demand is therefore:

```text
4 databases × jackhmmer_n_cpu × jackhmmer_max_parallel_shards
```

For one RNA chain it is:

```text
3 databases × nhmmer_n_cpu × nhmmer_max_parallel_shards
```

The official example uses 2 HMMER CPUs and 16 simultaneous shards, which means
128 protein-search CPUs or 96 RNA-search CPUs per chain. It is an illustration
for a large machine, not a safe default for the current 32-core Modal limit.
The guide also recommends similar shard byte/residue sizes across databases,
not one common shard count.

If `*_max_parallel_shards` is omitted, the wrapper permits one worker for every
shard. With the published 512-shard MGnify layout, adding only an `@N` path to
the current 8-CPU command could create extreme CPU and memory pressure. Passing
both maximum-parallel flags is therefore a correctness-of-operation requirement,
not an optional tuning detail.

### Z values are correctness inputs

Searching a shard in isolation changes the database size HMMER sees. AlphaFold
must therefore receive the full, unsplit database size so that E-values remain
scaled to the original database:

- For a protein database, Z is the number of sequences in the complete FASTA.
- For an RNA database, Z is the total number of nucleotide bases divided by
  one million; it is a floating-point value, not the byte size of the file.

Issue [#557](https://github.com/google-deepmind/alphafold3/issues/557)
clarifies the RNA unit. At the compatible source pin, the protein pipeline
passes the configured Z as both HMMER `-Z` and `--domZ`; Nhmmer receives `-Z`.
The relevant construction and command generation are in the
[protein pipeline](https://github.com/google-deepmind/alphafold3/blob/5a3d6b63656038fbb5285d405cd3389b190a5774/src/alphafold3/data/pipeline.py#L297-L367),
[Jackhmmer wrapper](https://github.com/google-deepmind/alphafold3/blob/5a3d6b63656038fbb5285d405cd3389b190a5774/src/alphafold3/data/tools/jackhmmer.py#L230-L250),
and
[Nhmmer wrapper](https://github.com/google-deepmind/alphafold3/blob/5a3d6b63656038fbb5285d405cd3389b190a5774/src/alphafold3/data/tools/nhmmer.py#L209-L246).

The performance guide publishes the following reference configuration. The
specification column below deliberately uses this app's downloaded filenames,
not the two inconsistent example names in the guide:

| Database | Biomodals source and shard specification | Shards | Full-database Z |
| --- | --- | ---: | ---: |
| small BFD | `bfd-first_non_consensus_sequences.fasta@64` | 64 | 65,984,053 sequences |
| MGnify | `mgy_clusters_2022_05.fa@512` | 512 | 623,796,864 sequences |
| UniProt | `uniprot_all_2021_04.fa@256` | 256 | 225,619,586 sequences |
| UniRef90 | `uniref90_2022_05.fa@128` | 128 | 153,742,194 sequences |
| NTRNA | `nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta@256` | 256 | 76,752.808514 megabases |
| Rfam | `rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta@16` | 16 | 138.115553 megabases |
| RNAcentral | `rnacentral_active_seq_id_90_cov_80_linclust.fasta@64` | 64 | 13,271.415730 megabases |

These values are appropriate only for the exact published database snapshots.
The layout manifest should record measured counts and source identity; runtime
configuration should not rely on filenames alone.

## Why issue #492 changes the correctness model

Issue [#492](https://github.com/google-deepmind/alphafold3/issues/492) compared
the same protein–DNA target locally and on AlphaFold Server. A maintainer
reproduced a large confidence difference: Server ipTM/pTM 0.86/0.88 versus
local 0.10/0.78. The investigated Server MSA had depth 112 rather than 100,
with 12 additional small-BFD subsequences.

The official
[known-issues explanation](https://github.com/google-deepmind/alphafold3/blob/5a3d6b63656038fbb5285d405cd3389b190a5774/docs/known_issues.md#L21-L58)
says AlphaFold Server used sharded databases but historically omitted
`--domZ`. In some cases this made the conditional domain E-value threshold
effectively about 100 times more permissive, admitted extra MSA rows, and
changed the prediction. This is not evidence that sharding itself improves
prediction quality. It is evidence that database-scale parameters are part of
the scientific configuration and must be included in provenance and cache
identity.

The HMMER maintainer explains the underlying behavior in
[HMMER issue #340](https://github.com/EddyRivasLab/hmmer/issues/340#issuecomment-4613076104):
`incdomE` is based on conditional domain E-values, while the default domZ is
the number of reported hits. Repetitive false positives can make those values
too optimistic. Setting both Z and domZ to the full database sequence count is
the safer local-equivalence mode.

Biomodals should not omit domZ to imitate historical Server behavior in the
normal performance rollout. If Server-replication behavior is ever offered, it
must be an explicit scientific mode with a distinct cache namespace.

Even with correct Z/domZ, sharded and unsharded output is not guaranteed to be
byte-identical. The upstream Jackhmmer wrapper explicitly warns that duplicates
across different shards cannot be recognized during the individual searches,
so the merged sharded MSA can contain additional low-ranked hits, especially
for deep MSAs. Jackhmmer merges by E-value ascending, bit score descending, and
name; Nhmmer merges by E-value and name. See the
[cross-shard duplicate warning](https://github.com/google-deepmind/alphafold3/blob/5a3d6b63656038fbb5285d405cd3389b190a5774/src/alphafold3/data/tools/jackhmmer.py#L50-L61),
[Jackhmmer merge](https://github.com/google-deepmind/alphafold3/blob/5a3d6b63656038fbb5285d405cd3389b190a5774/src/alphafold3/data/tools/jackhmmer.py#L283-L335),
and the
[Nhmmer merge](https://github.com/google-deepmind/alphafold3/blob/5a3d6b63656038fbb5285d405cd3389b190a5774/src/alphafold3/data/tools/nhmmer.py#L303-L360).
Claims in [issue #566](https://github.com/google-deepmind/alphafold3/issues/566)
that the “same MSA” is obtained should therefore be read as intended scientific
equivalence, not byte-for-byte identity.

## Source-version compatibility

The feature's commit history matters more than the package version label:

| Date | Revision | Consequence |
| --- | --- | --- |
| 2025-10-29 | [`805adc3`](https://github.com/google-deepmind/alphafold3/commit/805adc3863841d83d631ccd18136ad58ce3ecb34) | Introduced database sharding and advertised 10–30x faster search. |
| 2025-11-25 | [`2e3703e`](https://github.com/google-deepmind/alphafold3/commit/2e3703e82a9592efbb3fa76ca9e0714aedabacdb) | Fixed shard-aware path validation reported in [#561](https://github.com/google-deepmind/alphafold3/issues/561). |
| 2026-02-16 | [`703d5c8`](https://github.com/google-deepmind/alphafold3/commit/703d5c8375c1559d8e6d4975c8564ecd01f5d02a) | Made Jackhmmer tie ordering more robust by adding bit score. |
| 2026-04-20 | [`v3.0.2`](https://github.com/google-deepmind/alphafold3/tree/f6a5aecf6eea2b5de4f846df86921630c5036b0b) | Still failed to forward Jackhmmer domZ/max-parallel and Nhmmer Z/max-parallel. |
| 2026-05-06 | [`97639ff`](https://github.com/google-deepmind/alphafold3/commit/97639fff6fb22c0d9765089026fe296ee506b60a) | Forwarded the missing settings and fixed sharded RNA failure [#663](https://github.com/google-deepmind/alphafold3/issues/663). |
| 2026-05-07 | [`eba6189`](https://github.com/google-deepmind/alphafold3/commit/eba618977e136d092ba4b986dd3fa541d2fd0241) | Corrected Nhmmer Z annotations to floating point. |

`alphafold3_app.py` pins fork commit
[`987ad1cb`](https://github.com/y1zhou/alphafold3/commit/987ad1cb7d7028b6d35908cf63fe7d951d98d6b6).
Its upstream base is
[`5a3d6b6`](https://github.com/google-deepmind/alphafold3/commit/5a3d6b63656038fbb5285d405cd3389b190a5774),
which descends from all of the fixes above. The four fork-only commits affect
model-pipeline caching, not the data/MSA pipeline. Therefore:

- keep `987ad1c` for the first sharding rollout;
- describe it as a “3.0.2-derived fork at `987ad1c`,” not stock v3.0.2;
- include the exact source commit in the database and MSA-cache provenance;
- add a regression test that the pinned source forwards Z/domZ and
  max-parallel settings before changing the pin later.

As of this research date, official `main` was at
[`0d3facb`](https://github.com/google-deepmind/alphafold3/commit/0d3facb93b7c09edcd2ae475c2dd7c283f43cc81)
(2026-07-13).
Reviewing the changes after `5a3d6b6` found no later local-filesystem sharding
semantic fix required by this design. Later cloud-object path support is not
needed for databases mounted from a Modal Volume. This is a bounded source
audit, not a reason to float the production image to `main`.

The `${DB_DIR}` compatibility path deserves special care. At the pinned base,
the placeholder-expansion branch can still test a literal `...@N` path before
shard expansion. Pass fully expanded absolute shard specifications such as
`/AlphaFold3-msa-db/layouts/<profile>/uniref90_2022_05.fa@128`, while retaining
`--db_dir` only as the fallback for unsharded PDB sequence/template data. See
the
[path resolution code](https://github.com/google-deepmind/alphafold3/blob/5a3d6b63656038fbb5285d405cd3389b190a5774/run_alphafold.py#L680-L696).

## Current Biomodals bottlenecks and constraints

The current data function has these relevant behaviors:

1. It mounts the v2 `AlphaFold3-msa-db` Modal Volume and an MSA cache Volume.
2. It requests only 0.125 CPU with a 32.125 CPU limit, then fixes both HMMER
   tools at 8 CPUs. Four simultaneous protein searches can therefore consume
   approximately the full limit even before shard parallelism.
3. With `copy_msa_to_ssd=True`, it copies five monolithic protein/PDB FASTAs to
   a unique temporary directory before searching. RNA databases and mmCIF data
   continue to use the Volume. No explicit ephemeral-disk allocation or cleanup
   lifecycle is configured.
4. In the default chain-parallel path it starts one Modal data-pipeline call per
   MSA-bearing chain with `copy_msa_to_ssd=False`. Unless callers set
   `max_parallel_data_pipelines`, all those containers may run concurrently.
5. The MSA cache key is only the sequence SHA-256. It does not encode the AF3
   commit, HMMER version/patch, database snapshot, Z/domZ, shard layout, or
   template snapshot. Inline caller-provided MSAs can also be written into this
   shared sequence namespace.
6. The cache writes independent files without a completion manifest, so a
   reader can observe an incomplete protein entry or mixed writers.

The relevant code is in
[`alphafold3_app.py`](../../src/biomodals/app/fold/alphafold3_app.py), especially
`_load_msa_cache`, `_save_msa_cache`, `run_data_pipeline`, and
`search_msa_and_templates`.

Sharding can fix under-utilization within each database search, but it does not
remove storage, memory-bandwidth, cache, or orchestration bottlenecks. Deep MSA
merges can require more than 64 GiB, and every shard's output must remain
available until merge/truncation. Total shard count therefore affects merge
memory and overhead even when maximum concurrent shards is small.

The public Server topology is a useful upper bound, not a direct comparison.
Issue [#618](https://github.com/google-deepmind/alphafold3/issues/618) says the
Server runs each database search on a separate machine and keeps each database
in RAM, eliminating both inter-database resource contention and slow storage.
Issue [#525](https://github.com/google-deepmind/alphafold3/issues/525) reports
diminishing returns beyond roughly 8 CPUs for one unsharded Jackhmmer process
and emphasizes local SSD/RAM. Issue
[#566](https://github.com/google-deepmind/alphafold3/issues/566) estimates that
a suitable sharded setup could reduce one 2 h 20 min run below 30 min, but also
says shard sizing depends on the database and hardware and must be benchmarked.

### Related issue/discussion synthesis

| Thread | Evidence from the conversation | Consequence here |
| --- | --- | --- |
| [#134](https://github.com/google-deepmind/alphafold3/issues/134) | Maintainers confirm the four protein searches are intentionally concurrent and expose per-HMMER CPU flags. | Budget database concurrency before multiplying it by shard concurrency. |
| [#525](https://github.com/google-deepmind/alphafold3/issues/525) | Reports weak scaling above about eight CPUs for one Jackhmmer process and a strong dependence on fast local storage. | Prefer more independently useful shard processes over ever-larger `--cpu` for one HMMER process. |
| [#566](https://github.com/google-deepmind/alphafold3/issues/566) | Maintainer guidance puts sequence databases, not PDB/template data, on SSD/RAM and says shard count is hardware/database dependent. | Shard the seven MSA databases, retain template fallback, and tune empirically. |
| [#557](https://github.com/google-deepmind/alphafold3/issues/557) | Clarifies that RNA Z is nucleotide bases in megabases, not file bytes. | Store a typed value and unit in the manifest. |
| [#561](https://github.com/google-deepmind/alphafold3/issues/561) | Early `@N` specifications failed literal path validation. | The current pin contains the fix, but command construction still needs an integration test. |
| [#610](https://github.com/google-deepmind/alphafold3/issues/610) | A user on an older checkout could not use the new flags. | Detect support from the exact commit/CLI, not the reported release string. |
| [#618](https://github.com/google-deepmind/alphafold3/issues/618) | Maintainer describes Server database-per-machine, RAM-backed execution. | Do not attribute Server latency entirely to file sharding. |
| [#663](https://github.com/google-deepmind/alphafold3/issues/663) | Sharded RNA failed because Nhmmer Z and maximum parallelism were not forwarded. | Pin ancestry must include `97639ff` and `eba6189`; this app's current pin does. |

The speed figures in these threads are maintainer/user observations for their
setups, not a public controlled benchmark over the Biomodals database snapshot,
Modal Volume, and container shape. They motivate the benchmark matrix below but
do not establish its expected result.

## Recommended design

### 1. Build an immutable database profile

Keep the existing flat FASTAs as `legacy-root`. Create a side-by-side layout,
for example:

```text
/AlphaFold3-msa-db/layouts/af3-2026-msa-v1/
  manifest.json
  bfd-first_non_consensus_sequences.fasta-00000-of-00064
  ...
  uniref90_2022_05.fa-00127-of-00128
  ...
```

Do the shuffle/split operation once in a dedicated setup job, not during a
prediction. Stock SeqKit `shuffle --two-pass` is deterministic, but its
full-header maps reached 123.54 GiB RSS for UniProt and its serialized random
reads produced only about 1.9 MB/s. It is not feasible for MGnify under the
128 GiB limit; see the
[source and runtime audit](alphafold3-seqkit-two-pass-shuffle-audit.md).

Use the pinned occurrence-indexed two-pass helper instead. Pass one scans the
source sequentially into compact fixed-width offsets while teeing the exact
bytes into container-local `/tmp`, then creates an explicit seed-23
Fisher--Yates permutation. Pass two uses bounded concurrent reads from the
local copy while writing in permutation order. This preserves duplicate
headers by occurrence. The source of record remains only on the source Modal
Volume; the local source copy, compact index, and shuffled FASTA are ephemeral.

The completed UniProt production-candidate build measured 638.43 MB/s for the
first pass and 120.57 MB/s for the full 108.45 GB second pass, versus about
1.94 MB/s for stock SeqKit's serialized random reads. It published all
225,619,586 records with 0.2804% maximum residue imbalance. Its remaining
performance bottleneck was the single-input aggregate `seqkit sum`: `-j 8`
parallelizes across files, but concatenating the shards into the one logical
input required for direct source comparison leaves one global hash sort. Use a
parallel, deliberately composable multiset validator for subsequent builds;
independent per-shard SeqKit digests cannot be substituted for the aggregate.

Production-candidate recipe v5 implements that validator. It hashes canonical
full `(header, sequence)` records with SHA-256 and reduces all four digest lanes
using commutative sum, XOR, and sum-of-squares accumulators plus exact record,
header-byte, and sequence-byte totals. The source is scanned sequentially and
shards are scanned concurrently. Matching signatures therefore validate the
order-independent, multiplicity-sensitive full-record multiset without
SeqKit's global sort, while independent `seqkit stats` remains as a second
record/residue conservation check. Existing recipe v4 profiles remain valid.

A read-only UniProt benchmark validated all 225,619,586 records and produced
identical full-record signatures for the monolith and 256 shards. The
single-file, single-thread source scan took 488.801 seconds at 221.865 MB/s;
the eight-thread shard scan took 113.268 seconds at 957.449 MB/s; combined
scanner time was 602.069 seconds. This is 3.73 times shorter than the
historical 2,242.917-second post-shard-statistics-to-completion window, although
that old window also included artifact hashing, publication, and deep
verification. The selected production-candidate recipe uses the C shuffler and
full-record C validator; SeqKit remains responsible for statistics and
splitting. Full evidence and caveats are in the
[shuffle audit](alphafold3-seqkit-two-pass-shuffle-audit.md#read-only-recipe-v5-validator-benchmark).

When this builder moves into the production AlphaFold3 app, its setup
entrypoint must first validate all seven fixed profile manifests, collect only
the missing database IDs, and submit one builder container per missing database
concurrently. Builders retain their per-Profile-ID claims and write only to
distinct source, staging, profile, and evidence paths. Existing valid profiles
are reused rather than rebuilt. The coordinator waits for every submitted
builder before performing the one final Volume inventory and cleanup check;
workers must not clean shared staging state while peers are still active.

`split2 --by-part` does not emit AF3's required zero-based padded names by
default, so rename and validate every output. See the official
[split2 documentation](https://bioinf.shenwei.me/seqkit/usage/#split2).

After all seven database profiles pass validation, remove abandoned staging
generations and the obsolete small-BFD benchmark profile so the sharded
Volume's `/profiles/` directory contains exactly one published directory per
genetic database. Run the protein and RNA oracle comparisons only after that
cleanup barrier.

The manifest should contain, for each logical database:

- source filename, snapshot/date, byte size, digest, record count, and residue
  count;
- shard prefix, shard count, required Z value and unit;
- expected filenames, sizes, and preferably digests;
- shuffle algorithm/seed, splitter and version;
- schema version and exact AlphaFold source compatibility commit.

Write into a unique staging directory, validate record/residue conservation and
all shard names, commit the Volume data, and publish `manifest.json` last as the
readiness marker. Never delete or replace the legacy files as part of the same
operation. Upstream's
[`fetch_databases.sh`](https://github.com/google-deepmind/alphafold3/blob/5a3d6b63656038fbb5285d405cd3389b190a5774/fetch_databases.sh)
downloads only monolithic FASTAs; there is no official prebuilt shard artifact
or shard-builder workflow to depend on.

The performance guide's UniProt and UniRef90 example names differ from the
official downloader and this app. Derive the runtime shard prefix from the
validated manifest and actual source filename (`uniprot_all_2021_04.fa` and
`uniref90_2022_05.fa` here), not by copying the prose example blindly.

### 2. Select a profile and construct explicit AF3 flags

Runtime should accept a trusted profile ID, validate its ready manifest, and
derive the seven database specifications and Z values. It should not expose
arbitrary user-provided paths or Z values. The resulting command needs:

```text
--small_bfd_database_path=<absolute-prefix>@<N>
--small_bfd_z_value=<manifest-value>
--mgnify_database_path=<absolute-prefix>@<N>
--mgnify_z_value=<manifest-value>
--uniprot_cluster_annot_database_path=<absolute-prefix>@<N>
--uniprot_cluster_annot_z_value=<manifest-value>
--uniref90_database_path=<absolute-prefix>@<N>
--uniref90_z_value=<manifest-value>
--ntrna_database_path=<absolute-prefix>@<N>
--ntrna_z_value=<manifest-value>
--rfam_database_path=<absolute-prefix>@<N>
--rfam_z_value=<manifest-value>
--rna_central_database_path=<absolute-prefix>@<N>
--rna_central_z_value=<manifest-value>
--jackhmmer_n_cpu=<tuned-value>
--jackhmmer_max_parallel_shards=<tuned-value>
--nhmmer_n_cpu=<tuned-value>
--nhmmer_max_parallel_shards=<tuned-value>
```

Keep PDB seqres and mmCIF paths unsharded through the existing `--db_dir`
fallback. This mixed mode is supported; the upstream sharding guide covers the
four protein MSA and three RNA MSA databases, not the template assets.

### 3. Bound nested parallelism

Treat HMMER threads and active shards as runtime configuration, not image
environment. Under the present 32-core container limit, benchmark at least:

| HMMER CPUs | Parallel shards | Peak protein slots | Peak RNA slots |
| ---: | ---: | ---: | ---: |
| 8 | 1 | 32 | 24 |
| 4 | 2 | 32 | 24 |
| 2 | 4 | 32 | 24 |
| 1 | 8 | 32 | 24 |

`2 × 4` is a reasonable initial sharded candidate, not a final default. It
adds cross-shard concurrency without creating more than 32 nominal protein
slots. Request close to the CPU capacity needed for predictable latency rather
than depending on a 0.125-core request and burst availability.

Also bound the number of chain-level Modal calls. Approximate run-wide demand
is:

```text
active chain containers × databases per chain × HMMER CPUs × parallel shards
```

Modal input concurrency is not an additional speed lever for a function whose
subprocess pool already saturates its resources. The Volume is a distributed,
write-once/read-many filesystem with lazy loading and caching, and concurrent
I/O can use more bandwidth, but actual throughput depends on the access
pattern. Measure it rather than assuming either Volume or local disk wins. See
the [Modal Volume guide](https://modal.com/docs/guide/volumes) and
[resource configuration](https://modal.com/docs/guide/resources).

### 4. Use direct Volume reads before SSD staging

Mount the validated database layout read-only in the prediction function and
benchmark it first. The existing v2 Volume is suitable for a large immutable
file set and concurrent readers.

Do not copy one file per shard with the current `copy_files` helper: a profile
can contain roughly a thousand files using the published shard counts, and
copying the full dataset adds a large fixed delay and disk requirement to every
cold container. Modal's default ephemeral disk is 512 GiB, while the official
AlphaFold database set is about 630 GB uncompressed; explicit disk allocation
and its cost would be required for a full local layout. See Modal's
[ephemeral-disk resource documentation](https://modal.com/docs/guide/resources#ephemeral-disk)
and AlphaFold's
[database installation note](https://github.com/google-deepmind/alphafold3/blob/5a3d6b63656038fbb5285d405cd3389b190a5774/docs/installation.md#obtaining-genetic-databases).

If direct Volume reads remain the bottleneck, test local staging as a second
profile. Stage only the databases required by the input, copy in bounded
batches, allocate disk explicitly, report staging time/bytes separately, and
amortize the staged data over enough queries to justify the cold cost.

### 5. Make the MSA cache scientifically versioned

Create a new cache namespace whose identity includes at least:

```text
cache schema
AlphaFold/wrapper exact commit
HMMER version and seq-limit patch identity
database manifest digest
Z/domZ mode
template/reference snapshot
sequence type and sequence digest
```

Shard count and layout should be included initially. Scheduling-only knobs such
as HMMER CPUs and maximum active shards may be removed from identity only after
the equivalence suite shows that they do not change normalized results. A cache
bypass/fresh namespace is mandatory for benchmarks; otherwise the existing
sequence-only cache will hide all database work.

Write each result into a unique staging directory and publish a small complete
manifest last. Readers should accept only complete entries. Prevent or safely
resolve same-key writers. Do not allow inline caller-provided MSAs to populate
the shared computed-MSA namespace.

## Benchmark and validation plan

Use an external, pinned unsharded AlphaFold invocation as the correctness
oracle. Calling the same Biomodals cache path in both arms is not an independent
comparison.

Test inputs should include:

- a shallow-MSA protein monomer;
- a deep-MSA protein that stresses merge/truncation and memory;
- a multichain protein input;
- RNA and mixed protein/RNA inputs;
- the issue #492 protein–DNA case, if its input can be reproduced;
- repeated sequences/chains to exercise cache and concurrent writers.

For every mode, run at least three cold and three warm repetitions with cache
bypass:

1. legacy monolithic databases read directly from the Volume;
2. the current legacy SSD-copy path;
3. sharded direct-Volume search at `4×2`, `2×4`, and `1×8`;
4. sharded SSD staging only if direct-Volume performance warrants it.

Record per-database and total wall time, staging time and bytes, shard timing
distribution, CPU utilization, peak RSS, temporary disk, Volume throughput,
Modal cost, cache read/write time, and cache payload size. Report p50 and p95,
not only the fastest observation.

Correctness comparison should cover:

- raw per-database hit/domain identifiers, coordinates, E-values, and scores;
- normalized unique A3M rows and their order, with duplicate differences called
  out separately;
- paired and unpaired MSA depths/content;
- template identities and features;
- semantic comparison of the generated `_data.json` after removing paths and
  other nondeterministic metadata;
- model scores for the #492 fixture as a downstream sentinel, not as the only
  equivalence test.

Suggested promotion gates:

- no missing/corrupt shard accepted by runtime validation;
- Z/domZ observed in the spawned HMMER commands for protein and RNA Z observed
  in Nhmmer;
- no unexplained top-ranked hit, pairing, template, or semantic feature change;
- duplicate-only differences are characterized and accepted explicitly;
- no OOM or excessive temporary-disk growth on the deep-MSA case;
- material p50 and p95 end-to-end improvement after including staging and cache
  overhead;
- one-switch rollback to legacy database and cache profiles succeeds.

## Implementation sequence

1. Add manifest schema/validation, immutable profile selection, and a dry-run
   command builder with unit tests. Do not change the default profile.
2. Add a one-off sharding/setup function with measured disk allocation,
   conservation checks, and publish-last semantics.
3. Add explicit AF3 database/Z/max-parallel flags behind an opt-in profile;
   retain the current source pin and monolithic fallback.
4. Introduce the provenance-aware cache namespace before any production A/B
   comparison.
5. Add command, budget, manifest, incomplete-cache, same-key-writer, protein,
   RNA, and orchestration tests; then run a small real-Modal integration test.
6. Run the benchmark matrix and choose shard counts, active-shard limits, CPU
   requests, chain fan-out, and storage topology from measured results.
7. Promote the sharded profile only after the scientific and performance gates
   pass. Keep legacy data/cache until rollback confidence is established.

Database-per-Modal-function fan-out could more closely resemble AlphaFold
Server and remove inter-database contention, but it would require an upstream
pipeline seam/patch and a separate equivalence exercise. It is a possible later
phase, not necessary to obtain the supported in-container sharding benefit.

## Risks and rollback

| Risk | Control |
| --- | --- |
| Incorrect Z unit or snapshot count | Manifest-derived values, independent recount, command-log assertion. |
| Stock v3.0.2 behavior accidentally restored | Pin exact commit and regression-test forwarding of Z/domZ/max-parallel. |
| Cross-shard duplicate or ordering changes | Compare normalized raw results; retain unsharded oracle and profile. |
| Excess CPU from nested database/shard/chain fan-out | Central budget formula and explicit chain-container bound. |
| Volume bandwidth limits scaling | Measure per-shard timing and throughput; evaluate bounded SSD staging second. |
| SSD staging dominates latency or exceeds disk | Direct-Volume default, explicit disk, required-database-only staging. |
| Deep-MSA merge OOM | Deep fixture, RSS telemetry, conservative total shard counts and concurrency. |
| Partial shard profile | Unique staging layout, full validation, publish manifest last. |
| Stale or contaminated MSA cache | New provenance namespace, completion marker, inline-MSA isolation, bypass mode. |
| Scientific regression | Explicit equivalence gate and one-switch legacy database/cache rollback. |

## Primary source index

- [AlphaFold 3 performance guide](https://github.com/google-deepmind/alphafold3/blob/5a3d6b63656038fbb5285d405cd3389b190a5774/docs/performance.md#sharded-genetic-databases)
- [AlphaFold 3 issue #492](https://github.com/google-deepmind/alphafold3/issues/492)
- [Known issue: AlphaFold Server MSA discrepancy](https://github.com/google-deepmind/alphafold3/blob/5a3d6b63656038fbb5285d405cd3389b190a5774/docs/known_issues.md#L21-L58)
- [HMMER issue #340 maintainer explanation](https://github.com/EddyRivasLab/hmmer/issues/340#issuecomment-4613076104)
- [AlphaFold 3 issue #566: storage and shard tuning](https://github.com/google-deepmind/alphafold3/issues/566)
- [AlphaFold 3 issue #618: Server topology](https://github.com/google-deepmind/alphafold3/issues/618)
- [AlphaFold 3 issue #557: RNA Z calculation](https://github.com/google-deepmind/alphafold3/issues/557)
- [AlphaFold 3 issue #561: shard-path validation](https://github.com/google-deepmind/alphafold3/issues/561)
- [AlphaFold 3 issue #610: unsupported older source](https://github.com/google-deepmind/alphafold3/issues/610)
- [AlphaFold 3 issue #663: RNA sharding failure](https://github.com/google-deepmind/alphafold3/issues/663)
- [AlphaFold 3 issue #525: unsharded CPU and storage limits](https://github.com/google-deepmind/alphafold3/issues/525)
- [AlphaFold 3 issue #134: database-level parallelism](https://github.com/google-deepmind/alphafold3/issues/134)
- [Modal Volumes](https://modal.com/docs/guide/volumes)
- [Modal resource configuration](https://modal.com/docs/guide/resources)
- [SeqKit usage](https://bioinf.shenwei.me/seqkit/usage/)

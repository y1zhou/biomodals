# AlphaFold3 sharded-MSA production integration plan

Status: accepted.

Scope: finish the sharding method in
`src/biomodals/app/fold/alphafold3_msa_app.py`, validate it, then integrate the
minimal production implementation into
`src/biomodals/app/fold/alphafold3_app.py`.

This plan implements
[ADR 0005](../adr/0005-alphafold3-msa-sharding.md). That consolidated decision
record remains authoritative when this document abbreviates a boundary.

## Outcome

The production AlphaFold3 app will:

1. build immutable, validated sharded profiles in the separate
   `AlphaFold3-msa-db-sharded` Modal Volume;
2. replace the upstream one-process-per-database data pipeline with one
   resumable worker per missing sequence-by-database result;
3. read uncompressed shards directly from the Modal Volume with the measured
   `2 HMMER CPUs × 16 active shards` topology;
4. preserve completed database and template searches across an explicit rerun;
5. construct a fully enriched AlphaFold input in Biomodals and send it directly
   to upstream inference with `--run_data_pipeline=false`;
6. persist canonical seed outputs in `AlphaFold3-outputs`, safely reuse
   overlapping seed requests, and return a request-scoped local archive.

The temporary MSA app remains an independent validation harness. Production
does not import it or its benchmark-only campaign machinery.

## Fixed evidence and limits

The small-BFD Phase 1 campaign established:

- 64 shards are scientifically equivalent to the monolith for the tested
  queries, allowing permutations only inside exact equal-score groups;
- 16 active shard searches with two HMMER CPUs each completed the medium query
  in about 49.05 seconds, approximately 5.1 times faster than the measured
  monolithic baseline;
- `(0.125, 32.125)` is the selected Modal CPU allocation;
- the measured search did not show Modal Volume bandwidth to be the primary
  bottleneck;
- copying a full database to ephemeral SSD is not justified.

The generic builder subsequently validated and published all seven official
profiles. The protein scientific oracle has passed across the four protein
databases. The first RNA fixture returned no hits and was rejected. A
hit-bearing Picea stress fixture then exposed a one-row saturated-cutoff
limitation in stock Nhmmer output. The subsequent non-saturating RAGATH-1 RNA
oracle passed all three databases and final MSA assembly. The integrated
pipeline gates below remain mandatory before the corresponding production
paths are considered complete.

### Scientific oracle evidence

The 120-residue pembrolizumab VH oracle passed on 2026-07-24. Every monolithic
and sharded database hit-row multiset was equal, and every ordering difference
was confined to an exact E-value/bit-score tie block. The final unpaired and
paired A3M record multisets were equal at depths 10,600 and 23,249,
respectively.

The database calls ran concurrently, so per-call times below are not summed
campaign elapsed times. Cost is a per-search estimate from one-second CPU
cgroup samples and the one-GiB requested-memory floor at the Modal rates
observed on 2026-07-24. It is useful for database-to-database comparison but is
not an invoice.

| Database | Monolith search | Sharded search | Speedup | Monolith CPU-s | Sharded CPU-s | Monolith estimated cost | Sharded estimated cost | Hit rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MGnify | 3,074.5 s | 412.0 s | 7.46x | 7,721 | 10,096 | $0.1080 | $0.1332 | 594 |
| small BFD | 247.8 s | 31.0 s | 7.99x | 774 | 914 | $0.0107 | $0.0120 | 142 |
| UniProt | 1,609.3 s | 605.2 s | 2.66x | 4,175 | 12,430 | $0.0583 | $0.1642 | 23,248 |
| UniRef90 | 1,326.6 s | 113.4 s | 11.70x | 3,280 | 3,195 | $0.0459 | $0.0421 | 9,999 |

The concurrent critical path fell from 3,074.5 seconds to 605.2 seconds, a
5.08-fold speedup. Summed successful-search estimates rose from $0.2229 to
$0.3515 because the low-latency topology uses more aggregate CPU, especially
for UniProt.

### Protein shard-count A/B

A one-shot follow-up on 2026-07-24 tested whether bringing the larger protein
shards closer to the roughly 270 MiB size of small BFD and MGnify helps while
retaining the selected `16 active shards × 2 HMMER CPUs` topology. The
temporary app selected new immutable profiles; it did not overwrite the
baseline profiles:

- `uniprot-384-v1`, averaging 269.33 MiB per shard instead of 404.00 MiB;
- `uniref90-256-v1`, averaging 267.56 MiB per shard instead of 535.11 MiB.

Both builders passed the full source-versus-shard canonical-record-multiset
gate with zero recovered records. UniProt preserved 225,619,586 records and
78,608,056,346 residues with 0.4696% maximum residue imbalance; its durable
build took 2,238.45 seconds. UniRef90 preserved 153,742,194 records and
52,375,181,535 residues with 0.4836% maximum residue imbalance; its durable
build took 1,326.08 seconds.

The new searches were submitted concurrently and compared with the existing
pinned monolithic pembrolizumab-VH evidence. UniProt reproduced all 23,248 hit
rows and the depth-23,249 A3M. UniRef90 reproduced all 9,999 hit rows and the
depth-10,000 A3M. In both cases, the full row multiset was equal and ordering
differences occurred only within equal printed E-value/bit-score blocks.

| Database | Profiles | Average shard | Baseline 16x2 | Candidate 16x2 | Observed change | CPU-s | Estimated cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| UniProt | 256 → 384 shards | 404.00 → 269.33 MiB | 605.2 s | 147.9 s | 4.09x faster | 12,430 → 4,189 | $0.1642 → $0.0552 |
| UniRef90 | 128 → 256 shards | 535.11 → 267.56 MiB | 113.4 s | 218.5 s | 1.93x slower | 3,195 → 5,264 | $0.0421 → $0.0695 |

The two-call critical path fell from 605.2 to 218.5 seconds and the summed
successful-search estimate fell from $0.2063 to $0.1247. Those aggregates
should not hide the opposite per-database outcomes. At fixed active fanout,
UniRef90 doubled from eight to sixteen waves and became correspondingly
slower, so `uniref90-256-v1` is not a production candidate for the 16x2
topology.

The UniProt result is promising but does not isolate shard count. The baseline
ran in a four-database concurrent batch on GCP `europe-west1`; the candidate
ran in a two-call batch on Azure `eastus2` immediately after its profile was
published. UniRef90 likewise moved from Azure `eastus2` to `uksouth`. Each
combination has one sample, by design. The measured outcomes are valid, but
provider, region, concurrent Volume load, and profile warmth prevent treating
the 4.09x UniProt difference as shard-count-only causality. The fixed
production registry remains unchanged until that tradeoff is reviewed.

The durable machine-readable summary is stored in
`AlphaFold3-MSA-Benchmark-outputs` at
`production-candidates/experiments/protein-shard-count-16x2-2026-07-24/summary.json`
with SHA-256
`1799ae134a3ec33fe84bc8268e99215608777082c4b7d071fd31cf7684162f48`.

The Modal hourly billing report attributes $0.48946 to the main fresh protein
app. Including the initial failed attempt that exposed invalid empty-A3M
handling and the cached comparison retry gives a total campaign bill of
$0.53402:

| Protein app | Outcome | Shell elapsed | CPU bill | Memory bill | Total bill |
| --- | --- | ---: | ---: | ---: | ---: |
| Initial attempt | Failed; prompted empty-A3M fix | 4m41s | $0.04091 | $0.00227 | $0.04318 |
| Fresh search attempt | Search evidence completed; comparison telemetry bug | 1h11m35s | $0.47014 | $0.01932 | $0.48946 |
| Cached comparison retry | Scientific gate passed | 1m01s | $0.00078 | $0.00060 | $0.00138 |

The original 25-nucleotide RNA fixture produced zero non-query hits in all
three monolithic searches. Its query-only results matched, but the explicit
non-query-hit gate correctly rejected it because it could not test RNA hit
merging or deduplication. Its successful-attempt search timings remain useful
only as performance diagnostics:

| Database | Monolith search | Sharded search | Speedup | Monolith estimated cost | Sharded estimated cost | Hit rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NT-RNA | 1,968.2 s | 258.8 s | 7.61x | $0.1130 | $0.0881 | 0 |
| RFam | 5.2 s | 2.5 s | 2.10x | $0.0005 | $0.0007 | 0 |
| RNAcentral | 262.8 s | 60.7 s | 4.33x | $0.0195 | $0.0245 | 0 |

A sharded NT-RNA preemption made the client-observed sharded batch take
3,280.4 seconds even though its restarted successful attempt took 258.8
seconds. The app's exact bill was $0.54123, versus a $0.24646
successful-attempt estimate. This is why final reports keep platform-billed
campaign cost, client-observed latency, and successful-attempt telemetry as
separate measurements.

The next 121-nucleotide Picea fixture produced 27,240 monolithic non-query
hits and therefore exercised all three database merges and final RNA
deduplication. The first comparison showed that sorting shard hits by printed
E-value and target ID, as upstream currently does, can place a lower printed
bit score ahead of a higher one. Sorting by printed E-value, descending
printed bit score, and target ID repaired RFam and NT-RNA: both then had
identical hit-row multisets and differed only within equal-score blocks.

RNAcentral remained saturated at 9,999 hits. Its corrected result differed
from the monolith by exactly one cutoff row; both alternatives had printed
E-value `2.4e-08` and bit score `46.8`, but different identities and aligned
sequences. Stock HMMER ranks with an internal full-precision score while
`tblout` exposes rounded values, so the monolithic cutoff choice cannot be
reconstructed exactly from the shard outputs. This is not an equal-score
permutation and remains a scientific gate failure.

The fixed non-saturating oracle therefore uses the exact RFam 14.9
RAGATH-1 hammerhead ribozyme representative
`URS0000D698D3_12908/1-119`. Its 119 nucleotides guarantee a real RFam hit
without deliberately selecting a universal RNA family:

```text
AACUCAGCUAGGGAGAGUAGCGAGCAUUACGUAAUACUACGUAUUACUCCAAUAACAUUGUCACUGAUGAGACCUAGACGAAACUACGGUAAACAUUUGCAUCAUACUGUAGUCUGAUA
```

That oracle passed on 2026-07-24. RFam returned three non-query rows,
RNAcentral returned one, and NT-RNA returned none. All three sharded row lists
were exactly equal to their monolithic counterparts, including order. After
cross-database deduplication, the final `unpairedMsa` was byte-exact with a
depth of 4. The hit-bearing gate passed with four monolithic per-database hits
before deduplication.

The three database calls ran concurrently. Per-search cost below has the same
successful-attempt telemetry basis as the protein table:

| Database | Profile shards | Monolith search | Sharded search | Speedup | Monolith CPU-s | Sharded CPU-s | Monolith estimated cost | Sharded estimated cost | Hit rows | Scientific result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| RFam | 16 | 4.7 s | 1.8 s | 2.67x | 22 | 24 | $0.0003 | $0.0003 | 3 | exact |
| RNAcentral | 64 | 201.9 s | 73.5 s | 2.75x | 1,685 | 2,134 | $0.0225 | $0.0281 | 1 | exact |
| NT-RNA | 256 | 1,157.2 s | 452.1 s | 2.56x | 9,641 | 13,099 | $0.1289 | $0.1726 | 0 | exact |

The concurrent search-batch wall time fell from 1,168.7 to 464.6 seconds,
a 2.52-fold speedup. Summed successful-search estimates rose from $0.1517 to
$0.2011. The complete seven-call remote campaign took 1,649.8 seconds,
including queueing, container initialization, and the final comparison.
Modal's closed 21:00--22:00 billing interval attributed $0.11562 to the app;
the final partial interval is intentionally not inferred from telemetry and
must be read from the next closed hourly report.

## Stores and immutable database registry

### Volumes

| Purpose | Modal Volume | Access |
| --- | --- | --- |
| Official monolithic FASTAs and fixed template store | `AlphaFold3-msa-db` | builder read/write only for explicit source retirement; template workers read-only |
| Published database profiles | `AlphaFold3-msa-db-sharded` | builder read/write; search workers read-only |
| Generated MSA and template cache | existing MSA cache, AlphaFold3 subpath | search workers read/write |
| Durable inference inputs, predictions, requests, logs, and metrics | `AlphaFold3-outputs` | staging helper and inference workers read/write |

### Supported database specifications

Production accepts a `database_id`, never a free-form path, shard count, Z
value, or polymer type.

| `database_id` | Profile ID | Official source | Shards | Polymer |
| --- | --- | --- | ---: | --- |
| `small_bfd` | `small-bfd-64-v2` | `bfd-first_non_consensus_sequences.fasta` | 64 | protein |
| `mgnify` | `mgnify-512-v1` | `mgy_clusters_2022_05.fa` | 512 | protein |
| `uniprot` | `uniprot-256-v1` | `uniprot_all_2021_04.fa` | 256 | protein |
| `uniref90` | `uniref90-128-v1` | `uniref90_2022_05.fa` | 128 | protein |
| `ntrna` | `nt-rna-256-v1` | `nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta` | 256 | RNA |
| `rfam` | `rfam-16-v1` | `rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta` | 16 | RNA |
| `rnacentral` | `rnacentral-64-v1` | `rnacentral_active_seq_id_90_cov_80_linclust.fasta` | 64 | RNA |

Each specification also fixes the expected official source statistic, profile
recipe version, SeqKit version and seed, and compatibility pins. Protein Z and
domZ are the validated source sequence count. RNA Z is the validated total
nucleotide count divided by 1,000,000.

The completed `small-bfd-64-v1` profile was a benchmark-only predecessor to
`small-bfd-64-v2`, whose published payload omits the monolithic source and
whose shuffled FASTA stays under `/tmp`. After all seven production candidates
passed validation, a read-only Sandbox inventory on 2026-07-24 confirmed that
`/profiles/` contains exactly the seven fixed directories listed above. The
obsolete v1 profile is absent, `.staging` is empty, and `.orphaned` does not
exist. That inventory predates the shard-count A/B: the Volume now also
contains the validated immutable experimental profiles `uniprot-384-v1` and
`uniref90-256-v1`. They do not change the fixed registry without a separate
decision.

Runtime search reads a fixed `/profiles/{profile_id}/manifest.json` only to
obtain and bind the trusted profile identity and search-space value. It never
discovers profiles, follows a mutable `current` pointer, or revalidates shards.

## Profile builder

Expose one production Modal function:

```python
build_sharded_database(
    database_id: str,
    seqkit_threads: int = 8,
    source_policy: Literal["keep", "compress", "delete"] = "keep",
) -> dict[str, object]
```

The production setup entrypoint validates the seven fixed profile locations,
collects only missing databases, and submits all missing database IDs to this
function concurrently. Each invocation retains its own Profile-ID claim and
generation-scoped paths, so valid profiles are skipped and the same database
cannot be built twice. The coordinator waits for all builders before running a
single final inventory/cleanup pass; individual workers never remove shared
staging state.

One invocation handles one logical database:

1. Resolve the code-owned specification.
2. If the matching manifest already exists, validate the publication and
   return `reused`.
3. Otherwise acquire the minimal per-Profile-ID build claim. An active conflict
   fails immediately; there is no polling or heartbeat.
4. Require the official uncompressed source FASTA. If only `<filename>.zst`
   exists, fail with instructions to restore it manually in a Modal Sandbox or
   equivalent environment.
5. Run full source `seqkit stats`, source hashing, and an exact scratch-space
   preflight for two FASTA payloads, the occurrence index, and 1 GiB headroom.
6. Run the pinned occurrence-indexed two-pass helper. Its first sequential pass
   tees the source into an ephemeral `/tmp` copy while writing fixed-width
   occurrence offsets. It syncs the local copy, closes the Volume file, and
   creates a deterministic seed-23 Fisher--Yates permutation over `uint32`
   record ordinals.
7. In pass two, issue bounded concurrent reads only from the local copy and
   write completed records strictly in permutation order to
   `/tmp/shuffled.fasta`. Index by source occurrence so duplicate full headers
   are preserved without FAI recovery. Normalize only a missing terminal
   newline.
8. Run `seqkit split2`, which writes generation-scoped raw shards to the
   sharded Volume. Delete the local shuffled payload and rename raw shards to
   their exact AlphaFold-compatible filenames.
9. Run source and per-shard `seqkit stats`, then compare the source and shard
   union with the recipe-v5 canonical full-record multiset validator. It scans
   shard files concurrently and combines domain-separated per-record SHA-256
   values with commutative sum, XOR, and sum-of-squares accumulators. Validate
   sequence and residue conservation, full-header occurrence preservation,
   filenames, shard count, balance, sizes, and artifact digests.
10. Derive Z/domZ from the measured source, write compact validation evidence
    including the native helper identity and metrics,
    commit the shard payload, and publish `manifest.json` last.
11. Deeply revalidate the published profile.
12. Apply `source_policy` only after successful publication:
    - `keep`: leave the source untouched;
    - `compress`: create `<complete-filename>.zst`, round-trip-check source
      bytes and SHA-256, commit it, then remove the plain source;
    - `delete`: remove the plain source only after the explicit request.

Failure cleanup removes only that generation's partial shard payload and local
scratch while retaining compact diagnostics. It never modifies an existing
published profile.

The builder uses `(0.125, 32.125)` CPUs, `(1024, 262144)` MiB
requested/maximum memory, the default 512 GiB ephemeral disk, and Modal's
24-hour maximum function timeout. It does not request extra disk or place the
ephemeral source copy or shuffled FASTA on either persistent Volume.

The 2026-07-24 source-Volume inventory reports a 75.4 GiB NT-RNA FASTA and a
119.7 GiB MGnify FASTA. Using the rounded MGnify size and its known
623,796,864 records, the implemented preflight requires about 245 GiB:
239.4 GiB for the staged and shuffled FASTA payloads, about 4.65 GiB for
64-bit occurrence offsets, and 1 GiB headroom. This remains well below the
default 512 GiB container disk quota. The runtime still computes the exact
requirement from the unrounded source size and measured record count before
creating local payloads.

The same inventory found all seven required uncompressed source FASTAs.
Durable `production-candidates/profile-builds/` evidence exists for all seven
fixed profiles: `small-bfd-64-v2`, `mgnify-512-v1`, `rfam-16-v1`,
`rnacentral-64-v1`, `uniref90-128-v1`, `uniprot-256-v1`, and
`nt-rna-256-v1`. The final read-only Sandbox inventory found no obsolete
profile, abandoned staging generation, or orphaned profile requiring cleanup.
The later A/B added durable build evidence for `uniprot-384-v1` generation
`fca8768dc0a946ba83bfb7205ac3de52` and `uniref90-256-v1` generation
`2e372b231b584427aad022b3df07da64`.

Generation `44178e3a52864732b330491758d10d8f` republished
`small-bfd-64-v2` on 2026-07-24 after the mature C helpers and shared Python
sharding primitives were extracted into
`src/biomodals/app/fold/alphafold3/`. The refactored builder preserved the
source SHA-256, all 65,984,053 record occurrences, and all 16,748,600,902
residues. Its source and 64-shard canonical full-record signatures matched at
`5b07a3e612a0ef0e7d6957f2ef057e0e082a97b8f9f6e798093e22d18b371909`;
no duplicate-header record required recovery. Maximum shard residue imbalance
was 0.2269%, and the manifest SHA-256 is
`b2288d239d5f3b1d86582c0c8c9de5e339f83204c277cffdeec59ab97647f270`.
The durable operation took 8 minutes 11.852 seconds from its first log event
through its completion marker; the local Modal invocation, including image
startup, took 9 minutes 14 seconds.

Generation `6a08f17a689943b9ace9947ba285ece9` published the first recipe-v5
profile, `nt-rna-256-v1`, on 2026-07-24. The builder preserved all 37,105,891
record occurrences and 76,752,808,514 residues, recovered no records, and
measured 0.9710% maximum shard residue imbalance. Its one-file C validator
scan took 404.359 seconds at 200.260 MB/s; its eight-thread, 256-file scan took
47.284 seconds at 1.739 GB/s. The source and shard canonical full-record
multisets matched. The durable completion marker binds manifest SHA-256
`65c031c30fa49f300de25d2d9b55a6c467770cda5cf32fc45684fa1f5b8b33ed`;
the remote worker completed in 32 minutes 31 seconds, and the local Modal
invocation completed in 33 minutes 5 seconds.

Generation `660774ec2a9d4008bef5f3334ef909d1` published
`mgnify-512-v1` on 2026-07-24. The builder preserved all 623,796,864 record
occurrences and 114,578,946,467 residues, recovered no records, and measured
0.3403% maximum shard residue imbalance. The 128.580 GB source and 130.166 GB
shard union produced the same canonical full-record signature,
`cbd27240746abf41258fdf5cd173567142fb8bfa81051372c4da74a561fb49be`.
The claim-to-completion interval was 1 hour 7 minutes 35.758 seconds. Its
manifest SHA-256 is
`0f7236eeb26fe29032b2094511b797f916a7c515a9378cd2ef4fa4b09be8cc46`.

## Search cache layout

`sequence_hash` remains a hash of sequence text only. Polymer namespaces
prevent protein/RNA collisions:

```text
/<Protein|RNA>/<sequence_hash[:2]>/<sequence_hash>/
  raw-msa/
    <database_id>/
      <search_identity>/
        result.a3m
        metrics.json
        run.log
        done.json
  unpaired.a3m
  paired.a3m                 # protein only
  combined.done.json
  templates.json             # protein only
  templates.done.json        # protein only
```

A valid `done.json` is the reusable boundary for one sequence-by-database
search. It binds result digests, the fixed profile manifest, pinned AlphaFold
and HMMER behavior, and result-affecting parameters. CPU allocation, thread
count, active shard count, and container partitioning are operational and do
not affect Search Identity.

Per-shard tblout files remain transient. A database worker publishes only the
merged A3M, compact metrics/log, and marker written last.

Legacy unnamespaced MSA files are ignored. Existing unmarked combined/template
files are not cache evidence and are replaced only after a complete validated
new publication is ready.

## Search orchestration

Replace `run_data_pipeline`, its monolithic upstream subprocess, and
`copy_msa_to_ssd` with app-owned orchestration.

### Field resolution

The local entrypoint exposes:

```python
search_msa: bool = True
search_protein_templates: bool = True
max_parallel_search_workers: int = 4
```

Behavior is:

| MSA | Templates | Resolution |
| --- | --- | --- |
| on | on | preserve non-empty fields; populate every missing MSA and protein template field |
| on | off | populate missing MSAs; preserve non-empty templates and set missing/null templates to `[]` |
| off | either | run no searches; preserve supplied fields, set missing MSAs to `""`, and missing/null protein templates to `[]` |

Resolve fields independently:

- missing protein `unpairedMsa`: UniRef90, small BFD, and MGnify;
- missing protein `pairedMsa`: UniProt only;
- missing RNA `unpairedMsa`: RFam, RNAcentral, and NT-RNA;
- requested missing protein templates: after its unpaired MSA is resolved.

Identical sequences share canonical generated searches. Caller-supplied MSA or
template evidence stays attached to its original chain and is never published
to, or copied through, the shared cache.

### Worker topology

The MSA phase submits one Modal worker per missing unique
`(polymer, sequence, database)` result. Each worker:

- reads the fixed sharded profile directly from
  `AlphaFold3-msa-db-sharded`;
- uses `(0.125, 32.125)` Modal CPUs;
- runs HMMER with `n_cpu=2`;
- allows at most 16 concurrently active shards;
- uses the exact pinned upstream Jackhmmer or Nhmmer constructors and merge
  functions;
- writes through generation-exclusive staging and publishes `done.json` last.

The worker owns all shards for that database result. A preemption before
publication reruns that whole database, not individual shards.

Search Build Claims prevent duplicate expensive work across concurrent
requests. Other owners wait for a valid publication. A surfaced search failure
ends the current request after reporting exact incomplete tasks; the app does
not add a retry loop.

After the complete MSA phase, a separate protein-template phase runs one fixed
eight-CPU upstream template search per unique required
`(sequence, resolved-unpaired-MSA digest)`. It reads PDB seqres and selected
mmCIF files directly from `AlphaFold3-msa-db` and never copies the mmCIF store.
Its identity binds the unpaired-MSA digest, maximum template date, pinned tool
behavior, and result-affecting parameters, while treating the fixed reference
store as immutable.
The two phases share the request-wide worker cap and do not overlap.

### Upstream-compatible assembly

Biomodals performs only the storage-aware orchestration. A narrow pinned
adapter preserves upstream scientific semantics:

- protein unpaired: deduplicate in UniRef90, small-BFD, MGnify order;
- protein paired: preserve the UniProt A3M;
- RNA unpaired: deduplicate in RFam, RNAcentral, NT-RNA order;
- protein templates: search from the resolved unpaired alignment.

A canonical combined MSA is published at the sequence root only when every
constituent is a canonical Raw Database MSA. Mixed custom/generated assemblies
and their template results remain request-local.

Once all required fields are resolved, validate and serialize the Enriched
AlphaFold Input and skip upstream's data pipeline. Upstream inference still
performs input processing, featurization, model execution, and output writing.

## Local path materialization and inference identity

Before any remote work, the local helper:

1. resolves relative paths against the input JSON's parent directory;
2. reads `unpairedMsaPath` and `pairedMsaPath` into inline strings and clears
   their path fields;
3. reads `userCCDPath` into inline `userCCD` and clears its path;
4. rejects any inline/path pair that is simultaneously populated;
5. reads and hashes each custom template `mmcifPath`.

After search enrichment, construct an Inference Identity View by dumping the
validated input with explicit defaults, removing only `name` and `modelSeeds`,
and representing every custom template by content digest plus its mappings.

The app-local `hash_sequences` helper derives `run_id` from:

- the complete normalized identity view;
- recycle count;
- diffusion-sample count;
- pinned app and AlphaFold identity;
- the code-owned model checkpoint label;
- the identity-schema label.

It excludes display name, seeds, GPU class/count, worker counts, search policy,
paths, and other scheduling controls. Supported GPU classes are deliberately
cache-interchangeable and do not promise bitwise-identical outputs.

After computing `run_id`, upload each path-backed custom template once as
`custom-templates/{sha256}.cif`, rewrite the worker input to its mounted path,
persist the neutral identity under `inputs/`, and persist the request's
Enriched AlphaFold Input under its request directory:

```text
/{run_id[:2]}/{run_id}/
  inputs/
  custom-templates/
  outputs/
  requests/
  .markers/
  logs/
  metrics/
```

## Seed scheduling and durable inference

Normalize the submitted seeds to a non-empty, sorted unique list. Preserve the
submitted and normalized lists in the request manifest.

Derive:

```text
request_id = hash_sequences(run_id, canonical_normalized_seed_list)
```

For each requested seed:

1. Trust a matching `.markers/seeds/{seed}.json` without scanning its output
   directory.
2. Reuse marked seeds.
3. Atomically claim each unmarked `(run_id, seed)` through the append-only
   generation protocol.
4. Partition only seeds owned by this request into at most `max_num_gpus`
   disjoint, balanced lists. Never send the same seed to two workers.

Each GPU worker receives one or more seeds and invokes the pinned upstream
process with:

```text
--run_data_pipeline=false
--run_inference=true
```

The input name is replaced in the worker copy by
`af3-{run_id[:16]}`. The worker writes into
`outputs/.workers/{claim-generation}/`, not the shared output directory.
After upstream succeeds, it promotes each seed's native
`seed-{seed}_sample-{sample_index}` and optional seed-specific directories,
commits them, then writes the seed marker last. The marker records only run,
seed, claim, and `(sample_index, ranking_score)` rows.

There is no app retry loop after a surfaced GPU failure. Successful sibling
seeds remain reusable, incomplete seeds are reported, and a later explicit
invocation claims only the unmarked seeds.

## Rankings, request results, and retrieval

Both global and request rankings use:

```text
ranking_score descending, seed ascending, sample_index ascending
```

Equal-score permutations remain scientifically equivalent; the tie-breakers
only stabilize presentation.

A serialized Summary Build Claim rebuilds the global summary from the union of
all currently marked seeds without waiting for unrelated in-flight seeds. It
may never replace a summary with one covering fewer seeds.

Every successful request publishes a small
`requests/{request_id}/` view containing:

- request manifest;
- requested-seed ranking;
- request-best upstream-style files;
- references to every requested canonical seed directory;
- the observed global-summary marker.

Remote functions return compact Volume-relative metadata, never tarball bytes.
The local entrypoint downloads only manifest-declared request artifacts and
creates:

```text
{sanitized_display_name}_{request_id[:12]}_AlphaFold3.tar.zst
```

Downloaded copies replace the canonical `af3-{run_id[:16]}` basename prefix
with the sanitized display name. The archive also contains every referenced
`custom-templates/{sha256}.cif`; only the downloaded input copy rewrites its
`mmcifPath` values to those archive-relative files. Durable Volume inputs,
paths, and template files remain canonical.

An existing archive is reused only if it is non-empty and readable. A corrupt
or unreadable existing archive causes a clear failure instead of silent
overwrite.

## Incremental implementation

Each section is a separate reviewed commit. Run relevant local checks before
every commit and stage only the named files.

### 1. Finish the generic method in the temporary MSA app

Commit: `fold: generalize MSA profile builder`

- replace small-BFD-only builder constants with the seven fixed experimental
  specifications;
- preserve existing Phase 1 paths and evidence;
- move shuffled FASTA/index to `/tmp`;
- add source-policy handling and minimal profile claims;
- add generic Jackhmmer/Nhmmer profile search entrypoints.

Only `alphafold3_msa_app.py` and directly associated research documentation are
in scope.

### 2. Add the scientific gates

Commit: `fold: add RNA sharding oracle`

- retain the 25-nucleotide query as a rejected query-only negative control;
- use the official RFam 14.9 record `ALWZ042362541.1/2041-2161` as the
  hit-bearing 121-nucleotide gate;
- require at least one non-query monolithic hit;
- compare monolithic and sharded RFam, RNAcentral, and NT-RNA searches using
  the same full-database Z;
- compare identities, scores, E-values, aligned-sequence multisets, and the
  final deduplicated RNA A3M;
- retain equal-score permutation tolerance only.

The temporary app implements this as
`full-hit-rows-exact-modulo-contiguous-evalue-bit-score-ties-v1`. Every
per-database row's target ID, complete description, exact aligned A3M
sequence, textual E-value, textual bit score, and multiplicity must match.
Only row order inside a contiguous block with the same E-value and bit score
may differ. The earlier Phase 1 duplicate-tail/Jaccard characterization is
diagnostic evidence only and cannot make a production-candidate oracle pass.

The pinned RNA merge identifies hits as
`accession/alignment_from-alignment_to`, uses the last tblout line for a
repeated coordinate ID, and orders the merged rows by E-value then that ID.
The oracle evidence normalizer mirrors those rules before applying the strict
comparison above.

No Modal job is submitted without explicit permission. Record successful or
failed results in research documentation before production promotion.

### 3. Add production profile definitions and builder

Commit: `fold: add sharded database builder`

- add the separate sharded Volume and fixed registry;
- add the pinned compact occurrence-indexed shuffle and the proven
  split/validation code;
- add manifest-last publication, minimal claim, `/tmp` staging, and source
  policy;
- add a setup coordinator that submits every missing database profile
  concurrently, reuses valid profiles, and performs cleanup only after all
  builders finish;
- do not import benchmark code or campaign types.

### 4. Replace the production data-pipeline worker

Commit: `fold: add resumable sharded MSA search`

- add raw-result identities, markers, claims, and cache paths;
- add generic protein/RNA database workers with the selected topology;
- add upstream-compatible assembly and RNA deduplication;
- add independent field resolution and the request-wide worker budget;
- remove monolithic `run_data_pipeline` and all MSA-to-SSD copying.

### 5. Separate and resume template search

Commit: `fold: add resumable template search`

- add the post-MSA template phase and flat validated publication;
- preserve caller evidence locally;
- use the immutable template store directly;
- make incomplete search failures explicit and non-retrying.

### 6. Materialize inputs and establish run identity

Commit: `fold: stage enriched AlphaFold inputs`

- inline caller MSA and CCD path inputs;
- hash/upload custom templates;
- implement `hash_sequences`, the normalized identity view, `run_id`, and
  request ID;
- persist inputs under the hash-fanned output-Volume run root.

### 7. Persist and reconcile seed predictions

Commit: `fold: persist seed predictions`

- replace function-result tarball bytes with output-Volume worker staging;
- add per-seed claims, disjoint multi-seed workers, seed markers, and explicit
  partial-failure reporting;
- add deterministic request/global rankings and serialized global summary.

### 8. Return request-scoped results

Commit: `fold: retrieve request-scoped outputs`

- publish request manifests and request-best files;
- download only requested canonical artifacts;
- restore presentation prefixes locally;
- create and validate request-qualified `.tar.zst` archives;
- update the entrypoint flags and help text.

### 9. Record validation and remove obsolete production paths

Commit: `fold: document sharded MSA validation`

- record all scientific and integrated smoke results;
- confirm no production reference remains to `copy_msa_to_ssd`,
  `search_chains_in_parallel`, or `max_parallel_data_pipelines`;
- leave the temporary benchmark app available for evidence and future tuning.

## Verification gates

### Local and non-cost-incurring checks

For each implementation commit:

- `prek run --files <changed files>`;
- `git diff --check`;
- use `ty` as guidance if available;
- compile/import the app without submitting Modal functions;
- exercise pure helpers with temporary local fixtures outside
  `tests/app/fold/`;
- `uv run biomodals app list`;
- `uv run biomodals app help alphafold3`;
- `uv run biomodals workflow list`.

Pure-helper checks cover:

- registry/path and manifest validation;
- protein/RNA Z derivation;
- recovery-prefix removal and aggregate statistics;
- search and run identity stability;
- the three search-policy combinations;
- independent paired/unpaired field resolution;
- duplicate-sequence/custom-evidence behavior;
- claim generations and marker-last publication;
- seed normalization, reconciliation, partitioning, and tie ordering;
- request manifest selection and archive prefix rewriting.

The existing untracked `tests/app/fold/` tree is user-owned and remains
untouched and uncommitted.

### Cost-incurring Modal gates

Each group requires fresh user permission and should run in the current Herdr
space in a monitorable tab:

1. build and validate each required immutable profile;
2. run monolithic-versus-sharded RNA oracles, beginning with RFam;
3. run a protein integrated data-stage case using the pembrolizumab VH query;
4. run an RNA integrated data-stage case and verify final enriched JSON;
5. verify cache resume by withholding one database result and confirming only
   that result reruns;
6. run one minimal inference request, then overlapping seed requests to verify
   reuse, request-only retrieval, and global-summary accumulation.

Any scientific mismatch stops promotion. Any surfaced job error is diagnosed,
fixed, and resubmitted only with the authorization governing that paid
operation.

## Rollback and non-goals

Rollback is a code deployment reversal. The new sharded Volume is separate,
the original database Volume is retained by default, and legacy MSA cache paths
are ignored rather than migrated or deleted.

Not included initially:

- per-shard durable retries;
- automatic `.zst` source restoration;
- shard revalidation during normal searches;
- compressed runtime shards or prediction-time SSD staging;
- mutable profile aliases or automatic database upgrades;
- app-level automatic retry loops;
- GPU-class-specific inference cache identity;
- post-publication seed artifact audits;
- changes outside the two AlphaFold3 app files and accepted documentation.

## Definition of done

Integration is complete when:

1. the generic protein and RNA sharding oracles pass for every selected
   production profile;
2. production searches consume only fixed published profiles and no longer
   copy sequence databases to SSD;
3. protein and RNA enriched JSON matches pinned upstream scientific behavior;
4. one failed database/template task and one failed seed can be resumed without
   rerunning completed siblings;
5. overlapping seed requests cannot duplicate a seed worker and return only
   their requested artifacts;
6. all local checks and agreed paid smoke tests pass;
7. the final production diff is limited to
   `alphafold3_app.py` plus the separately retained experimental work and
   documentation authorized by this plan.

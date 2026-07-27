# AlphaFold3 sharded-MSA production integration plan

Status: accepted.

Scope: finish the sharding method in
`src/biomodals/app/fold/alphafold3_msa_app.py`, validate it, then integrate the
minimal production implementation into
`src/biomodals/app/fold/alphafold3_app.py` and its mature sibling modules under
`src/biomodals/app/fold/alphafold3/`.

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
slower in this observation.

The UniProt result is promising but does not isolate shard count. The baseline
ran in a four-database concurrent batch on GCP `europe-west1`; the candidate
ran in a two-call batch on Azure `eastus2` immediately after its profile was
published. UniRef90 likewise moved from Azure `eastus2` to `uksouth`. Each
combination has one sample, by design. The measured outcomes are valid, but
provider, region, concurrent Volume load, and profile warmth prevent treating
the 4.09x UniProt difference as shard-count-only causality.

Checklist 3 initially kept `uniref90-128-v1`, because its finer candidate was
scientifically equivalent but slower, and selected `uniprot-384-v1`. On
2026-07-27 the user explicitly superseded that operational choice and selected
`uniref90-256-v1` for production end-to-end testing. Both selected profiles
remain scientifically equivalent to their monoliths. The fixed identities
preserve the measured tradeoff and make either choice reversible if production
telemetry warrants it.

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

### Production protein end-to-end evidence

A two-chain protein production smoke test passed on 2026-07-27 after two
operational fixes. The input contained a 118-residue chain A and a 128-residue
chain H. The first preflight expected the superseded
`uniref90-128-v1` profile; production now selects the user-approved
`uniref90-256-v1`. The first inference attempt then exposed that
`model_dump_json(exclude_unset=False)` retained null sibling chain types in
each sequence entry. Upstream AlphaFold requires each entry object to have
exactly one key. The shared inference serializer now excludes null fields
while preserving empty MSA strings and template lists. A focused regression
test and the original remote reproduction both pass.

All eight cold sharded searches used 16 active shards with two HMMER CPUs.
The database tasks shared a four-worker request budget, so the per-call times
below must not be summed as wall time:

| Chain | Length | small BFD | MGnify | UniProt | UniRef90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 118 | 29.08 s | 356.24 s | 165.13 s | 100.18 s |
| H | 128 | 27.22 s | 189.92 s | 138.25 s | 97.50 s |

MGnify was the cold MSA critical path. From the earliest raw-search start to
the final raw-search completion, the batch took about 357.83 seconds. The two
subsequent eight-CPU template searches ran concurrently and took 83.45
seconds for A and 28.63 seconds for H.

The enriched upstream data JSON preserved both input sequences and contained:

| Chain | Unpaired rows | Unpaired bytes | Paired rows | Paired bytes | Templates |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 4,345 | 1,116,700 | 4,357 | 1,099,827 | 4 |
| H | 10,587 | 2,768,685 | 23,607 | 6,165,794 | 4 |

Upstream inference ran with `--run_data_pipeline=false`, produced both A and H
in every model, and published five valid sample directories per seed. Seed 1
featurization took 11.24 seconds and model inference took 26.93 seconds.
Seed 2 featurization took 10.13 seconds and model inference took 23.50
seconds. Exact platform cost is intentionally deferred to a closed Modal
billing interval rather than inferred from wall time.

The stable run ID was
`387616514785bd74d7429b67c0d3664f5cf87c14a2bbafcc7136a7140c25868a`.
The request sequence verified marker reuse, accumulated summaries, and
request-only retrieval:

| Request | MSA/template work | GPU work | Seed outcome | Archive contents | Shell elapsed |
| --- | --- | --- | --- | --- | ---: |
| `[1]` corrected baseline | 8/8 and 2/2 reused after the cold attempt | seed 1 only | published `[1]` | five seed-1 samples | 3m16s |
| `[1]` exact repeat | all reused | none | existing seed 1 reused | five seed-1 samples | 1m53s |
| `[1, 2]` overlap | all reused | seed 2 only | reused `[1]`, published `[2]` | ten requested samples | 4m31s |
| `[2]` subset | all reused | none | reused `[2]` | five seed-2 samples only | 2m25s |

The accumulated summary covers seeds `{1, 2}`. Seed 2/sample 1 is the current
global best with ranking score `0.32454750520095166`; the earlier seed-1-only
request archive remains a five-sample snapshot. The subset request embeds the
two-seed global summary but does not leak seed 1 artifacts into its archive.

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
| `uniprot` | `uniprot-384-v1` | `uniprot_all_2021_04.fa` | 384 | protein |
| `uniref90` | `uniref90-256-v1` | `uniref90_2022_05.fa` | 256 | protein |
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
passed validation, a read-only Sandbox inventory on 2026-07-24 confirmed the
then-fixed seven directories. That inventory predates the shard-count A/B.
The Volume now contains the validated immutable profiles `uniprot-384-v1` and
`uniref90-256-v1`. The 2026-07-27 production decision selects both; the
earlier profiles and one-shot timing comparison remain immutable evidence.

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
Durable `production-candidates/profile-builds/` evidence exists for the seven
baseline profiles: `small-bfd-64-v2`, `mgnify-512-v1`, `rfam-16-v1`,
`rnacentral-64-v1`, `uniref90-128-v1`, `uniprot-256-v1`, and
`nt-rna-256-v1`. The then-final read-only Sandbox inventory found no obsolete
profile, abandoned staging generation, or orphaned profile requiring cleanup.
The later A/B added durable build evidence for the now-selected
`uniprot-384-v1` generation
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

Status: implemented locally; no production Modal work submitted.

- add the separate sharded Volume and fixed registry;
- add the pinned compact occurrence-indexed shuffle and the proven
  split/validation code;
- add manifest-last publication, minimal claim, `/tmp` staging, and source
  policy;
- add a setup coordinator that submits every missing database profile
  concurrently, reuses valid profiles, and performs cleanup only after all
  builders finish;
- do not import benchmark code or campaign types.

The fixed contracts live in `alphafold3/profiles.py`, and the mature
Modal-independent construction path lives in
`alphafold3/profile_builder.py`. Both AlphaFold apps call that shared builder;
the production app does not import the temporary app.

The production `setup_sharded_databases` entrypoint is plan-only unless
`submit=true`. On submission it performs one lightweight manifest/artifact-size
inventory, rejects invalid publications, uses `starmap` with
`return_exceptions=True` to wait for every missing-profile builder, and only
then runs the final cleanup/inventory barrier. That barrier removes abandoned
staging/orphan generations and non-selected immutable profile directories, so
`/profiles/` contains exactly the seven code-owned selections. Setup evidence
is committed under `msa-profile-builds/` in `AlphaFold3-outputs`. The plan and
submission log expose the container cap, local worker count, and maximum
effective worker slots. Claims are released from a `finally` path; stale
takeover appends one atomically elected successor after fencing the expired
generation as abandoned. Claim and successor records are never deleted, so an
interrupted takeover remains resumable. A legacy active owner is adopted as
the root instead of being deleted. After election, the builder reloads the
Volume and reuses any manifest published during the claim race. On failure,
compact evidence is committed before that generation's partial payload is
removed. An existing compressed source never authorizes deleting a changed
plain FASTA: the plain source must still match the published source size and
digest.

### 4. Replace the production data-pipeline worker

Commit: `fold: add resumable sharded MSA search`

Status: implemented; the two-chain protein production search passed on
2026-07-27. The integrated RNA production case remains deferred.

- add raw-result identities, markers, claims, and cache paths;
- add generic protein/RNA database workers with the selected topology;
- add upstream-compatible assembly and RNA deduplication;
- add independent field resolution and the request-wide worker budget;
- remove monolithic `run_data_pipeline` and all MSA-to-SSD copying.

The mature scientific adapter and append-only generation-claim protocol now
live in `alphafold3/msa_search.py` and `alphafold3/generation_claims.py`.
Both AlphaFold apps call the shared Jackhmmer/Nhmmer execution, corrected RNA
merge, and pinned assembly functions. Production first performs one
lightweight marker inspection, then spends the request-wide budget only on
missing unique sequence-by-database searches. A complete canonical
protein/RNA assembly is published with `combined.done.json` last; mixed
caller/generated fields remain request-local.

### 5. Separate and resume template search

Commit: `fold: add resumable template search`

Status: implemented; both protein template searches and their cache reuse
passed on 2026-07-27.

- add the post-MSA template phase and flat validated publication;
- preserve caller evidence locally;
- use the immutable template store directly;
- make incomplete search failures explicit and non-retrying.

The pinned template adapter now lives in `alphafold3/template_search.py`. It
reconstructs upstream's fixed eight-CPU HMMsearch and template filters, reads
PDB seqres plus only selected mmCIF files from the immutable source Volume, and
serializes the same query-to-template mappings used by the upstream data
pipeline. A canonical result is keyed by the protein sequence, resolved
unpaired-MSA digest, maximum template date, pinned AlphaFold/HMMER versions,
and result-affecting parameters. It replaces `templates.json` and publishes
`templates.done.json` last under the sequence cache root.

Production runs the template phase only after every MSA assembly has
completed. Canonical generated MSAs inspect and reuse the shared template
cache; caller-supplied or mixed MSA evidence remains request-local. Missing
unique template tasks share `max_parallel_search_workers` with the earlier MSA
phase without overlapping it. Any failed worker is reported with its sequence
and unpaired-MSA digests, and the coordinator adds no retry loop.

### 6. Materialize inputs and establish run identity

Commit: `fold: stage enriched AlphaFold inputs`

Status: implemented; Volume staging, stable identity, custom-template
retrieval, and strict upstream JSON parsing passed on 2026-07-27.

- inline caller MSA and CCD path inputs;
- hash/upload custom templates;
- implement `hash_sequences`, the normalized identity view, `run_id`, and
  request ID;
- persist inputs under the hash-fanned output-Volume run root.

The local materialization and identity seam now lives in
`alphafold3/inference_inputs.py`. Before search submission it resolves every
relative path against the input JSON, inlines protein/RNA MSA and custom CCD
content, clears those path fields, rejects ambiguous inline/path pairs, and
captures every path-backed custom-template byte string and SHA-256.

After enrichment, the module validates and explicitly dumps the complete
input, removes only `name` and `modelSeeds`, and represents inline and
path-backed templates identically by content digest plus residue mappings.
`hash_sequences` length-frames canonical JSON fragments. The resulting
`run_id` covers that view, recycle/sample counts, pinned app/upstream identity,
the declared `AlphaFold3/af3.bin:v1` model label, and the run-identity schema.
Seeds are normalized to a non-empty sorted unique tuple and only affect
`request_id`.

The prepared staging payload uses `/{run_id[:2]}/{run_id}/`, deduplicates
custom templates at `custom-templates/{sha256}.cif`, rewrites worker paths to
the mounted `AlphaFold3-outputs` location, and includes
`inputs/identity.json` plus `requests/{request_id}/input.json`. The app uses
Modal's local `Volume.batch_upload(force=True)` interface for those exact
bytes. The output Volume is also mounted for inference so staged templates are
readable; canonical seed-output publication remains Checklist 7.

### 7. Persist and reconcile seed predictions

Commit: `fold: persist seed predictions`

Status: implemented; seed 1, the overlapping `[1, 2]` request, and marker-only
seed reuse passed on 2026-07-27.

- replace function-result tarball bytes with output-Volume worker staging;
- add per-seed claims, disjoint multi-seed workers, seed markers, and explicit
  partial-failure reporting;
- add deterministic global rankings and a serialized accumulated summary.

The durable inference boundary now lives in
`alphafold3/seed_predictions.py`. A lightweight coordinator trusts matching
seed markers without rescanning their artifacts, atomically claims only
unmarked `(run_id, seed)` work, and partitions owned seeds into disjoint,
balanced GPU-worker lists. Thus overlapping requests such as
`[1, ..., 20, 8080, ..., 8090]` reuse the first marked seeds and schedule only
the missing suffix.

Each GPU worker runs the pinned upstream inference process in an exclusive
`outputs/.workers/{worker_id}` directory. It validates the complete ranking
table and every expected sample output for all assigned seeds before promotion.
It then promotes native per-seed directories, commits them, and writes each
seed's completion marker last. A surfaced failure is recorded and is not
retried within the same request; already marked siblings remain reusable.

The summary finalizer serializes concurrent writers through a separate
generation claim, rebuilds the upstream-shaped data and ranking files from the
currently visible seed-marker union, and orders equal scores by ascending seed
then sample index. It never replaces a valid summary with one covering fewer
seeds and publishes the summary marker only after all declared artifacts are
committed. Request-specific rankings and presentation files remain Checklist
8.

### 8. Return request-scoped results

Commit: `fold: retrieve request-scoped outputs`

Status: implemented; one-seed, overlapping two-seed, and seed-2-only archives
passed content and manifest inspection on 2026-07-27.

- publish request manifests and request-best files;
- download only requested canonical artifacts;
- restore presentation prefixes locally;
- create and validate request-qualified `.tar.zst` archives;
- update the entrypoint flags and help text.

The request boundary now lives in `alphafold3/request_results.py`. After all
requested seed markers and the accumulated summary are complete, one CPU
finalizer publishes the deterministic requested-seed ranking, request-best
files, terms, and a content-addressed copy of the observed global-summary
marker. Its `manifest.json` is written last and records submitted/normalized
seeds, removed duplicates, reused/newly published seeds, the observed global
best, every requested sample file, optional seed outputs, per-artifact byte
sizes, and only the custom templates referenced by the enriched request input.
Before promotion, the finalizer snapshots the observed global-summary marker
and verifies that the copied bytes still match the loaded marker digest; a
concurrent summary expansion causes a clear retryable failure instead of
publishing a mixed request view. It never copies seed directories or declares
an unrelated completed seed. A failed publication commits compact evidence
under `requests/{request_id}/failures/` before its private staging directory is
removed.

The local entrypoint streams only those Volume-relative manifest artifacts,
rejects paths outside the hash-fanned run root, verifies each stream's declared
size, and restores upstream's exact sanitized display-name prefix in
downloaded basenames. The durable request input uses the canonical
`af3-{run_id[:16]}` name and content-addressed `custom-templates/{sha256}.cif`
paths, so callers with the same scientific input and seeds upload identical
bytes even when their display names or original inline/path template
representations differ. Only the downloaded input copy changes: it restores
the current display name and rewrites staged `mmcifPath` values to
archive-relative custom-template paths. The resulting
`{presentation_name}_{request_id[:12]}_AlphaFold3.tar.zst` is created through a
temporary path and promoted only after every expected member is readable. A
non-empty readable existing archive is reused; an unreadable one causes an
explicit error and is never overwritten.

### 9. Record validation and remove obsolete production paths

Commit: `fold: document sharded MSA validation`

Status: in progress. The AlphaFold3 app and sibling-module subtree contain no
references to `copy_msa_to_ssd`, `search_chains_in_parallel`, or
`max_parallel_data_pipelines`. A repository-wide scan found one separate
production dependency: `ppiflow_workflow.py` still calls the removed
`run_data_pipeline` interface with `copy_msa_to_ssd=True` and expects the old
tarball-returning inference signature. A focused TODO now records that
follow-up without changing PPIflow behavior under this plan's AlphaFold3-only
boundary.

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
- changes outside the two AlphaFold3 app files, their mature supporting
  modules, and accepted documentation.

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
   the two AlphaFold3 app files, their mature `alphafold3/` supporting modules,
   and documentation authorized by this plan.

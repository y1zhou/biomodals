# AlphaFold 3 small-BFD sharding Phase 1 results

Status: focused Phase 1 campaign complete; production integration not started
Results date: 2026-07-23
Campaign: `small-bfd-phase1-v2`
Implementation: `src/biomodals/app/fold/alphafold3_msa_app.py`

## Outcome

The provisional production topology is **S3: 16 simultaneously active shard
searches with two HMMER CPUs per search**. All cases scan the same 64 physical
small-BFD shards; S3 processes them in four waves inside one Modal container.

For the 120-residue pembrolizumab VH query, S3 reduced Search Wall Time from
249.940 seconds for the monolithic B1 oracle to 49.051 seconds, a 5.10-fold
speedup. Its estimated compute cost was 48.9% higher than B1. S3 was also
slightly faster and materially cheaper than the 32-search, one-CPU S6 case.

This is a single-query, single-observation recommendation. It establishes a
strong candidate for production design, not a stability estimate across
queries or Modal placements.

## Executed scope

The completed work was:

1. Repair, validate, and publish a 64-shard small-BFD profile.
2. Execute the nine-case Volume scan matrix.
3. Run the B0, B1, and S3 search smoke.
4. Reuse B1 and S3, then run one new sample each for S1, S2, S4, S5, and S6.
5. Compare every sharded result with B1 using the tie-aware scientific gate.

The original plan's three repeated screening blocks and GroEL stress blocks
were not submitted. After the smoke result, the user approved a smaller
one-shot sweep because the benchmark's purpose was to select a sharding
strategy rather than measure repeatability.

## Published database profile

The profile is stored at:

```text
AlphaFold3-msa-db-sharded:/profiles/small-bfd-64-v1/
```

| Property | Observed value |
| --- | ---: |
| Physical shards | 64 |
| Source record occurrences | 65,984,053 |
| Aggregate shard record occurrences | 65,984,053 |
| Source and aggregate residues | 16,748,600,902 |
| Source bytes | 18,171,626,364 |
| Recovered FAI-omitted records | 55,187 |
| Recovered residues | 24,934,582 |
| Maximum shard residue imbalance | 0.267% |
| SeqKit | 2.13.0, seed 23, eight threads |

Profile manifest SHA-256:

```text
f9215302fcb4426385777fb85052e2031c2cd3b11b3a0b8fedde30ca5406534a
```

The repair retained two-pass shuffle. It parsed SeqKit's duplicate FAI
warnings, recovered the omitted records from their source sequence byte
offsets, appended them under temporary UUID-prefixed headers, split the
repaired FASTA, and stripped the generation-scoped prefixes. Publication
required exact aggregate SeqKit record, residue, and sequence checksum
agreement and absence of the temporary namespace.

The preparation evidence also identifies one production-hardening gap:
`seqkit sum` does not include headers. The benchmark validates each recovered
header against its warning and source offset, but a generic builder should add
an order-independent parsed header-and-sequence multiset check or an explicitly
versioned equivalent.

## Volume scan evidence

Each case read the complete physical layout during a labelled first pass and an
immediate same-container repeat. These labels do not prove cold and warm cache
states. Rates are decimal aggregate GB/s; multi-container aggregate rates are
not per-container Volume guarantees.

| Case | Containers × readers | First pass | First GB/s | Repeat | Repeat GB/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| V0 monolith | 1×1 | 19.062 s | 0.953 | 2.283 s | 7.959 |
| V1 shards | 1×1 | 101.063 s | 0.182 | 2.899 s | 6.353 |
| V2 shards | 1×2 | 6.599 s | 2.791 | 1.283 s | 14.356 |
| V3 shards | 1×4 | 3.626 s | 5.079 | 0.719 s | 25.629 |
| V4 shards | 1×8 | 2.437 s | 7.557 | 0.434 s | 42.486 |
| V5 shards | 1×16 | 2.392 s | 7.700 | 0.393 s | 46.899 |
| V6 shards | 2×8 | 4.592 s | 4.011 | 0.371 s | 49.695 |
| V7 shards | 4×4 | 1.423 s | 12.940 | 0.248 s | 74.294 |
| V8 shards | 4×16 | 1.607 s | 11.460 | 0.161 s | 114.571 |

The scan results demonstrate strong cache effects and useful parallel delivery,
but they do not measure a resettable backend-only Volume bandwidth. In
particular, aggregate rates above Modal's advertised per-Volume maximum must
include distributed or local caching. The search benchmark is the
production-relevant decision input.

## Protein search results

Search Wall Time covers the pinned database query through merged A3M
completion. Sample Wall Time ends after durable core evidence publication.
Costs use the Modal prices recorded on 2026-07-22.

| Rank | Case | Active × CPU | Search | Speedup | Sample | CPU-core-s | Peak cores | Est. cost |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | S3 | 16×2 | 49.051 s | 5.10× | 55.242 s | 1,221.59 | 28.654 | $0.016125 |
| 2 | S6 | 32×1 | 49.568 s | 5.04× | 56.724 s | 1,507.55 | 32.424 | $0.019875 |
| 3 | S4 | 8×4 | 62.982 s | 3.97× | 75.774 s | 1,075.23 | 18.778 | $0.014254 |
| 4 | S2 | 8×2 | 67.411 s | 3.71× | 74.410 s | 1,092.53 | 18.769 | $0.014477 |
| 5 | S1 | 4×2 | 105.110 s | 2.38× | 110.406 s | 922.70 | 10.163 | $0.012332 |
| 6 | S5 | 4×8 | 113.721 s | 2.20× | 120.732 s | 1,011.30 | 10.456 | $0.013516 |
| — | B1 | monolith×8 | 249.940 s | 1.00× | 254.450 s | 783.51 | 3.646 | $0.010829 |

B0 deliberately omitted explicit Z/domZ and is descriptive rather than a
scientific oracle. It took 270.747 seconds and returned 223 hit rows containing
139 unique targets, compared with B1's 142 rows and 108 unique targets. This
reinforces that full-database Z and domZ values are scientific inputs, not
performance options.

## Scientific equivalence

Every S1 through S6 result passed against B1:

- full truncated unique-hit Jaccard was 1.0;
- all 108 unique targets were shared;
- score mismatch count was zero;
- sequence mismatch count was zero;
- top-order differences occurred only inside groups with identical E-values
  and bit scores.

Equal-score permutations are treated as scientifically equivalent. The gate
does not hide unequal scores, sequences, target sets, or movements between
non-tied rank groups.

## Interpretation

Inter-shard concurrency was more effective than additional HMMER threads:

- S2 and S4 both peaked near 18.8 cores even though S4 doubled the nominal
  per-shard CPU allocation.
- S1 and S5 both peaked near 10 cores, while S5's eight-thread searches were
  slower than S1's two-thread searches.
- S6 saturated approximately 32 cores but did not improve latency over S3.

S3 and S6 are within 1.1% on Search Wall Time, but S6 consumed 23.4% more
CPU-core-seconds and cost 23.3% more. S3 therefore wins the close-result review
without another paid tie-break sample. S1 is the lower-cost sharded alternative
when a 2.38-fold speedup is sufficient.

The observed search rate for the approximately 18.2 GB source is roughly
0.37 GB/s at S3 latency. Together with S6's CPU saturation, this provides no
evidence that Modal Volume bandwidth is the primary bottleneck at the selected
topology. It also does not establish a general Volume throughput guarantee.

## Recommendation and limits

Carry the following candidate into production design:

- retain 64 physical FASTA shards on a Modal Volume;
- allow 16 simultaneously active shard searches per database;
- give each HMMER process two CPUs;
- retain the dynamic container CPU range `(0.125, 32.125)`;
- merge all 64 durable shard results through the unchanged upstream merger;
- do not stage the full database on container-local SSD.

The benchmark did not test:

- repeated-run stability;
- the GroEL stress query or other deep protein MSAs;
- RNA Nhmmer searches or RNA database Z semantics;
- template search, the integrated AlphaFold data pipeline, or inference;
- simultaneous per-database containers or multiple unique input sequences.

Those are design and validation inputs for production integration, not implied
successes of the small-BFD result.

## Durable evidence

The authoritative result artifacts are:

```text
AlphaFold3-MSA-Benchmark-outputs:
  /benchmarks/small-bfd-phase1-v2/storage-scans/
  /benchmarks/small-bfd-phase1-v2/search/smoke/
  /benchmarks/small-bfd-phase1-v2/search/focused-sweep/
```

The focused-sweep completion identity is:

```text
e47feb56f61b9ad098e0637d8d1761834c29be53079c4e54a172fc43a081f59d
```

Its `done.json` validates `results.parquet`, `comparisons.json`,
`rankings.json`, `summary.json`, and `summary.md`. The campaign-root
`summary.md` is an earlier progress snapshot; the focused-sweep completion
marker and artifacts are the authoritative final topology result.

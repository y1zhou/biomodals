# SeqKit two-pass shuffle audit for AlphaFold 3 databases

Date: 2026-07-24

Scope: SeqKit `v2.13.0` at commit
[`d13b5fa`](https://github.com/shenwei356/seqkit/commit/d13b5fa388cc869de05abe1bdb07980eef5efb4e),
its pinned `github.com/shenwei356/bio v0.13.9` dependency at commit
[`d4c578a`](https://github.com/shenwei356/bio/commit/d4c578a731dbc713fe144e06b7c64a702e5cd9a2),
and the production-profile command in
[`alphafold3_msa_app.py`](https://github.com/y1zhou/biomodals/blob/1f28563627ef088228f2c2916fd7b8f641c6a97c/src/biomodals/app/fold/alphafold3_msa_app.py)

## Verdict

The app is invoking SeqKit's real two-pass path correctly. SeqKit is **not
retaining or memory-mapping the complete FASTA sequence payload**. The apparent
full-file memory use is nevertheless real: SeqKit 2.13.0 retains an
allocation-heavy index of every full header, rereads all headers into a second
in-memory structure, and retains a permutation over every record. For UniProt,
those structures occupy about 123.5 GiB of resident process memory.

The observed 1.7--1.9 MB/s is also consistent with the implementation. SeqKit
emits records serially and performs one small `ReadAt` at a shuffled source
offset for each record. Local `/tmp` output does not turn those source reads
into sequential I/O. The advertised Modal Volume rate of up to 2.5 GB/s is not
a guarantee, and it does not predict the throughput of roughly 4,000 serialized
random reads per second
([Modal Volume documentation](https://modal.com/docs/guide/volumes#using-a-volume-on-modal)).

This separates two conclusions:

1. **Invocation correctness:** the current command and redirection are behaving
   as implemented upstream.
2. **Algorithm suitability:** stock `seqkit shuffle -2` is not suitable for
   all seven AF3 databases under the current 128 GiB limit and no-source-copy
   constraint. MGnify is likely to exceed the memory limit by a wide margin.

No SeqKit command-line flag fixes both the memory scaling and serialized random
reads. Omitting `--update-faidx` on a retry can avoid rebuilding a trusted
index, but it does not reduce the resident index or accelerate shuffled output.

## Command under audit

The production builder constructs this command at lines 1668--1680 of
`alphafold3_msa_app.py` and directs stdout to a container-local file:

```text
seqkit shuffle \
  -j 8 \
  --two-pass \
  --update-faidx \
  --tmp-dir /tmp/af3-uniprot-i9lq8prr/seqkit \
  --rand-seed 23 \
  /mnt/AlphaFold3-msa-db/uniprot_all_2021_04.fa
```

The live process command line matched that argv exactly. Because the input is
an uncompressed, plain FASTA, `--tmp-dir` is not used. SeqKit only creates a
temporary FASTA there for stdin or a non-plain input. For a plain FASTA,
`newFile` remains the source path and its index is
`<source>.seqkit.fai`
([two-pass input selection](https://github.com/shenwei356/seqkit/blob/d13b5fa388cc869de05abe1bdb07980eef5efb4e/seqkit/cmd/shuffle.go#L170-L205)).
This matches the intended app design: source and FAI remain on the source
Volume, while only shuffled output is written to `/tmp`.

SeqKit documents two-pass mode as reducing memory by reading sequence IDs,
shuffling them, and extracting records through a FASTA index. It does not
promise constant or record-count-independent memory
([official 2.13.0 shuffle documentation](https://github.com/shenwei356/seqkit/blob/d13b5fa388cc869de05abe1bdb07980eef5efb4e/doc/docs/usage.md#L3965-L3998)).
The documentation's example is a genome with only 194 indexed records. The AF3
databases instead contain tens to hundreds of millions of relatively short
records, which is the unfavorable shape for this implementation.

## Exact two-pass algorithm

### 1. `--update-faidx` deletes and rebuilds the index

When `--update-faidx` is set, SeqKit deletes an existing `.seqkit.fai`, then
calls `getFaidx`
([shuffle source](https://github.com/shenwei356/seqkit/blob/d13b5fa388cc869de05abe1bdb07980eef5efb4e/seqkit/cmd/shuffle.go#L205-L220)).
`getFaidx` either creates or reads the FAI and passes the resulting in-memory
index to `fai.NewWithIndex`
([helper source](https://github.com/shenwei356/seqkit/blob/d13b5fa388cc869de05abe1bdb07980eef5efb4e/seqkit/cmd/helper.go#L444-L463)).

FAI creation scans the FASTA with a buffered reader one line at a time. It does
not retain completed sequence bodies. It does, however, retain one
`map[string]Record` entry per unique full header while simultaneously writing
the complete textual FAI
([FAI representation and reader](https://github.com/shenwei356/bio/blob/d4c578a731dbc713fe144e06b7c64a702e5cd9a2/seqio/fai/fai.go#L13-L85),
[FAI creation loop](https://github.com/shenwei356/bio/blob/d4c578a731dbc713fe144e06b7c64a702e5cd9a2/seqio/fai/fai.go#L109-L281)).

Therefore, `--update-faidx` causes a full **sequential scan** of the source and
a full in-memory **metadata index**. It does not cause the sequence payload to
be retained in heap.

### 2. The source FASTA is explicitly not memory-mapped

The shuffle command sets:

```go
fai.MapWholeFile = false
```

before constructing the index
([shuffle source](https://github.com/shenwei356/seqkit/blob/d13b5fa388cc869de05abe1bdb07980eef5efb4e/seqkit/cmd/shuffle.go#L62-L72)).
With this setting, `Faidx` opens the source as an ordinary file and
`SubSeqNotCleaned` allocates only the requested record span and calls
`reader.ReadAt(data, pstart)`. The whole-file `mmap` branch is not taken
([pinned `bio` source](https://github.com/shenwei356/bio/blob/d4c578a731dbc713fe144e06b7c64a702e5cd9a2/seqio/fai/faidx.go#L49-L126)).

### 3. SeqKit duplicates the full-header metadata

After creating/loading `faidx.Index`, SeqKit rereads the complete textual FAI
with `getSeqIDAndLengthFromFaidxFile`. That function materializes both an
`[]string` of every full header and an `[]int` of every sequence length
([helper source](https://github.com/shenwei356/seqkit/blob/d13b5fa388cc869de05abe1bdb07980eef5efb4e/seqkit/cmd/helper.go#L486-L525)).

Shuffle then:

- copies all IDs into `map[int]string index2name`;
- allocates `[]int indices` with one machine integer per record; and
- shuffles that integer slice
  ([shuffle source](https://github.com/shenwei356/seqkit/blob/d13b5fa388cc869de05abe1bdb07980eef5efb4e/seqkit/cmd/shuffle.go#L229-L245)).

At the start of output, the process has at least:

- the full `map[string]fai.Record`;
- a second full-header representation reachable from `map[int]string`;
- the integer permutation;
- Go map buckets and string/allocation metadata; and
- memory allocated for the temporary ID and length slices, some of which may
  remain resident in the Go heap even after becoming unreachable.

The memory complexity is proportional to record count and total header bytes,
not merely the size of one record. "Two pass" avoids retaining all sequence
bodies, but it is not bounded-memory shuffling.

### 4. Output is serialized randomized `ReadAt`

SeqKit loops over the shuffled integer permutation on one goroutine. For each
entry, it looks up the full header, looks up its FAI record, performs
`SubSeqNotCleaned`/`ReadAt`, and writes that record before advancing to the next
one
([shuffle output loop](https://github.com/shenwei356/seqkit/blob/d13b5fa388cc869de05abe1bdb07980eef5efb4e/seqkit/cmd/shuffle.go#L247-L269)).

`-j 8` sets Go's `GOMAXPROCS`, but this output loop has no worker pool or
goroutines. Raising `-j` cannot parallelize its source reads. SeqKit's own
documentation also notes that more than four threads generally do not improve
commands bottlenecked on FASTA/Q I/O
([parallelization documentation](https://github.com/shenwei356/seqkit/blob/d13b5fa388cc869de05abe1bdb07980eef5efb4e/doc/docs/usage.md#L112-L135)).

The stdout path is buffered. SeqKit's pinned `xopen v0.4.0` wraps stdout in a
64 KiB `bufio.Writer`, including when the output name is `-`
([writer implementation](https://github.com/shenwei356/xopen/blob/411040eb39f09e52ad3c32c46642378d1e68dbcd/xopen.go#L419-L527)).
The Python wrapper's direct stdout-to-file descriptor is therefore not causing
one local SSD write syscall per FASTA field.

## Live UniProt evidence

The following was collected from container
`ta-01KY86XC6WR5FZMHYKP72XKA6R`, SeqKit PID 98, at
2026-07-24 01:25:35 UTC while the output phase was active.

| Measurement | Value |
|---|---:|
| UniProt records | 225,619,586 |
| Source FASTA | 108,447,942,931 bytes |
| `.seqkit.fai` | 33,130,008,158 bytes |
| FAI bytes per record | 146.84 |
| Shuffled output at snapshot | 30,507,204,608 bytes |
| `VmRSS` | 129,545,348 KiB (123.54 GiB) |
| `VmData` | 129,595,136 KiB (123.59 GiB) |
| Configured hard limit | 131,072 MiB (128 GiB) |
| Threads | 14 |

Linux defines `VmRSS` as resident memory and `VmData` as private data segments
([kernel `/proc` documentation](https://docs.kernel.org/filesystems/proc.html#chapter-1-collecting-system-information)).
Both values were byte-for-byte unchanged in a second snapshot 30 seconds later.
This stable 123.5 GiB footprint during ongoing output contradicts progressive
accumulation of sequence bodies and supports a fully built metadata heap.
`/proc/meminfo` simultaneously reported only about 186 MiB under `Cached`, so
the process footprint was not explained by source file page cache.

Some lower-bound components, before Go-map buckets, header bytes, duplicate
representations, and allocator overhead, are:

| Component | Approximate size |
|---|---:|
| 225,619,586 `fai.Record` values at 48 bytes | 10.09 GiB |
| `[]int` permutation | 1.68 GiB |
| temporary `[]string` headers | 3.36 GiB |
| temporary `[]int` lengths | 1.68 GiB |

The 33.13 GB FAI averages 146.84 textual bytes per record. Two copies of its
long header strings plus two Go maps, their load-factor slack, and allocation
overhead plausibly account for the remaining resident memory. The observed
total is about 588 resident bytes per database record.

The same 30-second interval measured:

| Delta | Rate |
|---|---:|
| `rchar` | 42,712,918 bytes (1.42 MB/s) |
| shuffled output | 58,064,896 bytes (1.94 MB/s) |
| `syscr` | 120,910 calls (4,030 calls/s) |
| input bytes per read syscall | 353 bytes |
| output bytes per read syscall | 480 bytes |

Linux defines `rchar` as bytes passed to `read`/`pread` and `syscr` as the
number of read-family syscalls
([kernel `/proc/<pid>/io` documentation](https://docs.kernel.org/filesystems/proc.html#proc-pid-io-display-the-io-accounting-fields)).
The source averages 480.67 bytes per FASTA record. The live 353 source bytes per
read plus a header supplied from the in-memory index producing about 480 output
bytes per read is direct runtime evidence of the source-code path: one tiny
random sequence read per output record.

`read_bytes` and `write_bytes` were both zero, which is not useful for this
Volume/FUSE path; the documented `rchar`/`syscr` counters and output file growth
provide the relevant evidence.

## Why the Modal bandwidth figure does not apply

Modal says Volumes are designed for up to 2.5 GB/s, explicitly notes that
actual throughput is not guaranteed, and describes caching and chunking as
throughput optimizations
([official Volume guide](https://modal.com/docs/guide/volumes)).
That ceiling is compatible with the earlier high-throughput sequential and
multi-shard scans. It does not promise 2.5 GB/s for a single dependency chain
of hundreds of millions of approximately 353-byte randomized `pread` calls.

The active loop exposes almost no I/O parallelism:

```text
lookup shuffled record
        |
        v
one randomized ReadAt from source Volume
        |
        v
buffered write to local /tmp
        |
        v
next record
```

At roughly 4,030 records/s, latency/IOPS is the limiting dimension. Local SSD
bandwidth is downstream of the blocking source read and is not the bottleneck.

## Flag and code assessment

| Item | Assessment |
|---|---|
| `--two-pass` | Correctly selects the intended upstream algorithm. |
| stdout redirected to `/tmp/shuffled.fasta` | Correct; local writes are buffered and are not the observed bottleneck. |
| `--tmp-dir /tmp/...` | Harmless but has no effect for this plain FASTA. |
| `-j 8` | Correct as the requested default, but cannot parallelize the serialized output loop. |
| `--update-faidx` | Correct for an uncertain first build; forces a source scan and FAI rewrite, but does not retain the FASTA payload. |
| FAI beside source Volume | Expected for plain FASTA and required by stock SeqKit's path selection. |
| Current memory request/limit | Barely sufficient for UniProt: live RSS was about 96.5% of 128 GiB. |

For a retry, `--update-faidx` may be omitted **only** when a separate durable
marker proves that the immutable source identity matches a completely written
FAI. SeqKit writes its FAI directly rather than publishing it atomically, so
blindly trusting any existing sidecar after preemption could accept a partial
index. Reuse would save the source scan and 33 GB sidecar rewrite, but
`fai.Read` would still materialize the full map and the shuffled output loop
would remain unchanged.

## MGnify feasibility warning

Do not submit MGnify with the current stock algorithm and 128 GiB hard limit.
MGnify has 623,796,864 records, 2.765 times UniProt's 225,619,586. A naive
record-count projection of UniProt's measured footprint is about 342 GiB.
Header lengths differ, so this is not a peak-memory prediction, but the
direction is unambiguous: all dominant structures scale with record count, and
MGnify is very likely to exceed 128 GiB by a wide margin.

Even before header bytes and map overhead, MGnify's logical structures include
tens of GiB of `fai.Record` values, map keys/values, and a multi-GiB
permutation. Running it unchanged would spend time building a large FAI and
then likely fail before or during output.

## Implemented follow-up

The initial no-source-copy recommendation was reversed after the stock
UniProt run demonstrated that serialized random Volume reads were the dominant
runtime. The replacement remains a narrowly scoped, pinned two-pass shuffler
for this database shape:

1. Use the first sequential pass over the original source to tee an exact
   ephemeral copy onto container-local SSD while building the index.
2. Store ordinal record offsets and lengths in compact fixed-width arrays,
   rather than maps keyed by duplicated full headers.
3. Store the permutation as 32-bit ordinals; all seven databases have fewer
   than \(2^{32}\) records.
4. Preserve seed 23 and deterministic Fisher--Yates permutation semantics.
5. Close the Volume source after pass one. In the second pass, use a bounded
   concurrent `pread` window against only the local copy, but write completed
   records in permutation order.
6. Index records by occurrence, not header, so duplicate full headers are
   preserved naturally. Retain byte-offset diagnostics and the existing
   occurrence/header validation gates.
7. Require exact source-copy bytes, a bijective source-occurrence permutation,
   deterministic duplicate-header regression coverage, aggregate
   `seqkit stats`, a canonical full-record multiset comparison, shard balance,
   and the existing scientific search oracle.

This is a change to the implementation behind "two-pass shuffle," not to the
scientific sharding recipe: the source is still scanned/indexed first, records
are still assigned a deterministic random permutation, and only the shuffled
output is stored under `/tmp`. The source staging changes only I/O topology;
it does not change record identity, permutation order, or shard membership.

If "use two-pass shuffle" is interpreted as requiring the **unmodified stock
SeqKit binary**, then the current constraints cannot all be satisfied:

- keeping the source only on the Modal Volume preserves the serialized random
  Volume reads;
- stock SeqKit preserves the full-header maps and record-count-scaled memory;
- writing output to `/tmp` changes neither behavior.

The project chose the latter option: the persistent source remains immutable
on its Volume, but the one-time builder creates an ephemeral local copy. The
general builder now has a `(1024, 262144)` MiB memory request/limit range; its
compact offset and permutation structures, rather than that higher ceiling,
make MGnify feasible.

## Staged-local UniProt result

Generation `ec27ba8d37294ed38872720362daee44` published
`uniprot-256-v1` on 2026-07-24. It used the occurrence-indexed C helper with
eight workers, source policy `keep`, and a 256 GiB memory ceiling. The job ran
from 02:28:34 to 03:46:49 UTC, including all source scans, sharding,
validation, artifact hashing, Volume commits, and final deep verification.

| Stage | Duration or rate |
|---|---:|
| source `seqkit stats` | 5m 08s |
| source SHA-256 | 2m 14s |
| source `seqkit sum --all` | 5m 38s |
| first pass: Volume read plus local-source tee and occurrence index | 169.87s; 638.43 MB/s |
| Fisher--Yates permutation | 5.44s |
| second pass: eight ordered local-SSD `pread` workers | 899.48s; 120.57 MB/s |
| `split2` plus final shard renames | 9m 04s; 199.52 MB/s of FASTA payload |
| shard `seqkit stats` | 50.48s |
| aggregate checksum, artifact digests, publication, and deep verification | 37m 23s |
| complete builder | 1h 18m 15s |

The second pass emitted 108,447,942,931 bytes and all 225,619,586 source
occurrences. Warm one-minute intervals were 120--138 MB/s. The complete
second-pass average was 120.57 MB/s, about 62 times the stock SeqKit
observation of 1.94 MB/s. This validates the local-source staging hypothesis;
the job was retained under the predeclared 20 MB/s stop threshold.

The final profile contains 225,619,586 records and 78,608,056,346 residues.
Its maximum shard residue imbalance is 0.2804%, recovered-record count is zero,
and manifest SHA-256 is
`62bac582c973db700de978fe89474fb311c067ea49fc7b69dc81ad07b0b12194`.

The aggregate shard checksum used:

```text
cat <256 explicit shard paths> | seqkit sum -j 8 --all -
```

This is scientifically valid because it presents the shard union as one
logical FASTA, directly comparable with the source checksum. It is also
operationally serial: SeqKit parallelizes `sum` across input files, while this
command has one stdin input and one global sort of 225,619,586 per-sequence
hashes. Running `seqkit sum <256 shard paths>` would use file-level
parallelism, but would produce 256 non-composable final digests rather than the
required aggregate digest.

`seqkit sum` ignores headers and validates the sequence multiset only. The
duplicate-header guarantee instead comes from the helper's source-occurrence
index and bijective permutation, the exact record-count gate, and the
byte-for-byte duplicate-header regression.

Recipe v5 replaces this serial checksum for all subsequent profile builds with
an explicitly composable record-multiset validator. For each parsed FASTA
record it computes a domain-separated SHA-256 over the full header, concatenated
sequence, and explicit header and sequence lengths. Header and sequence case
are significant; line endings and sequence line wrapping are not. The report
combines all four 64-bit digest lanes using modular sums, XORs, and sums of
squares, together with record, header-byte, and sequence-byte totals. These
commutative aggregates are order-independent and multiplicity-sensitive.

The original source is scanned as one sequential file, while the final shards
are scanned concurrently over up to the configured worker count. Their
canonical signatures must match exactly, and their record and sequence-byte
totals must also match the independently generated SeqKit statistics. This
removes both the global hash sort and the header-awareness gap. Recipe v4
manifests, including the already published UniProt profile above, remain
readable and do not need to be rebuilt.

### Read-only recipe-v5 validator benchmark

A read-only benchmark against the published `uniprot-256-v1` profile completed
on 2026-07-24. Generation `cf1bff66cb1542d0ab288deaf0d75f08`
mounted both the source and sharded database Volumes read-only and wrote only
to the benchmark-output Volume. It did not mutate the source or profile.

| Scan | Files | Threads | Duration | Throughput |
|---|---:|---:|---:|---:|
| source monolith | 1 | 1 | 488.801s (8m 08.801s) | 221.865 MB/s |
| published shards | 256 | 8 | 113.268s (1m 53.268s) | 957.449 MB/s |
| combined scanner time | 257 | source 1; shards 8 | 602.069s (10m 02.069s) | — |

The source and shard sides both contained 225,619,586 records,
27,966,533,331 header bytes, and 78,608,056,346 sequence bytes. Every
aggregate lane matched, producing signature SHA-256
`f32c0e9f8f6ab2a8a647e84323e916485f4f761eecc1db1440ce5f99ca276c34`.
This is a full canonical `(header, sequence)` multiset result, not only a
sequence checksum.

The one-file source side is intentionally single-threaded in the current
helper because its parallelism is across files. The 256-file shard side used
all eight requested workers and was 4.32 times faster by throughput. The
source scan was 1.44 times slower than the historical 338.390-second source
`seqkit sum --all` scan. Conversely, the 602.069-second combined validation
was 3.73 times shorter than the historical 2,242.917-second interval from
completed shard statistics through the old builder's completion. That latter
ratio is contextual rather than a direct validator-only comparison: the old
interval also included artifact hashing, Volume publication, and final deep
verification.

The persistent evidence is under
`production-candidates/record-multiset-benchmarks/uniprot-256-v1/`
`cf1bff66cb1542d0ab288deaf0d75f08/` on
`AlphaFold3-MSA-Benchmark-outputs`.

The final decision is to use the occurrence-indexed C shuffler and this
full-record C validator for new profiles. SeqKit remains in the builder for
independent statistics and splitting, not aggregate checksum validation. The
current simple file-parallel validator is the accepted implementation; a
future optimization may reuse a source signature produced while staging only
if it preserves the same canonicalization and aggregate contract.

### First recipe-v5 builder: NT-RNA

Generation `6a08f17a689943b9ace9947ba285ece9` published
`nt-rna-256-v1` on 2026-07-24 with source policy `keep`. The remote worker ran
from 05:59:13.946 to 06:31:44.713 UTC. The local Modal invocation, including
app startup and final log handling, completed in 33 minutes 5 seconds.

| Stage | Duration or rate |
|---|---:|
| source `seqkit stats` | 428.425s (7m 08.425s) |
| source SHA-256 and shuffle setup | 76.513s |
| first pass: Volume read plus local-source tee and occurrence index | 61.930s; 1.308 GB/s |
| Fisher--Yates permutation | 1.124s |
| second pass: eight ordered local-SSD `pread` workers | 150.303s; 538.760 MB/s |
| `split2` plus final shard renames | 577.031s (9m 37.031s); 140.334 MB/s of source payload |
| source full-record validator | 404.359s; 200.260 MB/s |
| 256-shard full-record validator, eight threads | 47.284s; 1.739 GB/s |
| shard statistics through durable completion marker | 191.285s |
| complete remote worker | 1,950.767s (32m 30.767s) |

The source contained 37,105,891 records, 76,752,808,514 residues, and
80,977,012,680 physical bytes. SeqKit's split output occupied 82,237,359,282
physical bytes because FASTA wrapping changed, but the validator's
line-ending- and wrapping-independent full-record signatures matched exactly.
No record required recovery. The maximum shard residue imbalance was 0.9710%,
below the 5% publication threshold.

The published manifest SHA-256 is
`65c031c30fa49f300de25d2d9b55a6c467770cda5cf32fc45684fa1f5b8b33ed`.
Durable evidence is under
`production-candidates/profile-builds/nt-rna-256-v1/`
`6a08f17a689943b9ace9947ba285ece9/` on
`AlphaFold3-MSA-Benchmark-outputs`.

### Refactor regression: small BFD

Generation `44178e3a52864732b330491758d10d8f` rebuilt
`small-bfd-64-v2` on 2026-07-24 after the native helpers and shared Python
sharding primitives moved into `src/biomodals/app/fold/alphafold3/`. The
remote operation ran from its first durable log event at 07:34:50.826 UTC to
the completion marker at 07:43:02.678 UTC, or 491.852 seconds. The local Modal
invocation, including image startup and final log handling, completed in
9 minutes 14 seconds.

| Stage | Duration or rate |
|---|---:|
| source `seqkit stats` | 86.084s |
| source SHA-256 and shuffle setup | 32.327s |
| first pass: Volume read plus local-source tee and occurrence index | 28.048s; 647.880 MB/s |
| Fisher--Yates permutation | 1.997s |
| second pass: eight ordered local-SSD `pread` workers | 181.106s; 100.337 MB/s |
| staged-source SHA-256 verification | 12.963s |
| `split2` to 64 Volume shards | 83.270s |
| source full-record validator | 93.692s; 193.951 MB/s |
| 64-shard full-record validator, eight threads | 13.969s; 1.318 GB/s |
| overlapping split and validator stage | 97.410s |

The rebuilt profile contains exactly 64 shards, 65,984,053 record
occurrences, and 16,748,600,902 residues. The shard files occupy
18,417,671,014 physical bytes after SeqKit rewrapping, compared with the
18,171,626,364-byte source. Source and shard scans both reported
1,225,073,303 header bytes and the same canonical signature SHA-256:

```text
5b07a3e612a0ef0e7d6957f2ef057e0e082a97b8f9f6e798093e22d18b371909
```

Every aggregate sum, XOR, and sum-of-squares lane matched. The staged source
also matched the immutable Volume source SHA-256
`fd87dca06401b03f4ac3c59a82dac14db491a7933ed6abaa19e14e02c6eb1af5`.
Maximum shard residue imbalance was 0.2269%, below the 5% publication gate.
The published manifest SHA-256 is
`b2288d239d5f3b1d86582c0c8c9de5e339f83204c277cffdeec59ab97647f270`.

The earlier generation `d2eeaef371b249bd90734fb885127cf8` had the same
source identity, final record count, and residue total only after recovering
55,187 records and 24,934,582 residues omitted by SeqKit's duplicate-header
FAI. The occurrence-indexed implementation recovered zero records because no
occurrence was discarded in the first place. This result therefore validates
the refactored construction path and eliminates the known duplicate-ID
omission mechanism. It does not replace the end-to-end protein and RNA MSA
search oracles. The final MGnify profile and inventory gates have now passed,
so those oracles are the next cost-incurring validation step.

Durable evidence is under
`production-candidates/profile-builds/small-bfd-64-v2/`
`44178e3a52864732b330491758d10d8f/` on
`AlphaFold3-MSA-Benchmark-outputs`.

### Final recipe-v5 builder: MGnify

Generation `660774ec2a9d4008bef5f3334ef909d1` published
`mgnify-512-v1` on 2026-07-24 with source policy `keep`. The claim ran from
07:52:46.256 to 09:00:22.014 UTC, or 4,055.758 seconds
(1h 07m 35.758s).

| Stage | Duration or rate |
|---|---:|
| source `seqkit stats` | 505.443s (8m 25.443s) |
| source SHA-256 and shuffle setup | 130.723s |
| first pass: Volume read plus local-source tee and occurrence index | 150.550s; 854.066 MB/s |
| Fisher--Yates permutation | 23.495s |
| second pass: eight ordered local-SSD `pread` workers | 1,507.247s (25m 07.247s); 85.308 MB/s |
| staged-source SHA-256 verification | 95.302s |
| `split2` to 512 Volume shards | 1,158.053s (19m 18.053s) |
| source full-record validator | 812.756s (13m 32.756s); 158.202 MB/s |
| 512-shard full-record validator, eight threads | 160.347s; 811.776 MB/s |
| overlapping split and validator stage | 1,319.231s (21m 59.231s) |

The source contained 623,796,864 records, 114,578,946,467 residues,
12,129,365,959 header bytes, and 128,579,703,018 physical bytes. The 512
shards occupied 130,165,736,027 physical bytes after SeqKit rewrapping.
Source and shard scans matched every aggregate lane and produced canonical
signature SHA-256:

```text
cbd27240746abf41258fdf5cd173567142fb8bfa81051372c4da74a561fb49be
```

No duplicate-header record required recovery. Maximum shard residue imbalance
was 0.3403%, and the staged copy matched source SHA-256
`9e7f50956c19cbcd8181dc5e9d7d6eebc08257cc858fc07d3ec88fd6b48dbbc9`.
The published manifest SHA-256 is
`0f7236eeb26fe29032b2094511b797f916a7c515a9378cd2ef4fa4b09be8cc46`.

The ordered second pass averaged 85.308 MB/s despite MGnify's 623.8 million
short records, approximately 44 times the abandoned stock SeqKit observation
of 1.94 MB/s. This confirms that the occurrence-indexed C implementation
remains practical for the largest official AlphaFold 3 database under the
256 GiB memory ceiling.

A final read-only Sandbox inventory found exactly the seven fixed profile
directories under `/profiles/`, an empty `.staging` directory, and no
`.orphaned` directory. No cleanup mutation was required.

Durable evidence is under
`production-candidates/profile-builds/mgnify-512-v1/`
`660774ec2a9d4008bef5f3334ef909d1/` on
`AlphaFold3-MSA-Benchmark-outputs`.

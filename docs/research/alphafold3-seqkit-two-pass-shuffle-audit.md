# SeqKit two-pass shuffle audit for AlphaFold 3 databases

Date: 2026-07-24

Scope: SeqKit `v2.13.0` at commit
[`d13b5fa`](https://github.com/shenwei356/seqkit/commit/d13b5fa388cc869de05abe1bdb07980eef5efb4e),
its pinned `github.com/shenwei356/bio v0.13.9` dependency at commit
[`d4c578a`](https://github.com/shenwei356/bio/commit/d4c578a731dbc713fe144e06b7c64a702e5cd9a2),
and the production-profile command in
[`alphafold3_msa_app.py`](../../src/biomodals/app/fold/alphafold3_msa_app.py).

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

## Recommended next step

The current UniProt job may finish if its resident memory remains stable; there
is no evidence that cancelling and resubmitting the same command would improve
it. Capture its final peak RSS, output rate, and completion evidence. Do not
launch MGnify afterward.

Before MGnify, implement and test a narrowly scoped, pinned two-pass shuffler
for this database shape:

1. Keep a first sequential pass over the original source without copying it.
2. Store ordinal record offsets and lengths in compact fixed-width arrays,
   rather than maps keyed by duplicated full headers.
3. Store the permutation as 32-bit ordinals; all seven databases have fewer
   than \(2^{32}\) records.
4. Preserve seed 23 and deterministic Fisher--Yates permutation semantics.
5. In the second pass, use a bounded concurrent `ReadAt` prefetch window, but
   write completed records in permutation order. This preserves deterministic
   output while exposing enough I/O concurrency to test the Volume.
6. Index records by occurrence, not header, so duplicate full headers are
   preserved naturally. Retain byte-offset diagnostics and the existing
   occurrence/header validation gates.
7. First compare the helper against stock SeqKit on a manageable database:
   exact record/header/sequence occurrence multiset, deterministic rerun,
   aggregate `seqkit stats` and `seqkit sum`, shard balance, and the existing
   scientific search oracle. Benchmark several small prefetch bounds before
   selecting one.

This is a change to the implementation behind "two-pass shuffle," not to the
scientific sharding recipe: the source is still scanned/indexed first, records
are still assigned a deterministic random permutation, and only the shuffled
output is stored under `/tmp`.

If "use two-pass shuffle" is interpreted as requiring the **unmodified stock
SeqKit binary**, then the current constraints cannot all be satisfied:

- keeping the source only on the Modal Volume preserves the serialized random
  Volume reads;
- stock SeqKit preserves the full-header maps and record-count-scaled memory;
- writing output to `/tmp` changes neither behavior.

Under that stricter interpretation, the only choices are to accept the current
runtime and raise memory substantially, or relax the no-source-copy constraint.
There is no command correction that supplies a third option.

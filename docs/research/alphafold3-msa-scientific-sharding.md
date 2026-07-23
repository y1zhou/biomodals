# Scientifically faithful AlphaFold 3 MSA database sharding

Status: validated research conclusion after Phase 1 small-BFD preparation
Research dates: 2026-07-22 through 2026-07-23
Scope: database partition semantics, profile preparation, and search equivalence
Implementation target: `src/biomodals/app/fold/alphafold3_msa_app.py`

## Decision summary

The failed profile was not repaired by deleting records or changing the
small-BFD Z value. A valid shard set is a disjoint partition of all 65,984,053
source FASTA record occurrences: every occurrence appears exactly once with its
original header and sequence, and every shard search uses the full database
search space.

The published profile retains AlphaFold 3's recommended randomized
distribution while repairing SeqKit's duplicate-header omission:

1. Run SeqKit's two-pass shuffle and retain its FAI diagnostics.
2. Parse each duplicate warning's source sequence byte offset.
3. Recover the exact omitted record from that offset, prefix its header with a
   generation-scoped UUID, and append it to the shuffled FASTA.
4. Run `seqkit split2 --by-part 64` over the repaired shuffled FASTA.
5. Strip only that generation's anchored UUID prefixes from the final shards.
6. Reject publication unless aggregate SeqKit statistics and checksums match
   the source and no temporary prefix remains.

The UUID is a reversible transport identity. It makes the appended occurrences
unique while SeqKit partitions them, but it never reaches HMMER or an AlphaFold
MSA. The published profile recovered 55,187 records and 24,934,582 residues,
then validated all 65,984,053 records and 16,748,600,902 residues.

Direct round-robin `split2` remains a scientifically valid alternative because
randomization is a load-balancing recommendation rather than part of HMMER's
statistical model. It was not the selected Phase 1 recipe: the user chose to
retain the upstream two-pass shuffle and repair its observed omissions.

Do not use either SeqKit shuffle mode directly on the original small-BFD file.
In SeqKit 2.13.0, both implementations assume unique full headers in different
ways:

- two-pass shuffle uses a FASTA index keyed by the full header, which ignored
  55,187 duplicate-named records in our preparation;
- one-pass shuffle stores records in a map keyed by the full header, so a later
  occurrence overwrites an earlier occurrence and the output can repeat the
  replacement record rather than preserve the source multiset.

These behaviors are visible in the official SeqKit 2.13.0 source for the
[one-pass map](https://github.com/shenwei356/seqkit/blob/v2.13.0/seqkit/cmd/shuffle.go#L110-L166)
and the
[two-pass FASTA-index path](https://github.com/shenwei356/seqkit/blob/v2.13.0/seqkit/cmd/shuffle.go#L205-L269).
The observed loss was therefore caused by an unstated unique-header assumption
in the example command, not by Modal or HMMER.

## Why preserving duplicate identifiers is required

A duplicate FASTA identifier does not establish that two records contain the
same biological sequence. Even byte-identical duplicate records are members of
the published database snapshot and contribute to its declared search space.
Deduplicating by identifier or sequence would create a different database and
would make the published Z value inconsistent with the records searched.

HMMER 3.4 defines `-Z` as the total number of target sequences and explicitly
identifies a database split into multiple files for parallel search as a case
where the caller must supply the original total
([Jackhmmer manual](https://github.com/EddyRivasLab/hmmer/blob/9acd8b6758a0ca5d21db6d167e0277484341929b/documentation/man/jackhmmer.man.in#L264-L275)).
It also warns in source that iterative-ranking behavior may be strange when
target names are not unique
([HMMER top-hits source](https://github.com/EddyRivasLab/hmmer/blob/9acd8b6758a0ca5d21db6d167e0277484341929b/src/p7_tophits.c#L1010-L1092)).
That warning is a reason to preserve and measure the reference behavior, not a
license to remove targets.

AlphaFold also interprets MSA identifiers as metadata. In particular, it parses
the first token of protein MSA descriptions to obtain species identifiers used
downstream
([`msa_identifiers.py`](https://github.com/google-deepmind/alphafold3/blob/a3b8355da1694eb28a6e54047db8c29cb699b1fa/src/alphafold3/data/msa_identifiers.py#L26-L94)).
Permanently applying `seqkit rename`, or leaving an ordinal prefix in the
shards, could therefore change pairing semantics. A production-wide reheadering
scheme would require a reversible mapping and separate scientific validation;
it is not the Phase 1 fix.

## Published lossless preparation recipe

The official AlphaFold 3 guide recommends random record distribution so shards
have similar sequence-length distributions, then uses `split2 --by-part` to
produce the parts
([performance guide](https://github.com/google-deepmind/alphafold3/blob/a3b8355da1694eb28a6e54047db8c29cb699b1fa/docs/performance.md#L85-L120)).
The recommendation is about load balance. The example's direct two-pass shuffle
is not lossless for this small-BFD snapshot because its full headers are not
unique.

The published profile pins SeqKit 2.13.0, seed 23, and eight threads. It first
runs:

```bash
seqkit shuffle -j 8 --two-pass --update-faidx \
  --rand-seed 23 source.fasta
```

The app parses only diagnostics matching SeqKit's exact duplicate warning
format. Each warning identifies the omitted record's sequence-start byte
offset. The app seeks to that offset in the immutable source, verifies that the
preceding full header matches the warning, copies the exact sequence, and
appends the record to the shuffled FASTA under a unique temporary header:

```text
>__AF3_RECOVERED_{generation_uuid}_{record_uuid}__{original_header}
{original_sequence}
```

The generation namespace prevents the stripping expression from matching
unrelated source headers. A separate JSONL report records each source byte
offset, original header digest, sequence digest and length, and temporary UUID.
The expected record and residue deficits from the failed run are fixed
validation inputs, so a changed warning set is a hard failure rather than a
best-effort repair.

After recovery, the app runs `seqkit split2 -j 8 --by-part 64`, then applies an
anchored `seqkit replace -j 8` to remove only the active generation's temporary
prefix. SeqKit documents `split2 --by-part` as round-robin, and its
implementation streams records to successive parts
([`split2` documentation](https://github.com/shenwei356/seqkit/blob/v2.13.0/doc/docs/usage.md#L2780-L2825),
[`split2` source](https://github.com/shenwei356/seqkit/blob/v2.13.0/seqkit/cmd/split2.go#L297-L489)).

The source and repaired shards are compared using aggregate `seqkit stats` and
order-independent `seqkit sum`, and each final shard is scanned for the active
temporary namespace. SeqKit may reserialize FASTA line wrapping; scientific
identity is the parsed header-and-sequence record multiset rather than byte
identity with the monolith.

A direct `split2` recipe or a custom streaming randomized partitioner remains a
fallback if a future database snapshot changes SeqKit's warning behavior. Such
a change requires a new profile recipe version and scientific validation.

Co-locating records that share a target identifier could reduce one source of
cross-shard duplicate differences. That is an inference from the merge code,
not an AlphaFold or HMMER requirement, and it should not complicate Phase 1
unless the monolith comparison identifies duplicate placement as material.

## Search-space parameters on every shard

For small BFD, every shard invocation must use:

```text
-N 1
-Z 65984053
--domZ 65984053
```

`-Z` is the complete protein database's number of target sequences, not the
number in one shard, the number of shards, or the number of residues. HMMER
uses it for per-sequence E-value calculations. `--domZ` is the search space for
conditional domain E-values
([HMMER manual](https://github.com/EddyRivasLab/hmmer/blob/9acd8b6758a0ca5d21db6d167e0277484341929b/documentation/man/jackhmmer.man.in#L664-L675)).

AlphaFold's current protein pipeline deliberately passes the configured
full-database count as both Z and domZ, including for small BFD
([pipeline configuration](https://github.com/google-deepmind/alphafold3/blob/a3b8355da1694eb28a6e54047db8c29cb699b1fa/src/alphafold3/data/pipeline.py#L305-L378)),
and its Jackhmmer wrapper forwards both values to every shard
([command construction](https://github.com/google-deepmind/alphafold3/blob/a3b8355da1694eb28a6e54047db8c29cb699b1fa/src/alphafold3/data/tools/jackhmmer.py#L239-L259)).
The official known-issues note documents that omitting `--domZ` from sharded
search made domain inclusion roughly 100 times more permissive in some cases
([known issues](https://github.com/google-deepmind/alphafold3/blob/a3b8355da1694eb28a6e54047db8c29cb699b1fa/docs/known_issues.md#L17-L39)).

Independent shard jobs are scientifically valid for this pipeline because
AlphaFold runs these protein searches with one Jackhmmer iteration and rejects
sharded configurations with `n_iter != 1`
([wrapper guard](https://github.com/google-deepmind/alphafold3/blob/a3b8355da1694eb28a6e54047db8c29cb699b1fa/src/alphafold3/data/tools/jackhmmer.py#L99-L119)).
Running multiple iterations independently would be incorrect: each shard would
construct a different next-round profile rather than one profile from globally
included hits.

For later RNA work, Nhmmer's Z has different units: the full database's total
nucleotide count divided by one million. It is not a sequence count. The RNA
case is outside this small-BFD phase.

## Correct merge behavior

Splitting execution across threads or containers does not change the required
logical merge. Every successful shard must return both its A3M and `tblout`, and
one coordinator must pass the complete ordered collection to AlphaFold's
unchanged merge implementation. A partial merge is a failed search.

For Jackhmmer, the upstream merger:

1. maps table rows by first-token target name;
2. associates each A3M hit with its table row;
3. globally orders hits by E-value ascending, bit score descending, then name;
4. applies `max_sequences` once to the globally merged A3M.

These operations are in
[`_merge_jackhmmer_results`](https://github.com/google-deepmind/alphafold3/blob/a3b8355da1694eb28a6e54047db8c29cb699b1fa/src/alphafold3/data/tools/jackhmmer.py#L285-L340).
Do not concatenate per-shard A3Ms, truncate each shard and retain all survivors,
or add a new identifier- or sequence-based deduplication step. Use the upstream
merge, then allow AlphaFold's existing downstream MSA construction to perform
the deduplication it already defines.

AlphaFold explicitly warns that sharded and monolithic MSAs are not guaranteed
to be identical because duplicate hits are not recognized across shard-local
HMMER searches. The effect is usually extra low-ranked hits and is more visible
for deep MSAs
([Jackhmmer warning](https://github.com/google-deepmind/alphafold3/blob/a3b8355da1694eb28a6e54047db8c29cb699b1fa/src/alphafold3/data/tools/jackhmmer.py#L59-L70)).
The merger's name-keyed table also makes duplicate target identifiers
ambiguous. Consequently, no partitioning layout can promise byte-identical
MSAs with the current wrapper when duplicate identifiers are present.

## Does shard layout change scientific results?

For distinct target identifiers, one iteration, identical HMMER flags, and no
exact score ties, every target is scored independently. Searching every record
with the same full-database Z values and globally ranking the outputs makes
contiguous, random, and round-robin partitions statistically equivalent.
HMMER's own threaded Jackhmmer implementation merges worker hit lists, sorts
them, and applies thresholds after the scan
([HMMER source](https://github.com/EddyRivasLab/hmmer/blob/9acd8b6758a0ca5d21db6d167e0277484341929b/src/jackhmmer.c#L663-L683)).

The current AlphaFold integration is not completely partition-invariant,
however. Moving duplicate names between shards changes which duplicates HMMER
can observe together; exact ties can be resolved differently; shard-local hit
limits and the name-keyed merge can affect the final global cutoff. Therefore:

- randomization principally improves load balance;
- contiguous or round-robin splitting is not scientifically invalid;
- partition choice can still change a small, usually low-ranked portion of the
  produced MSA because of implementation-level duplicate and truncation
  behavior;
- monolithic comparison is a required scientific benchmark, not merely a
  performance baseline.

## Publication and benchmark evidence

The profile remained in staging until all of the implemented publication checks
passed:

- exactly 64 correctly named, nonempty shards exist;
- aggregate `num_seqs` is exactly 65,984,053;
- aggregate residue count exactly equals the source;
- `seqkit stats --all --tabular -j 8` succeeds for the source and every shard;
- `seqkit sum --all -j 8` matches between source and the union of shards;
- the recovered-record count and residue count match the observed deficits;
- each recovered record's warning name matches the source header at its byte
  offset;
- no final header retains the active `__AF3_RECOVERED_...` namespace;
- shard record, residue, and byte distributions are recorded and reviewed;
- the source digest, each shard digest, SeqKit version, seed, thread count,
  transformation commands, and validation outputs are durable before
  publication.

`seqkit sum` is necessary but not sufficient because it deliberately ignores
headers while hashing the order-independent sequence multiset
([SeqKit documentation](https://github.com/shenwei356/seqkit/blob/v2.13.0/doc/docs/usage.md#L864-L887)).
The Phase 1 implementation does not compute a generic order-independent digest
over every parsed `(full header, sequence)` pair. For this fixed snapshot, it
instead validates every observed omission against the source byte offset and
combines that evidence with exact record, residue, and sequence-multiset
agreement. A reusable production builder should close this remaining
generality gap or explicitly version and document an equally strong
snapshot-specific recovery contract.

The published profile contains 65,984,053 records and 16,748,600,902 residues.
Its source is 18,171,626,364 bytes, and its maximum per-shard residue imbalance
is 0.267%. Its manifest SHA-256 is
`f9215302fcb4426385777fb85052e2031c2cd3b11b3a0b8fedde30ca5406534a`.

The approved pembrolizumab VH comparison used identical HMMER, Z/domZ,
filtering, truncation, and merge settings. Every tested sharded topology had
full unique-hit Jaccard 1.0 against the monolithic oracle, zero score
mismatches, and zero sequence mismatches. Its only ordering differences were
permutations inside equal E-value and bit-score groups. The detailed timings
and scope limitations are in
[the Phase 1 results](alphafold3-msa-phase1-results.md).

GroEL was not submitted after the user replaced the repeated matrix with a
focused one-shot topology sweep. RNA databases also remain untested; neither
protein result should be generalized to Nhmmer without an RNA oracle
comparison.

## Source audit scope

This conclusion was checked against:

- AlphaFold 3 official `main` at
  [`a3b8355`](https://github.com/google-deepmind/alphafold3/commit/a3b8355da1694eb28a6e54047db8c29cb699b1fa),
  dated 2026-07-21;
- the Biomodals fork's relevant upstream base
  [`5a3d6b6`](https://github.com/google-deepmind/alphafold3/commit/5a3d6b63656038fbb5285d405cd3389b190a5774),
  whose sharding, Z/domZ, and merge semantics are the same for this decision;
- SeqKit 2.13.0 at
  [`d13b5fa`](https://github.com/shenwei356/seqkit/commit/d13b5fa388cc869de05abe1bdb07980eef5efb4e);
- HMMER 3.4 at
  [`9acd8b6`](https://github.com/EddyRivasLab/hmmer/commit/9acd8b6758a0ca5d21db6d167e0277484341929b).

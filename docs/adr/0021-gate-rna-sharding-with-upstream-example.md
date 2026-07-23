# Gate RNA sharding with the upstream example

Status: accepted.

The initial RNA Sharding Oracle uses the 25-nucleotide sequence from
AlphaFold3's documented modified-RNA example:
`GGCCCGAUAGCUCAGUCGGUAGAGC`. Its length exercises the pinned Nhmmer wrapper's
special short-RNA filter. The fixture is acceptable only if the monolithic
search produces at least one non-query hit; a query-only result does not pass
the gate and requires selecting a longer documented RNA query first.

For RFam, RNAcentral, and NT-RNA separately, the oracle compares monolithic and
sharded hit identities, scores and E-values, and aligned-sequence multisets
using the same full-database Z value. It then compares the Combined Unpaired
MSA produced by deduplicating in RFam, RNAcentral, and NT-RNA order. Ordering
differences are scientifically equivalent only among exact score ties; other
hit, score, alignment, or final-MSA differences fail the gate.

This test must pass against the exact pinned AlphaFold/HMMER behavior before
RNA sharded profiles are selected by the production app.

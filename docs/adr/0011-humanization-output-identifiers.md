# Preserve humanization output identifiers

Status: accepted.

FASTA exports replace each whitespace character in a parental or candidate ID
with `_` before appending the chain suffix. CSV and Parquet tables retain the
original IDs, as does pair-seed derivation. Input validation rejects pairs
whose IDs collide after whitespace replacement, including `clone 1` and
`clone_1`, before dispatch.

p-AbNatiV2 uses short ASCII names for upstream structure work in each isolated
pair directory. Collected structures use `structures/pair_NNNN/`; the bundle
manifest's `structure_ids` mapping associates each directory with its original
parental ID. This avoids upstream dot truncation and filesystem byte limits.

The output changes advance Sapiens and Humatch output-protocol identities to 3
and p-AbNatiV2 and HuDiff-Ab wrapper-protocol identities to 3. Manifest schema
version 2 and scientific sequence-selection algorithms are unchanged. Existing
execution publications are not reused under the new identities.

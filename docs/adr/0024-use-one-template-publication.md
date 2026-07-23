# Use one template publication per sequence

Status: accepted.

Protein templates remain at the existing sequence-root
`/{prefix}/{sequence_hash}/templates.json` path. Production does not add a
template-identity directory. A `templates.done.json` marker written last binds
the file to the combined unpaired-MSA digest, PDB seqres and mmCIF reference
snapshot, maximum template date, pinned tool behavior, and template file size
and digest.

The pipeline always writes `templates.json`, including a valid empty list.
Existing unmarked template files are ignored as cache evidence and remain
untouched until a complete new Template Search Result is ready to replace them.
A missing or mismatched marker causes template search to rerun without
invalidating or rerunning the underlying Raw Database MSAs.

This retains only the latest validated template publication at the convenient
legacy path. Template identity remains a logical validation key and Search
Build Claim key even though it is not represented as a directory component.

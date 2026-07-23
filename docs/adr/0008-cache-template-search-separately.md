# Cache template search separately

Status: accepted.

Protein template search is a durable stage after Combined Unpaired MSA
assembly. Its identity binds the combined A3M digest, maximum template date,
pinned AlphaFold and HMMER versions, and result-affecting search parameters.
Reference-store identity is refined by ADR 0039. The stage always publishes
`templates.json`, including a valid empty list, followed by a validating
completion marker. A cache hit requires both artifacts to be valid.

The template worker reads PDB seqres and individual mmCIF source files directly
from the mounted database Volume. It does not copy the reference store into the
MSA cache or local SSD. The cache contains only the selected AlphaFold template
records, including their mappings and serialized structures; the pinned
pipeline limits these to four hits. A failed template stage can therefore retry
without rerunning any valid Raw Database MSA while avoiding a second bulk copy
of the PDB/mmCIF assets.

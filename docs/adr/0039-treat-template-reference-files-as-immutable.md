# Treat template reference files as immutable

Status: accepted.

The fixed upstream `pdb_seqres_2022_09_28.fasta` file and `mmcif_files/`
directory are treated as an Immutable Template Store. Template cache identity
does not include a reference label, directory inventory, file digest, or
runtime scan of those paths.

Template Search Result identity still covers the Combined Unpaired MSA,
maximum template date, pinned upstream/tool behavior, and result-affecting
parameters. The reference files are an operator-controlled infrastructure
invariant and will not be updated in place.

If that invariant is ever broken, existing template publications may be stale;
the unsupported change must be accompanied by explicit cache removal or a
future identity-policy revision. Avoiding reference versioning keeps template
lookup and search free of a large mmCIF metadata traversal.

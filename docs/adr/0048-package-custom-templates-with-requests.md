# Package custom templates with requests

Status: accepted.

A Request Retrieval Archive includes every Staged Custom Template referenced by
its Enriched AlphaFold Input under:

`custom-templates/{sha256}.cif`

While assembling the archive, the local entrypoint rewrites only the downloaded
input copy's `mmcifPath` values to those archive-relative paths. The durable
input, content-addressed template files, Inference Identity View, and canonical
Volume paths remain unchanged.

The request manifest declares the required template artifacts so retrieval
downloads no unrelated custom templates from the shared AlphaFold Run Root.
Inline mmCIF templates remain inline and do not create duplicate files.

This makes path-backed custom-template requests portable without duplicating
template files in every durable request directory or exposing Modal-only paths
as the archive's sole reference.

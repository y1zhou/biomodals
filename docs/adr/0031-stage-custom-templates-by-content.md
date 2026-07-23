# Stage custom templates by content

Status: accepted.

For every caller `mmcifPath`, the local helper reads the file and computes its
full SHA-256 before constructing the Inference Run Identity. The identity view
substitutes the digest for the path while retaining `queryIndices` and
`templateIndices`; local paths and basenames never affect identity. Inline
`mmcif` values use the same content-digest representation.

After `run_id` is known, each path-backed template is uploaded once to
`<run-root>/custom-templates/{sha256}.cif` and its `mmcifPath` is rewritten to
the mounted remote path. Identical content is deduplicated within the run.
Inline mmCIF remains inline.

Computing identity before path rewriting avoids a circular dependency between
`run_id` and its staging destination. Content-addressed naming also prevents
basename collisions and lets equivalent local paths share one staged object
without treating operational Volume paths as biological input.

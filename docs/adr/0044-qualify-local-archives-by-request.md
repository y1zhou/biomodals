# Qualify local archives by request

Status: accepted.

The local entrypoint names a Request Retrieval Archive:

`{presentation_name}_{request_id[:12]}_AlphaFold3.tar.zst`

The Inference Request identity already covers `run_id` and the normalized
requested seed set, so this filename distinguishes overlapping seed requests
without exposing another run digest. The caller's sanitized display name keeps
the archive recognizable.

Repeating an identical request resolves to the same local path. An existing
file is reused only after the entrypoint verifies that it is a non-empty,
readable tar archive; otherwise it fails rather than silently overwriting it.
Different requested seed sets cannot trigger the old display-name-only skip
behavior.

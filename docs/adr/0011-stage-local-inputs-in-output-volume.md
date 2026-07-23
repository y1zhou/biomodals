# Stage local inputs in the output Volume

Status: accepted.

Before submitting remote work, an AlphaFold3 local helper materializes
file-backed caller inputs. It reads protein and RNA `unpairedMsaPath` values and
protein `pairedMsaPath` values into their corresponding inline MSA strings, then
clears the path fields. All relative caller paths are resolved against the
input JSON's parent directory.

The helper also reads `userCCDPath` into inline `userCCD` and clears the path
before constructing run identity. CCD content participates in identity, while
its source path does not. Setting both non-empty `userCCD` and `userCCDPath` is
rejected as ambiguous.

Caller-supplied template `mmcifPath` files remain path-backed: the helper
uploads them with Modal Volume batch upload into
`<run-root>/custom-templates/` and rewrites each path to its mounted remote
location.

Content-addressed template naming and run-identity ordering are refined by ADR
0031.

The AlphaFold3 output Volume is mounted at that same AlphaFold Run Root during
inference. AlphaFold writes its output tree and logs below the run root and
commits the Volume before reporting completion; the Volume copy is the durable
result, while an optional local download is only a retrieval step. The MSA
cache and sharded database Volume remain separate stores.

This boundary prevents local-only paths from reaching a container without
embedding every custom mmCIF in the function arguments. It also keeps uploaded
custom templates beside the specific job that consumes them rather than in a
shared scientific cache.

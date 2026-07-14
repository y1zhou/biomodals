# Upstream Tools, Patches, and Scientific Equivalence

Pin upstream source. Treat changes to candidates, features, filters, references,
or reductions as scientific behavior changes even when schemas match.

## Execute upstream code safely

- Validate files, counts, IDs, and alphabets before upstream use; map labels to
  stable unique internal IDs.
- Invoke argv lists. If upstream builds shell strings, patch it or supply only
  generated IDs and validated domain text; quoting is not a trust boundary.
- Confine destructive/output paths to a fixed root. Reject traversal, symlink
  escape, absolute archive members, and unexpected outputs.
- Bound inputs and subprocess output; do not capture unused logs.

## Patch discipline

- Prefer a small patch asset or narrow build helper over embedded source.
- Guard rewrites with the exact preimage; fail if they do not apply. Test the
  rewrite against the pinned commit and compile/import the result.
- Document rationale, target lines/commit, scientific effect, and removal
  condition. Version behavior changes into every affected cache stage.
- Pin result-affecting base image, Python/CUDA/tools, dependencies, weights, and
  references. Record asset content identity across shared volumes.

## Scientific equivalence

For compatibility changes, compare with an external pinned oracle. For an
intentional fork, record upstream and corrected behavior, rationale, affected
outputs, and tests in an ADR; do not claim exact equivalence.

Compare the earliest meaningful evidence as well as final outputs:

- Candidate identities/counts and patched features/schemas.
- Per-backend raw evidence before reductions can hide mismatches.
- Final values/order/nulls/thresholds/failure policy.

Never use Biomodals as its own oracle. Keep pinned fixtures and boundary cases;
separate live equivalence from fast tests and record exact code/asset identities.

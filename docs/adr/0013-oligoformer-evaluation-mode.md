# OligoFormer efficacy inference uses evaluation mode

## Decision

For upstream commit `e2f53ad63387bbe166bf123949151e2bc9bf6ec3`, apply
`src/biomodals/app/score/oligoformer_inference.patch` before the existing stage
patches. The build checks the exact patch context and fails on drift.

The upstream inference function loads trained weights without calling `eval()`;
its encoders and classifier contain dropout. It also retains autograd during
prediction. Biomodals calls `eval()` after loading weights and decorates inference
with `torch.inference_mode()`. Both FASTA and manually supplied candidate paths
use this function. No candidate generation, model weights, or filtering rules
change.

This is an intentional scientific correction, not numerical equivalence with
unmodified upstream. Scores and rankings can change. The efficacy identity now
includes `eval-inference-mode-v1`; downstream evidence and final-table identities
already derive from that key, so old predictions cannot be reused.

## Validation and failure modes

Patch application must fail if the pinned preimage changes or the patch was
already applied. Compile the patched upstream file. Numerical acceptance requires
two fresh uncached runs and an independently executed pinned upstream reference
with the same explicit evaluation-mode correction. Retain input and checkpoint
digests, scores, candidate identities, ordering, and tolerances. GPU floating-point
variation must be distinguished from active dropout; bitwise reproducibility is
not promised across hardware.

Remove the patch when a newly pinned upstream provides both evaluation mode and
disabled gradient tracking, after repeating the scientific comparison. Do not
reuse old cache identities merely because the patch is no longer necessary.

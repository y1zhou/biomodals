# Split OligoFormer immutable assets from run intermediates

OligoFormer performance work will keep immutable upstream assets in the standard
model volume and store run-specific cached intermediates in the app output
volume. RNA-FM weights and full-human off-target references belong in the model
volume, keyed by upstream commit and reference content, while efficacy,
off-target, toxicity, and packaging intermediates belong in the output volume,
keyed by app version, input FASTA content, optional siRNA FASTA content, flags,
thresholds, and reference hashes.

This preserves the model volume as a shared read-mostly asset store while
allowing reruns to reuse expensive per-input work without baking user inputs or
derived results into an image or model cache. The first implementation will not
auto-prune output-volume cache entries; retention policy can be added after the
runtime split has real usage data.

The first split preserves OligoFormer's standalone tarball return contract. The
output-volume cache is internal until a workflow consumer needs durable
`AppRunResult` outputs.

GPU efficacy and CPU post-processing will hand off through upstream-shaped files
in the output volume rather than an in-memory Python result object. This keeps
the cache inspectable, preserves upstream output formats, and avoids invasive
changes inside `scripts/infer.py`.

The wrapper may patch upstream minimally to add explicit stage selection, such
as efficacy-only execution and CPU post-processing from existing efficacy
outputs. That is preferable to running the full upstream pipeline and then
reverse-engineering partial state because it gives Modal a clear GPU/CPU
boundary while keeping upstream file formats as the contract.

The local entrypoint will orchestrate the GPU efficacy function and the CPU
post-processing function as separate remote calls. GPU functions should not own
downstream CPU scheduling in the first split; keeping orchestration local makes
each function single-purpose and keeps retry behavior visible at the app
boundary.

When final output files are already present in the output-volume cache, reruns
should skip GPU efficacy and CPU post-processing compute. They still need a
small remote packaging path so the standalone entrypoint can return fresh
tarball bytes without exposing output-volume paths as public API.

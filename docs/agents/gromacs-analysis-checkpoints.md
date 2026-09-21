# GROMACS analysis publication

`collect_traj_stats` is shared by standalone GROMACS, the service and ShortMD.
The streaming statistics implementation uses an app-owned `ContentBoundFileSet`
covering all CSVs, requested PNGs and the aligned final-frame PDB. Identity binds
the processed XTC and template digests, analysis policy, pinned analysis packages
and child plot title. Stable per-output temporary paths are overwritten on
recovery; the complete marker is published last. Missing or corrupt output
replays analysis, never MD. Unmarked historical CSV/PNG files are not cache hits.

The old timestamp-only CSV/PNG exception is retired. Only the established native
PBC postprocessing path retains its raw-XTC modification-time invalidation.
Cross-worker input reads reload the Volume. A stale processed-XTC deletion is
committed before remote postprocessing, and the caller reloads its outputs.
Same-container CSV/plot work adds no intermediate Volume barriers.

See the [continuation specification](../specs/gromacs-continuation.md#streaming-analysis)
for scientific definitions, versioning, bounded plotting and verification.

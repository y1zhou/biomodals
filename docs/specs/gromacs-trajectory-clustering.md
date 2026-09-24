# GROMACS trajectory clustering

Status: implemented with offline native/API/browser verification; deployment verification pending.

Researched 2026-09-23 against backend `aa7d853` and GROMACS 2026.1.
For existing cumulative-run semantics, see
[Production Continuation](gromacs-continuation.md).

## Clustering Job outputs

- A CSV assigning each trajectory frame a cluster ID, with `is_medoid` equal
  to `yes` for medoid frames and null otherwise (empty CSV field).
- A directory of medoid PDBs named `{run_name}-frame_{frame}.pdb`, using a safe
  presentation filename without changing scientific identity.
- Compact provenance binding the source trajectory and clustering settings.
  Do not duplicate the simulation trajectory or other MD artifacts in this
  archive; the source-job link provides their existing download.

Frame indices are zero-based in the full processed production trajectory;
include `time_ns` alongside `frame,cluster_id,is_medoid`. Retain native
one-based cluster IDs. All frames participate; no sampling.

## Verified native capability

GROMACS `gmx cluster` supports RMSD-based clustering and selecting the member
with the smallest mean distance to other members. That is an actual sampled
structure, not an averaged structure. GROMOS clustering uses an RMSD cutoff;
it does not ask for a fixed number of clusters. The default native method is
linkage, so a chosen GROMOS policy must be explicit. Native outputs include
membership, times and central structures, but need conversion into the
requested CSV and individually named PDBs.

Source: [GROMACS 2026.1 gmx cluster](https://manual.gromacs.org/2026.1/onlinehelp/gmx-cluster.html).

## Local integration and scaling

- `app/bioinfo/gromacs/analysis.py` streams processed protein coordinates in
  128-frame chunks. Preserve the current full-resolution RMSD/Rg/RMSF contract;
  a clustering extension must not load the whole cumulative trajectory merely
  to reuse those calculations.
- `assets/postprocess-traj.sh` produces the PBC-corrected protein trajectory.
  Its `-pbc cluster` makes molecules spatially whole; it is not frame clustering.
- `assets/gmx_mdp/production.mdp` writes every 5000 steps at 0.002 ps per step:
  approximately 25,000 frames at 250 ns. All-pairs comparison then involves
  approximately 312 million frame pairs. Continuations grow beyond this.
- Exact within-cluster medoids and medoids selected on a bounded sample are
  different contracts. Assigning all remaining frames to sampled medoids
  does not make those representatives exact medoids of the full clusters.
- The existing analysis image has numerical packages; the simulation runtime
  image has GROMACS. Native clustering could run as a separate CPU execution
  node in the analysis-only Job using the source's completed production
  analysis, without adding another clustering library. It does not become a
  node in fresh or continuation simulation Jobs.
- `execution.py`, `execution_runtime.py` and `service/gromacs/archive.py`
  publish and validate exact output inventories. A variable number of medoid
  files requires a content-bound inventory, not archive globbing. The new
  analysis-only plan/archive must be distinct from unchanged MD publications.
- The frontend already downloads the complete archive. Archive-only cluster
  outputs do not require a new molecular viewer or download endpoint.
- Native medoid logs use times rather than original frame ordinals. A pinned
  fixture must establish an unambiguous frame mapping; rounded times must not
  be treated as unique frame IDs. Pairwise fitted RMSD also differs from
  distances between frames aligned once to a common reference.
- Continuations preserve their parent's native file stem. New PDB presentation
  filenames must deliberately use the chosen child identity; do not inherit
  a misleading name accidentally. Recluster the cumulative history rather
  than appending memberships to old clusters if full-history scope is chosen.

## Accepted direction (2026-09-23)

- Clustering is available only through the completed simulation's results
  page. Do not add clustering options to fresh or continuation submission
  forms, and do not change ordinary simulation outputs. This supersedes the
  earlier optional-on-submission decision and removes the proposed mixed
  MD-success/clustering-failure Partial outcome. A clustering-only Job can
  fail independently without affecting its source simulation.
- Use cutoff-based GROMOS clustering rather than a requested cluster count.
- Compare protein C-alpha atoms and export complete processed protein medoids.
  The existing processed trajectory already excludes solvent and ligands.
- Expose an editable cutoff initially set to 2 angstroms (0.2 nm for GROMACS).
  This is a configurable starting point, not a universal structural threshold.
- Cluster the complete cumulative production history on extensions, not only
  the additional segment, and exclude equilibration.
- Users must be able to cluster an existing completed simulation without
  running additional MD. A **Cluster trajectory** button on a completed
  GROMACS Job opens settings and creates a linked GROMACS analysis Job, not
  another Tool. Its results page links back to the source simulation Job.
  Preserve the original Job and archive without mutation. This replaces the
  initially suggested zero-nanosecond extension UX.
- Use native all-frame GROMOS clustering and native medoid selection. Warn
  about estimated large memory/work requirements using offline evidence;
  do not reject solely for trajectory size or estimated computational cost.
  Do not silently subsample frames or introduce nearest-representative
  approximation.
- Apply a 12-hour clustering-task execution deadline. Oversized-trajectory
  warnings do not prevent explicit submission. Actual runtime exhaustion or
  memory failure terminates the analysis Job with an explanatory error,
  leaving its source unchanged. Ordinary authentication, input-integrity and
  reference-validity checks remain blocking; resource warnings do not bypass
  those checks or the platform's finite memory/disk capacity.

## Execution boundary

The linked analysis Job reuses normal progress/cancel/download handling and
runs against retained trajectory data, not a zero-step `mdrun`.
Current continuation validation and provenance require positive additional
duration; changing a UI minimum to zero would not implement analysis-only
behavior. Analysis requires a matching trajectory/topology, not a restart
checkpoint or a copied append-state directory.

Calculation clarification: native `gmx cluster -method gromos -fit -noav`
already provides the partition and medoid structures for all frames it reads.
Use C-alpha fitting/comparison and protein coordinate output; convert native
membership/representatives into the requested archive format. Native `-skip`
does not assign omitted frames. The previously proposed bounded-sample option
would therefore require an additional nearest-representative assignment pass
and is an approximation, not merely a native performance flag. That approach
is not selected for this release.

## Approved implementation plan

1. Add a source-linked GROMACS analysis operation using the existing execution
   kernel, ownership, admission, cancellation and download lifecycle. Keep
   normal MD plans/archives and ShortMD unchanged. Expose the same analysis
   definition through the app's coordinator-aware CLI and service.
2. Verify the source's content-bound processed protein trajectory and matching
   atom template remotely, with no API-host download or restart-state copy.
   Report missing/incompatible retained input without running MD to recreate
   it. Treat the source as read-only throughout.
3. Run native GROMOS over all frames, fit/compare C-alpha atoms, and retain
   native one-based cluster IDs and native tie behavior. Use source binary
   times/member ordinals for exact zero-based frame identity; reject ambiguous
   timestamps rather than guessing. Export actual whole-protein source frames
   corresponding to the native medoids, not averaged structures.
4. Estimate frame/atom/workspace demand and show advisory warnings before
   submission, without a hard trajectory-size or estimated-work admission
   ceiling. Enforce the 12-hour execution deadline. No sampling or truncation
   fallback. Keep native intermediate matrices out of downloads. Protect
   against native failures and partial publications.
5. Publish an exact content-bound CSV/PDB inventory and compact provenance,
   then package a clustering-only archive using existing download helpers.
   Validate full frame coverage and exactly one correctly named medoid per
   cluster. Source simulation outputs remain unchanged on success or failure.
6. Add the completed-MD **Cluster trajectory** action and small cutoff form.
   Add server-owned Job operation/source metadata so analysis Jobs reuse the
   shared detail page without displaying MD trajectory plots or Extend
   simulation actions. Link to the source in every analysis Job state.
7. Verify native cluster/medoid equivalence, exact frame mapping, full protein
   PDBs, cumulative-history input, owner isolation, idempotent submission,
   nonblocking resource warnings, deadline enforcement, cancellation,
   publication integrity, independent
   failures and frontend lifecycle. Commit cohesive milestones.

No clustering result viewer, new Tool catalog entry, new MD-submission
controls, or mutation of historical archives is included. Deployments and paid
smoke runs require separate authorization.

## Implemented contracts and rollout

The app request discriminates `trajectory_clustering` from MD without artificial
duration or velocity settings. Its graph is `cluster_trajectory` →
`prepare_result`; the ordinary MD graph is unchanged. The worker reloads the
Volume before reading, rechecks the source publication identity and hashes the
trajectory before native analysis. Source inspection reads the published
template/statistics plus trajectory existence/size, not trajectory contents.
It shares overlapping same-source/deployment requests, with a 45-second service
deadline and a two-container inspector cap.

The worker builds `clusters.zip` with `clusters.csv`, `medoids/*.pdb`, and
`provenance.json`. Provenance contains source identity, cutoff, indexing/coordinate
conventions, policy version, and the exact CSV/PDB size/digest inventory. The
kernel content-binds the complete ZIP as one publication; the API downloads it
with the shared transfer helper and verifies that digest before caching. Native
matrices and binary medoid scratch stay outside the published directory. The
source frame count must match its published statistics; ambiguous binary times
fail explicitly instead of being mapped through rounded native logs.

The CPU worker has a 12-hour deadline and 64 GiB memory ceiling. Metadata estimates
`8 * frames² + 12 * frames * CA_atoms` bytes; estimated memory ≥8 GiB or
pairwise atom work ≥10¹¹ triggers a warning, not rejection. These conservative
estimates omit some native/scratch overhead and are not guarantees of completion.

Authenticated GET/POST `/api/v1/gromacs/jobs/{job_id}/clustering` share ownership,
CSRF and normal admission. POST accepts optional `display_name` and a finite
positive `cutoff_angstrom` (default 2; no arbitrary upper bound). Exact idempotent
replay precedes source/deployment I/O. Shared Job views persist `operation` and
`source_job_id` in every state; analysis Jobs are neither MD continuation sources
nor MD-plot providers. The SQLite 8 → 9 upgrade preserves existing records; see
the [service database contract](api-tool-service.md#database-upgrades-and-historical-cutover).

CLI (one existing entrypoint, so fresh/continuation command resolution is unchanged):

```bash
uv run biomodals app run gromacs::submit_gromacs_task --environment production --version <version> -- --cluster-from <execution-run-id> --clustering-cutoff-angstrom 2 --run-name trajectory-clusters
```

MD settings cannot be combined with `--cluster-from`. Use `biomodals run restart`
for compatible recovery of the same analysis plan. Deploy the updated GROMACS
app, pin its exact version, then restart the matching API and deploy the frontend.
Back up SQLite before the API upgrade; do not delete deployments existing Jobs
still reference. The AF3 chemistry feature separately requires its updated app
deployment. Neither humanization workflow needs redeployment for this release.

Offline tests cover native medoid/RMSD agreement, rounded-time collisions, full
protein PDBs, exact sparse assignments/inventory, source-only metadata inspection,
warm-worker reload ordering, changed-input rejection, CPU graph/recovery, archive
transfer integrity, CLI dispatch, owner/replay/admission and database preservation.
Browser verification uses deterministic fake science; it does not verify Modal
image installation, cold starts, or production-scale 12-hour execution.

## Offline native feasibility (2026-09-23)

Local GROMACS 2026.1 mixed-precision, one thread, synthetic 200-residue protein
with 200 C-alpha fitting atoms and 800 protein output atoms, GROMOS cutoff
0.1 nm: single native invocations at 100/500/1000 frames took
0.04/0.40/1.58 seconds and peaked at 23.5/28.8/37.4 MiB RSS. The fixture used
0.1 nm for validation, not the accepted 0.2 nm product default. These small
algorithmic fixtures are not production benchmarks or defensible hard limits.

Native binary medoid TRR timestamps, joined to unique source XTC timestamps,
plus native member frame indices recovered exact zero-based frame identities
despite deliberately colliding rounded text-log times. An independent
within-cluster pairwise RMSD calculation matched all three 100-frame medoids.
Output structures retained all protein atoms. Duplicate/nonfinite timestamps
need explicit handling; do not guess from a rounded log. Distance matrices
and cutoff-neighbor lists can both grow quadratically.

Temporary reproducibility report/harness:
`/tmp/gromacs-cluster-native-2vj8cB/REPORT.md` and `benchmark.py`; not committed
production instrumentation. Larger atom/frame and dense-neighbor cases are
needed to calibrate resource estimates; these measurements are not admission
ceilings.

Follow-up 2026-09-24, local single-thread GROMACS 2026.1 at the accepted
0.2 nm cutoff, single measurements (no production timing guarantee):

| Frames | C-alpha atoms | Clusters | Native seconds | Peak RSS MiB |
| --- | --- | --- | --- | --- |
| 5000 | 200 | 3 | 30.20 | 162.9 |
| 5000 | 200 | 1 (dense neighbors) | 29.94 | 226.3 |
| 1000 | 1000 | 3 | 3.10 | 40.5 |

For this follow-up the native output group was C-alpha, reducing retained
coordinates; a bounded pass over the original protein trajectory extracted
whole-protein medoid frames. The small paired comparison produced identical
memberships and medoid indices to native Protein output, with fitted full
coordinates agreeing within 0.000006 angstrom. This optimization preserves
all frames and the C-alpha clustering metric; it is not sampling.

At fixed 200 C-alpha atoms, 25,000 frames imply approximately 25 times the
pair work of 5000 frames, or roughly 12.5 minutes by simple extrapolation.
That size was not measured and this is not a runtime promise. Cutoff-dependent
neighbor storage, frame count and atom count all matter; a frame-only limit
is insufficient. Record/run provenance under
`/tmp/gromacs-cluster-budget-KrZuZ2/attempt2/results.json`.

Accepted operational policy: a 12-hour clustering-task deadline, with
nonblocking estimated memory/work warnings instead of oversized-input errors.
The earlier proposed one-hour deadline and hard workload guards are not
selected. Warning thresholds and validation timeout/concurrency limits are
implementation tuning, not biological rules or trajectory-size admission
caps. Resource estimates must be labeled as estimates, not promised runtime
or guaranteed completion.

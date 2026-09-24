# GROMACS production continuation

Status: accepted, 2026-09-16. Implemented; deployed smoke verification pending.

## Product contract

A Production Continuation adds production MD from a completed simulation's
checkpoint. It creates a new linked Job and root Execution Run; the source's
files, status, ledger and Result remain unchanged. It is not a rerun from a
structure, compatible kernel restart, or Retry fetching results.

- Fresh duration and each added interval accept **1–250 whole nanoseconds**.
  Added time is required. Cumulative time may exceed 250 ns.
- Completed children may be continued; multiple children may branch from one
  source. Siblings share checkpoint state, not independent random seeds.
- Editable settings are the new display name, added time and CPU/GPU mode.
  Mode defaults to the source. All physical settings, velocities, random
  state, thermostat/barostat state and atom ordering are inherited. Hardware
  changes do not promise bitwise-identical trajectories.
- Results cover the complete cumulative production trajectory. RMSD keeps the
  original first-frame reference; RMSF is recomputed over the whole history.
  NVT/NPT analyses are inherited, not rerun or concatenated.
- Website, API and CLI use the app-owned execution graph. Generic scheduling,
  admission, cancellation and compatible-restart rules do not change.

See [ADR 0011](../adr/0011-gromacs-continuations-preserve-source-runs.md) for the
immutable-source trade-off. Accumulated storage and analysis cost grow with
history. Coordinate analysis is streamed in bounded chunks, but copying,
postprocessing, hashing and output size still grow with cumulative history.
There is no hidden lifetime-duration cap. Service admission requires a Completed source;
repair failed local Result preparation first using Retry fetching results.

## Native implementation

The discoverable module is
[`bioinfo/gromacs/app.py`](../../src/biomodals/app/bioinfo/gromacs/app.py).
App-only graph, request, publication and continuation modules live alongside
it. The catalog name `gromacs`, Modal app `Gromacs`, Volume
`Gromacs-outputs`, existing function names and `bioinfo` tag stay unchanged.
ShortMD imports the package without adding continuation to its own workflow.

GROMACS remains pinned to **2026.1**. Full checkpoint state is required, not a
PDB/GRO or last XTC frame. The source TPR is extended once using
`gmx convert-tpr -extend <additional_ns * 1000>`. Production runs the fixed
extended TPR with `mdrun -cpi <checkpoint> -append`, without deriving remaining
steps from XTC timestamps or extending the child's TPR again on redelivery.

A child uses a new directory but retains the native source file stem:
checkpoint append validation requires original filenames. Every
checkpoint-referenced output is copied into independent child files, never
hardlinked to the source. Appending preserves cumulative trajectory/energy.
No new preparation, minimization, NVT or NPT simulation Tasks run.

### Source validation and recovery

Completed status alone is insufficient. A small read-only Modal CPU function on
the current target deployment reads the saved request inside the mounted Volume
and uses its final/production file records to locate production TPR, raw
XTC, energy, checkpoint and log. A required file absent from a record is resolved
by its native filename in the same source run directory. No other runs are
searched, and missing or empty files fail with the filename in the error.
Only file metadata is inspected: no checkpoint/trajectory content is read or
copied. The function returns compact settings, source locators and recorded
digests; no PDB, checkpoint, trajectory or other file payload reaches the API
host. Checkpoints larger than 64 MiB are rejected by size. Source and target
must share a Modal environment and the pinned GROMACS scientific version.

Continuation reads each publication's own inventory, not today's exact list or
ordering. Scientific identity, record validity, unique safe filenames and
recorded sizes/hashes remain enforced; conflicting records fail rather than
fall back. Historical records that omit energy/checkpoint/log can therefore use
retained native files without rewriting the old publication. Version 3 and later
production publications additionally bind checkpoint and log. This policy is
local to continuation: ordinary cache and result-publication validation stays
unchanged.

The child request binds source Execution Run ID, directory, native stem,
cumulative endpoint, canonical request digest, final-publication digest and
published checkpoint SHA-256 when available. Historical sources without a
published checkpoint digest are verified by native append checks and have their
actual digest recorded during preparation. New child requests reference the
original PDB by digest instead of embedding it. Existing inline-input requests
retain their encoding and remain readable. No files are copied until the new
Job is admitted. Its normal CPU preparation Task then:

1. Rechecks source records, the saved request identity and required files.
   Unlisted files use the same-directory fallback; invalid records never do.
2. Reads the TPR parameters and checkpoint through the pinned native binary.
   Checkpoint step/time must equal the completed TPR endpoint and saved time.
3. Checks native append filenames, offsets, sizes and checksums. GROMACS uses
   MD5 over up to the last 1 MiB ending at the saved offset; this is native
   compatibility checking, not the application's SHA-256 content identity.
4. Copies native state, original PDB/MDP, inherited NVT/NPT analyses and source
   TPR into a child-specific temporary directory on the Volume. Validates all
   recorded hashes, the original-input and checkpoint digests and native append
   compatibility on that copied snapshot before promoting files into the child
   directory. Extends the TPR once and publishes immutable preparation evidence.

The checkpoint's complete append inventory is authoritative, including any
additional output beyond XTC/energy/log. All those files are copied, with no
`-noappend` or trajectory-concatenation fallback. PDB/MDP and inherited analyses
are retained for the existing archive contract, not as MD restart prerequisites;
old production figures and processed trajectories are regenerated and do not
gate restart discovery.

Checkpoint dumps are consumed completely: GROMACS can print a corruption
warning without a failing exit code. Large dumps use container-local temporary
directories, removed after validation even on failure; they are not retained in
the output Volume, Python lists or streamed provider logs.
Missing/corrupt/incompatible state fails explicitly, with no fallback to a fresh
simulation.

Only a validated staging snapshot is promoted, with atomic per-file renames,
so an interrupted checkpoint copy cannot be mistaken for child MD progress.
Staging is child-owned and cleaned on normal exit/interruption. Recovery by the
sole admitted preparation writer also removes abandoned child staging left by
hard termination. This does not sweep sibling directories, infer ownership
from elapsed time, or remove promoted checkpoint progress. Staging is never
created by form reads and never moves or modifies the completed source.
Preparation redelivery reuses its content-bound publication. It never copies
the source over child progress. Production redelivery keeps the fixed target
and current child checkpoint. Explicit append mode rejects a missing checkpoint.
Native endpoint validation precedes completed production publication.

Workers reload the Volume before reading another worker's outputs. Analysis
commits stale processed-trajectory deletion before remote postprocessing, then
reloads the postprocessor's committed outputs before opening them.

Plan version 4 includes immutable `continuation.json` in the child's final
content-bound publication. The API verifies its size, SHA-256 and plan/source
identity, then embeds it once in the archive's existing
`metadata/provenance.json`. It includes the actual validated source checkpoint
SHA-256, step and time even when the original source publication omitted its
checkpoint. The submitted request is never rewritten. Older plan publications
keep their exact inventories and historical archives keep their original
request-derived metadata; unbound JSON is not retroactively trusted.

### Streaming analysis

`analysis.py` reads the processed XTC in 128-frame chunks with Biotite's
`XTCFile.read_iter`, selecting the same amino-acid atoms as before. Every chunk
is superimposed on the original first frame using all selected protein atoms,
never its own first frame. RMSD uses that fixed reference; radius of gyration
keeps Biotite's standard atomic-mass weighting. Coordinates and scores remain
in angstroms; time is converted from ps to ns.

Cα RMSF uses float64, chunk-merged central moments (Welford/parallel variance)
after full-protein alignment, with population variance over every frame. It
does not average per-chunk RMSFs. Only a chunk, fixed reference, last frame and
per-Cα mean/M2 state are retained: coordinate memory is independent of the
number of frames. Streaming changes reduction order, so equivalence is
numerical within tested tolerance, not a bitwise guarantee.

RMSD/Rg CSVs are written incrementally; RMSF is written after the final frame.
All rows and the established five-decimal CSV format remain. PNGs use at most
5000 evenly spaced rows, including both endpoints; sampled plots explicitly
say they are overviews and point to the full-resolution CSV. Sampling may omit
brief extrema; it never changes CSV statistics. The aligned final-frame PDB
and all public output filenames remain unchanged.

Protein selection follows GROMACS 2026.1 `residuetypes.dat`, including force-field
protonation, disulfide and terminal names such as `HIE` and `CYX`. Names,
coordinates and atom order are preserved; water, ions and ligands are excluded.
This avoids the PDB chemical-component classifier silently omitting valid MD
residues from alignment, RMSD, Rg and Cα RMSF.

Analysis publications bind processed-XTC/template SHA-256, `streaming-v2`,
pinned numerical/plotting packages and the child title to exact output digests.
Temporary CSV names are stable and overwritten after interruption; incomplete
or old-policy outputs cannot become a cache hit. Plan version 5 covers corrected
native protein selection; version 4 introduced streaming analysis and validated
continuation provenance. ShortMD also declares the
analysis policy in its scientific identity. Historical plans retain their
fingerprints; a new deployment does not silently upgrade an old Run.
Historical published statistics remain unchanged and may omit force-field
aliases under the old classifier. Continuing a source recomputes cumulative
production statistics under the new policy; inherited NVT/NPT outputs remain
the source's original publications.

Offline tests compare streamed statistics to pinned Biotite 1.6.0 batch
calculations across chunk sizes 1, 2, 7 and 128, with rigid displacement,
internal motion, non-protein exclusion, XTC quantization and uneven tails.
They also check units, complete CSV rows, child plot titles, final-frame
coordinates, bounded plots, cache tampering and interrupted-write recovery.

A local synthetic reducer benchmark (2026-09-17, Python 3.12, pinned analysis
packages, 2000 protein atoms, 128-frame chunks, median of three fresh processes)
measured the following whole-process peak RSS:

| Frames | Previous batch reducer | Streaming reducer |
| --- | --- | --- |
| 1000 | 285 MiB / 0.150 s | 146 MiB / 0.151 s |
| 8000 | 1226 MiB / 1.064 s | 146 MiB / 1.008 s |

The harness generated deterministic float32 coordinates with internal motion,
then computed first-frame alignment, RMSD, mass-weighted Rg and Cα RMSF. It
excluded XTC decoding, filesystem hashing, CSV/PNG output and GROMACS; these
are reducer measurements, not end-to-end or deployed throughput guarantees.
The new cache identity adds one sequential processed-XTC hash pass before
decoding; output validation hashes the much smaller analysis files. Benchmark
code is not part of production or the committed test suite.

## API and website

Owner-scoped routes under `/api/v1/gromacs`:

- `GET /jobs/{job_id}/continuation` returns `GromacsContinuationInfo`: source
  Job ID/name, cumulative `simulation_time_ns`, inherited `cpu_only`, direct
  `parent_job_id`, eligibility/code/detail and interval limits. Missing inputs
  may leave duration/mode null. GET neither repairs state nor launches science.
- `POST /jobs/{job_id}/continue` accepts `GromacsContinuationSubmission`:
  optional `display_name` (120 characters; blank generates a name), required
  `additional_time_ns` and `cpu_only`. Existing session/CSRF and
  `Idempotency-Key` protections apply; response is a normal 202 JobView.
  Exact replay precedes source I/O and preflight. New requests recheck source
  eligibility, pin the current compatible target, snapshot current provider
  limits and use normal atomic admission.

Nonexistent/foreign sources return 404. Ineligibility codes are
`source_not_completed`, `source_unavailable` and `source_environment_mismatch`;
an old target returns `deployment_incompatible` before admission. Client-supplied
source paths or endpoints are never accepted.

The source check has a 45-second API deadline, including lookup, queue/cold-start
latency and the metadata RPC. The read-only inspector has a 40-second execution
timeout and may incur a small CPU charge; it does not launch molecular dynamics.
Both metadata GET and the submission's pre-admission source recheck return HTTP
504 with code `source_check_timeout` on expiry. No Job is admitted on timeout;
exact submission replay still precedes source I/O. The API stops waiting on
expiry; an already-running read-only inspector remains bounded by its own
timeout. The form visibly indicates an active check and
offers an explicit retry after failure; it never automatically submits science.
Overlapping GET/POST checks for the same source and target deployment share one
in-flight inspection per API process. A caller disconnect does not cancel the
shared check or extend its deadline. Completed and failed checks are discarded;
later requests inspect again. Different sources/targets remain independent,
without an additional inspection concurrency setting.

Extend simulation opens a dedicated form: source link/endpoint, optional
name, blank added-time input and inherited CPU/GPU mode. Eligibility is checked
there, not on every Job-detail read. Opening/reloading never submits.
Lost-response recovery keeps exact JSON/key per owner/source, separately from
fresh simulation intent, and permits unchanged replay even if eligibility
subsequently changes.

The child timeline shows Verify continuation source, Run production, Analyze
production and Prepare result, with no fictional NVT/NPT execution. Existing
PNG and archive helpers serve the cumulative outputs. Archive provenance
identifies source/digests, added/total time, cumulative scope and inherited
analyses. Retained `production.mdp` is original input; the extended TPR is
authoritative for the new duration. Historical archive bytes remain unchanged.

## CLI and rollout

Fresh runs accept either `biomodals app run gromacs -- input.pdb` or the explicit
`--input-pdb input.pdb` spelling.

Use `biomodals app run gromacs -- --continue-from <execution-run-id>
--additional-time-ns 10 --run-name continued-example`.
`--cpu-only` selects CPU; `--no-cpu-only` selects GPU; omission inherits.
No PDB is needed. Continuation is mutually exclusive with `--input-pdb` and
the outer `--restart-from` compatible-recovery flag. Physical settings remain
inherited. Normal global container limits apply to the new root.
Explicit fresh-only duration, PDBFixer, seed, thread-count or OpenMP options
are rejected in continuation mode rather than silently ignored, even when they
equal a fresh-run default. Plot titles use the child run identity; filenames
retain the native source stem for checkpoint compatibility.
Use `biomodals run restart` for same-plan recovery of a continuation.

Deploy the updated GROMACS app including `inspect_continuation_source`, pin its
version in the API, restart the API and
deploy the matching frontend. New plan-version-four requests require the updated
coordinator and analysis image. API preflight checks for the preparation
entrypoint; the coordinator additionally checks scientific versions before
admitting work. Upgrade the API and pinned deployment together. Source metadata
reads also require the updated inspector;
they use the current target, not the source Job's historical deployment.
Existing Jobs retain their deployment snapshots; do not
delete deployments they still need.

Offline verification covers package discovery/ShortMD imports, source integrity,
append checksums/endpoints, redelivery, limits, chaining/branching,
owner/idempotency/admission, archive provenance and browser lifecycle.
A tiny local GROMACS 2026.1 CPU fixture exercises real checkpoints, cumulative
append, interrupted-child recovery and completed-input redelivery without
changing the source. It tests native contracts, not protein MD quality or
numerical equivalence. Deployed protein analysis and Modal Volume/coordinator
redelivery still require an explicitly authorized deployed smoke test.

## Primary sources

- [Biotite trajectory iterator](https://www.biotite-python.org/latest/apidoc/biotite.structure.io.TrajectoryFile.html#biotite.structure.io.TrajectoryFile.read_iter)
  and [pinned Biotite 1.6.0 source](https://github.com/biotite-dev/biotite/tree/v1.6.0/src/biotite/structure).
- [2026.1 continuation](https://manual.gromacs.org/documentation/2026.1/user-guide/managing-simulations.html)
  and [convert-tpr](https://manual.gromacs.org/documentation/2026.1/onlinehelp/gmx-convert-tpr.html).
- [Checkpoint step accounting](https://github.com/gromacs/gromacs/blob/v2026.1/src/gromacs/fileio/checkpoint.cpp#L2875-L2904)
  and [append handling](https://github.com/gromacs/gromacs/blob/v2026.1/src/gromacs/mdrunutility/handlerestart.cpp#L271-L406).
- [Parameter extraction](https://github.com/gromacs/gromacs/blob/v2026.1/src/gromacs/tools/dump.cpp#L104-L145),
  [checkpoint inventory](https://github.com/gromacs/gromacs/blob/v2026.1/src/gromacs/fileio/checkpoint.cpp#L1115-L1201),
  [checksum window](https://github.com/gromacs/gromacs/blob/v2026.1/src/gromacs/fileio/gmxfio.cpp#L370-L449),
  and [dump corruption warnings](https://github.com/gromacs/gromacs/blob/v2026.1/src/gromacs/fileio/checkpoint.cpp#L3052-L3137).

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
history. The existing analysis implementation retains its 64 GiB container
memory limit; this feature does not claim constant-memory processing or add
a hidden lifetime-duration cap. Service admission requires a Completed source;
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

Completed status alone is insufficient. Retained request, final and production
publications, TPR, raw XTC, energy, checkpoint and log must be available.
Checkpoint reads are bounded to 64 MiB. Source and target must share a Modal
environment and the pinned GROMACS scientific version.

Historical execution-plan version 2 publications keep their original inventory.
They did not bind checkpoint/log, so explicit continuation validates retained
native state without changing the historical publication. Version 3 production
publications additionally bind checkpoint and log.

The child request binds source Execution Run ID, directory, native stem,
cumulative endpoint, canonical request digest, final-publication digest and
checkpoint SHA-256. Read-only form metadata checks retained evidence without
launching a container. A normal CPU provider Task then:

1. Validates source publications and hashes for every reused published file.
2. Reads the TPR parameters and checkpoint through the pinned native binary.
   Checkpoint step/time must equal the completed TPR endpoint and saved time.
3. Checks native append filenames, offsets, sizes and checksums. GROMACS uses
   MD5 over up to the last 1 MiB ending at the saved offset; this is native
   compatibility checking, not the application's SHA-256 content identity.
4. Copies native state, original PDB/MDP, inherited NVT/NPT analyses and source
   TPR; extends the TPR once; publishes immutable preparation evidence.

Checkpoint dumps are consumed completely: GROMACS can print a corruption
warning without a failing exit code. Large dumps use container-local temporary
directories, removed after validation even on failure; they are not retained in
the output Volume, Python lists or streamed provider logs.
Missing/corrupt/incompatible state fails explicitly, with no fallback to a fresh
simulation.

Initial native files are copied to temporary siblings and atomically renamed,
so an interrupted checkpoint copy cannot be mistaken for child MD progress.
Preparation redelivery reuses its content-bound publication. It never copies
the source over child progress. Production redelivery keeps the fixed target
and current child checkpoint. Explicit append mode rejects a missing checkpoint.
Native endpoint validation precedes completed production publication.

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

Continue production opens a dedicated form: source link/endpoint, optional
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

Use `biomodals app run gromacs -- --continue-from <execution-run-id>
--additional-time-ns 10 --run-name continued-example`.
`--cpu-only` selects CPU; `--no-cpu-only` selects GPU; omission inherits.
No PDB is needed. Continuation is mutually exclusive with `--input-pdb` and
the outer `--restart-from` compatible-recovery flag. Physical settings remain
inherited. Normal global container limits apply to the new root.
Use `biomodals run restart` for same-plan recovery of a continuation.

Deploy the updated GROMACS app, pin its version in the API, restart the API and
deploy the matching frontend. Version-three requests cannot use an old
coordinator: API preflight resolves the new preparation function without
starting a container. Existing Jobs retain their deployment snapshots; do not
delete deployments they still need.

Offline verification covers package discovery/ShortMD imports, source integrity,
append checksums/endpoints, redelivery, limits, chaining/branching,
owner/idempotency/admission, archive provenance and browser lifecycle.
A tiny local GROMACS 2026.1 CPU fixture exercises real checkpoints, cumulative
append, interrupted-child recovery and completed-input redelivery without
changing the source. It tests native contracts, not protein MD quality or
numerical equivalence. Full protein analysis and Modal Volume/coordinator
redelivery still require an explicitly authorized deployed smoke test.

## Primary sources

- [2026.1 continuation](https://manual.gromacs.org/documentation/2026.1/user-guide/managing-simulations.html)
  and [convert-tpr](https://manual.gromacs.org/documentation/2026.1/onlinehelp/gmx-convert-tpr.html).
- [Checkpoint step accounting](https://github.com/gromacs/gromacs/blob/v2026.1/src/gromacs/fileio/checkpoint.cpp#L2875-L2904)
  and [append handling](https://github.com/gromacs/gromacs/blob/v2026.1/src/gromacs/mdrunutility/handlerestart.cpp#L271-L406).
- [Parameter extraction](https://github.com/gromacs/gromacs/blob/v2026.1/src/gromacs/tools/dump.cpp#L104-L145),
  [checkpoint inventory](https://github.com/gromacs/gromacs/blob/v2026.1/src/gromacs/fileio/checkpoint.cpp#L1115-L1201),
  [checksum window](https://github.com/gromacs/gromacs/blob/v2026.1/src/gromacs/fileio/gmxfio.cpp#L370-L449),
  and [dump corruption warnings](https://github.com/gromacs/gromacs/blob/v2026.1/src/gromacs/fileio/checkpoint.cpp#L3052-L3137).

# Job deletion

Status: Implemented on `feat/job-deletion`; offline verification and coordinated frontend handoff precede user review. No live deletion or deployment is authorized by this work.

## Product contract

Deletion is an owner-only action for terminal Jobs: `succeeded`, `partial`, `failed`, and `cancelled`. Queued, running, finalizing, cancellation-requested, blocked, and state-unknown Jobs must finish or be resolved/cancelled first. Deletion is separate from cancellation and never makes a remote execution disappear from admission accounting.

The Job immediately becomes unavailable through normal owner and administrator endpoints. Its SQLite row, identity, outcome, submission key, and operational provenance remain internally available. There is no Trash, Restore, bulk deletion, metadata-scrubbing feature, or administrator deleted-Job browser. Deletion is not complete data erasure: remote Modal outputs, execution records, logs, scientific caches, historical billing, downloaded copies, and already-loaded browser data remain untouched.

A red Delete button appears beside Refresh on eligible Job detail pages only. A confirmation explains permanent website removal, eventual local cleanup, remote retention, and preservation of existing child Jobs. After acceptance, the frontend clears affected queries and navigates to My Jobs. Other already-open tabs need not update instantly; subsequent requests and reloads see unavailability. No additional browser synchronization service is needed.

Existing GROMACS extensions and clustering Jobs survive source deletion. Deleted sources cannot start new child Jobs, but exact retry of an already-admitted child still recovers that same child. Source identifiers remain provenance; following an old source link shows the ordinary unavailable page.

## HTTP and submission identity

[Shared Job routes](../../src/biomodals/service/jobs_api.py) expose authenticated, owner-scoped, CSRF-protected `DELETE /api/v1/jobs/{job_id}`:

- `202` with an empty body acknowledges the durable deletion intent, including a repeated DELETE by the same owner. It does not promise that local bytes are already gone.
- `404` covers missing/foreign Jobs, including foreign tombstones.
- `409 job_not_deletable` rejects nonterminal Jobs without changing visibility or execution.
- Authentication, Origin, and CSRF errors follow the shared HTTP contract.

Normal Job detail/list, logs, inputs, previews, results, refresh, cancellation, retry, and download routes no longer expose deleted Jobs. Existing in-flight readers may finish; deletion does not recall bytes already delivered. New archive leases and publication attempts check the durable tombstone, including requests that began before deletion.

[Store replay lookup](../../src/biomodals/service/store.py) retains consumed idempotency keys. Reusing a deleted Job's submission returns `409 job_deleted`, never a fresh paid Job or a resurrected view. Frontend recovery displays that rejection without automatically rotating a key or resubmitting. A user can deliberately make a new submission with a new intent.

Continuation and clustering submission routes may inspect the owner's source tombstone solely to reproduce the original default display name for digest comparison. Exact admitted-child replay occurs before source availability checks; new admission transactionally rejects a deleted source after any slow inspection. New continuations and clustering Jobs both record `source_job_id`. Historical continuations without that column populated are not backfilled or cascaded.

## Persistence and access boundaries

Schema 10 adds `deleted_at`, `cleanup_completed_at`, and `cleanup_retry_at` to Jobs. Startup supports fresh schema 10 and upgrades schemas 8 and 9, chaining the earlier operation/source migration when necessary. No new table or scientific execution state is introduced.

Owner queries hide tombstones. Internal `get_job_by_id` and AlphaFold3 shared-publication predecessor lookup retain them; callers must not use those methods to bypass public visibility. Administrator log authorization explicitly rejects tombstones. Lifecycle writes recheck deletion inside their SQLite transaction so stale reconciliation, restoration, or result-preparation retry cannot revive the Job. History cursors can still use an owner-scoped deleted row as an ordering anchor.

Local cached-result accounting excludes deleted Jobs, while filesystem usage continues to count bytes still present. Billing aggregates Modal charges independently and is unchanged. Startup pending-input retention covers queued Jobs across all Tools, not only GROMACS.

## Local ownership and cleanup

Paths use independently configured `BIOMODALS_STATE_DIR` and `BIOMODALS_CACHE_DIR`; never assume they share one parent or recursively remove the work directory.

| Resource | Removal scope |
| --- | --- |
| Result archive | `CACHE_DIR/results/<job UUID>.result` |
| Archive staging | `CACHE_DIR/results/.<job UUID>.*.part` |
| Archive assembly | `CACHE_DIR/results/.<job UUID>.archive-*` |
| Pending request | `STATE_DIR/pending-inputs/<job UUID>/` |
| Claimed AlphaFold3 input | The validation UUID referenced by this Job, under `STATE_DIR/validated-inputs/` |
| SQLite, other Jobs, therapeutic reference cache, unclaimed validations, remote artifacts | Not deleted by this operation |

GROMACS clustering now uses the Job-scoped archive-assembly prefix. Unattributable historical `clustering-download-*` directories are not guessed at or swept during owner deletion. Preview readers do not create a separate persistent preview tree; in-memory decoded data is not a reason to restore website access.

The existing [Job reconciler](../../src/biomodals/service/tool_runtime.py) processes bounded cleanup work. It reuses the per-Job lifecycle lock and Tool adapter's local pending-input discard hook. Restoration holds that same lock; cleanup defers if a producer is active. The [archive cache](../../src/biomodals/service/artifacts.py) checks durable deletion at lease/publication boundaries. Once producers are quiescent, cleanup removes exact Job-owned files without following scratch-directory symlinks into other data.

Active archive leases finish before removal. Prepared-download grace does not permit a new download after deletion. Busy or failed cleanup is retried after at least 60 seconds, subject to the reconciler interval, and survives API restart. Filesystem failures are logged with the Job ID and leave cleanup incomplete. Missing files are already clean; permission/storage errors are not silently treated as success. No new cleanup service or polling endpoint is introduced.

The deployment contract remains one API worker. This change does not claim multi-process coordination through process-local locks.

## Verification and rollout

Offline coverage must exercise owner/admin/foreign access and CSRF, all terminal/nonterminal states, repeated deletion, deleted submission replay, supported migrations, deleted history cursors, stale writes, claimed/pending inputs, retained internal predecessor evidence, leased downloads, restoration versus deletion, cleanup failure/restart, preservation of unrelated files, and exact child recovery after source deletion. Frontend checks cover confirmation, lost DELETE response with explicit retry, stale queries, unavailable historical links, state changes, reauthentication, and unchanged child results.

Relevant tests: [store](../../tests/service/test_job_store.py), [deletion API and cleanup](../../tests/service/test_job_deletion.py), [lifecycle](../../tests/service/test_tool_runtime.py), [continuation](../../tests/service/test_gromacs_continuation_api.py), and [clustering](../../tests/service/test_gromacs_clustering_api.py). Browser integration uses the offline fake-provider fixture, not real scientific submissions.

Rollout requires a stopped-service SQLite backup, updated API/schema migration, and matching frontend. No Modal app or workflow redeployment is required. Older API builds cannot read schema 10; rollback requires the corresponding database backup. See the [deployment runbook](../deployment/mvp-runbook.md#pre-release-service-schema).

Backend verification on 2026-09-24: the complete offline suite passed 2,131 tests with 7 skipped. Service-wide type checks and required pre-commit hooks passed. The deletion tests explicitly exercise a held download, paused restoration, failed cleanup followed by a fresh lifecycle/reconciler, exact source-child replay, retained AlphaFold3 predecessor identity, and removal of only the target Job's claimed validation. No live Job was deleted or submitted.

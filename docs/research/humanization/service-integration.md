# Humanization service integration

Research snapshot: 2026-09-07. Baselines: local humanization stack versus fetched upstream main `b5c2421734d5238f27393287ff024e45b99745a6`. **Use current main's service seams.** Inspection used `git show`; no checkout, deployment, jobs, or production changes occurred. Current-main links below pin that commit; local links deliberately describe the older stack.

This is the pre-implementation research snapshot. The stack is now merged;
accepted decisions and implementation status are recorded in the
[service specification](../../specs/humanization-service.md). Relative code
links now resolve to the integrated checkout, not the historical stack.

## Facts: current main

- Production registers **GROMACS and AlphaFold3**. `ToolRegistration` contains `ToolDefinition` and `ToolAdapter`; typed routers are supplied separately. The narrow adapter protocol is `stage(job)`, `discard_pending(job)`, and `prepare_result(job, cache, completed_at)`. Shared `JobLifecycle` manages service launch/observation/cancellation/finalization without scheduling scientific tasks. [Assembly][api], [adapter/lifecycle][runtime].
- `RemoteExecutionClient` already supplies the remote-authority bridge. Job UUID equals Execution Run UUID; environment/app/version are pinned. It resolves the exact coordinator, launches, polls root calls, reads status, cancels, resumes, pages provider diagnostics and streams SDK logs. Returned overviews must match UUID/deployment. Preflight requires `run`, `resume`, `status`, `cancel`, `provider_calls`, and `provider_call`. [Client][remote], [accepted ADR][adr].
- Submission requires cookie authentication, exact Origin/CSRF on mutations, and UUID `Idempotency-Key`; ownership, idempotency and active limits are shared. GROMACS validates multipart input, persists an app request in `PendingRequestStore`, admits a Job and wakes the reconciler. AF3 first retains an owner-scoped validated document, then admits a validation ID. Both return after durable admission, before background staging/launch. A bounded humanization JSON request need not copy AF3's large-document validation lifecycle. [GROMACS router][grouter], [AF3 router][arouter], [store][store].
- A lifecycle pass stages immutable request/launch identity, discards verified pending input, records submission-in-progress, and launches. Inconclusive spawn becomes `state_unknown`; no-root-handle unknown launches are not automatically replaced. AF3's adapter can return `SubmissionWait` while an earlier Job may own shared publications. [Lifecycle][runtime], [AF3 adapter][aadapter].
- Service SQLite stores service data and bounded state projections, **not kernel tables**. `ToolStageDefinition` groups node keys into labeled semantic stages with task counts. Shared detail reads may refresh stale state; `POST /jobs/{id}/refresh` explicitly refreshes and can resume suspended/unknown remote execution after conclusive root observation. Background reconciliation defaults to 60 seconds, concurrency four, and polls root calls before waking coordinator status. Remote scientific success becomes service `finalizing` until delivery preparation completes. [Projection][tools], [routes][jobs], [lifecycle][runtime].
- Results are **archives, not ZIP-only**. `PreparedResult` carries filename/media type/size/SHA-256/schema. GROMACS packages a validated scientific publication as ZIP; AF3 reuses its app archive builder and returns `application/zstd`. Shared prepare-download/download routes support recorded media type and ranges. Exact restoration checks size/hash/schema and blocks on mismatch without compute replay. There is no generic candidate-table or individual-artifact preview endpoint. [Delivery][jobs], [restoration][runtime], [GROMACS adapter][gadapter], [AF3 adapter][aadapter].
- Logs are shared remote capabilities with bounded provider diagnostics. Tool definitions map stages and default owner log visibility off; GROMACS opts in. Humanization does not need a bespoke log opener. Router validation must enforce its own sequence/batch bounds: global middleware currently allows up to 256 MiB. [Client][remote], [tool metadata][tools], [assembly][api].

## Baseline differences and concrete conflicts

| Concern | Local humanization stack | Current main |
| --- | --- | --- |
| Registration | `WorkloadRegistration`, only GROMACS | `ToolRegistration`, GROMACS + AF3 |
| Authority | API-hosted GROMACS graph and local kernel ledger | Deployed coordinator and remote ledger; service projection |
| Adapter | Provider calls plus archive lifecycle | Stage/discard pending/prepare Result |
| Identity | Separate Job and Run UUIDs | Job UUID is Run UUID |
| Logs | Local operation rows and workload opener | Shared remote diagnostics and SDK logs |
| Downloads | ZIP assumptions | Recorded MIME; ZIP and zstd examples |

Local evidence: [assembly](../../../src/biomodals/service/api.py), [registration](../../../src/biomodals/service/jobs.py), [GROMACS host](../../../src/biomodals/app/bioinfo/gromacs_execution.py), [store](../../../src/biomodals/service/store.py). The apparent missing remote bridge and ZIP-only delivery are **not current-main gaps**.

Current main marks the old [architecture note][historical] partially superseded by [ADR 0007][adr] and the [API Tool service specification][spec]. Its service-local DAG passages are historical and must not guide this implementation. The local copy lacks that banner. Local prose also excludes all unknown states from reconciliation whereas local code admits unknown cancellation outcomes; current main has a different root-evidence-based lifecycle.

**Remaining real workflow protocol gap:** the generic workflow coordinator exposes `prepare_run`/`drive_prepared`, while the service preflight/launch expects staged `run`. The workflow driving result is `AppRunResult`, whereas `RemoteExecutionClient.poll_root` requires `ExecutionOverview`. Consequently the existing service lifecycle is reusable but generic workflow hosting is not a drop-in match. Provide a narrow workflow-owned request/coordinator adaptation meeting the service contract; do not fork Job lifecycle or introduce a service-local scheduler. [Remote launch/result verification][remote], [generic workflow coordinator][coordinator].

One current-main prose/code discrepancy: the spec says “Root-call failure is terminal,” while `poll_root` catches `modal.exception.RemoteError` and reads the coordinator ledger because the Run may be durably suspended. The ADR and lifecycle preserve recoverable suspension. Follow the remote ledger and tighten that sentence if still present. [Spec][spec], [client][remote], [ADR][adr].

## Recommendation

Build on current main: a humanization `ToolDefinition`, one typed router and one `ToolAdapter`, registered beside GROMACS/AF3. Reuse shared auth/admission, `PendingRequestStore` for bounded requests when appropriate, request/launch staging, `RemoteExecutionClient`, `JobLifecycle`, semantic progress, refresh/cancel/log/download routes and exact restoration. Reuse scientific publication/archive logic as AF3 does. Adapt the humanization workflow's coordinator contract at its scientific owner to provide staged `run` and an `ExecutionOverview` root result.

## Open decisions for the interview

- Website input convenience and admission bounds: pasted complete VH-VL pair, CSV batches, or both. Complete pairs and explicit pairing are already settled scientific requirements; single-chain input is out of scope.
- Browser candidate summaries/previews versus archive download alone; typed owner-authorized result retrieval is additional work.
- Map the existing scientific success/partial-result policy into service delivery; choose deterministic archive membership and recovery without compute replay, without redefining scientific outcomes.
- Pinned humanization deployment, node-to-stage grouping, admission defaults and owner log visibility.
- Bringing the humanization stack onto current-main service/kernel contracts before implementation; this research did not migrate branches.

[api]: https://github.com/y1zhou/biomodals/blob/b5c2421734d5238f27393287ff024e45b99745a6/src/biomodals/service/api.py
[runtime]: https://github.com/y1zhou/biomodals/blob/b5c2421734d5238f27393287ff024e45b99745a6/src/biomodals/service/tool_runtime.py
[remote]: https://github.com/y1zhou/biomodals/blob/b5c2421734d5238f27393287ff024e45b99745a6/src/biomodals/service/remote_execution.py
[store]: https://github.com/y1zhou/biomodals/blob/b5c2421734d5238f27393287ff024e45b99745a6/src/biomodals/service/store.py
[grouter]: https://github.com/y1zhou/biomodals/blob/b5c2421734d5238f27393287ff024e45b99745a6/src/biomodals/service/gromacs/router.py
[arouter]: https://github.com/y1zhou/biomodals/blob/b5c2421734d5238f27393287ff024e45b99745a6/src/biomodals/service/alphafold3/router.py
[gadapter]: https://github.com/y1zhou/biomodals/blob/b5c2421734d5238f27393287ff024e45b99745a6/src/biomodals/service/gromacs/modal.py
[aadapter]: https://github.com/y1zhou/biomodals/blob/b5c2421734d5238f27393287ff024e45b99745a6/src/biomodals/service/alphafold3/modal.py
[tools]: https://github.com/y1zhou/biomodals/blob/b5c2421734d5238f27393287ff024e45b99745a6/src/biomodals/service/tools.py
[jobs]: https://github.com/y1zhou/biomodals/blob/b5c2421734d5238f27393287ff024e45b99745a6/src/biomodals/service/jobs_api.py
[adr]: https://github.com/y1zhou/biomodals/blob/b5c2421734d5238f27393287ff024e45b99745a6/docs/adr/0007-api-jobs-use-remote-coordinators.md
[spec]: https://github.com/y1zhou/biomodals/blob/b5c2421734d5238f27393287ff024e45b99745a6/docs/specs/api-tool-service.md
[historical]: https://github.com/y1zhou/biomodals/blob/b5c2421734d5238f27393287ff024e45b99745a6/docs/research/api-service-architecture.md
[coordinator]: https://github.com/y1zhou/biomodals/blob/b5c2421734d5238f27393287ff024e45b99745a6/src/biomodals/execution/modal/orchestrator.py

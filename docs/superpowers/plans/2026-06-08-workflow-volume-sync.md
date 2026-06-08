# Workflow Volume Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate intermittent workflow `FileNotFoundError` failures caused by workflow volume writes, commits, reloads, and reads becoming visible in the wrong order across Modal containers.

**Architecture:** Keep Modal volume ownership at the runtime/orchestrator boundary instead of importing Modal into ledger or artifact helpers. Add tests that model Modal visibility rules, then serialize workflow-volume `commit()`/`reload()` with local workflow-volume file access and ensure artifact materialization plus ledger updates are committed as one visible runtime boundary. Preserve remote-node parallelism where possible, but do not allow `reload()` to run while an orchestrator-local node is reading from or writing to the mounted workflow volume.

**Tech Stack:** Python 3.13 workflow runtime, Modal Volume v2, SQLite workflow ledger, Pydantic app result schemas, `orjson`, `pytest`, `prek`.

---

## Investigation Summary

Modal documentation says writes made in one container are not reliably visible to another container until the writer commits the volume, and a reader container must reload to observe changes made after its mount view was created. Modal also warns that during `reload()` the volume can appear empty to the initiating container, and concurrent writes to the same file are last-write-wins.

The core audit found these write and sync sites:

- `src/biomodals/workflow/core/runtime.py` owns the Modal boundary through `WorkflowVolume.commit()` and `WorkflowVolume.reload()` and calls `_commit_volume()` or `_reload_volume()` around most runtime transitions.
- `src/biomodals/workflow/core/artifacts.py` writes inline outputs and log files under attempt directories and writes artifact manifests, but it is intentionally Modal-agnostic and relies on `WorkflowRuntime` to commit after materialization.
- `src/biomodals/workflow/core/ledger.py` creates, deletes, and mutates run/node state on disk and in SQLite, but it is intentionally Modal-agnostic and relies on `WorkflowRuntime` to commit after those mutations.
- `WorkflowRuntime._run_ready_nodes()` currently uses a `ThreadPoolExecutor`, so ready nodes can run in parallel while other worker threads call `_reload_volume()` and `_commit_volume()`.

The highest-risk gap is not a simple missing `commit()` after artifact writes: the current runtime already commits after node finalization. The credible intermittent failure is that `reload()` or `commit()` can happen from one scheduler worker while another orchestrator-local node is using the same mounted workflow volume. Because Modal reload can make the volume temporarily unavailable to that container, this can surface as occasional file-not-found failures.

## Task 1: Add Volume Visibility Regression Tests

**Files:**

- Modify: `tests/workflow/test_runtime.py`

- [ ] **Step 1: Add a fake volume that records committed snapshots**

Add this helper near the existing `FakeVolume` in `tests/workflow/test_runtime.py`:

```python
class SnapshotVolume(FakeVolume):
    def __init__(self, root: Path) -> None:
        super().__init__()
        self.root = root
        self.committed_paths: set[str] = set()

    def commit(self) -> None:
        super().commit()
        self.committed_paths = {
            path.relative_to(self.root).as_posix()
            for path in self.root.rglob("*")
            if path.is_file()
        }
```

- [ ] **Step 2: Add a downstream node test that only trusts committed artifacts**

Add this test in `tests/workflow/test_runtime.py`:

```python
def test_downstream_node_sees_committed_inline_artifact(tmp_path: Path) -> None:
    workflow = Workflow("demo")
    volume = SnapshotVolume(tmp_path)
    upstream = workflow.add_node(
        FakeNode(
            result=AppRunResult(
                status=AppRunStatus.SUCCEEDED,
                outputs=[
                    AppOutput(
                        name="report",
                        kind=ArtifactKind.REPORT,
                        storage=InlineBytes(data=b"ok\n", filename="report.txt"),
                    )
                ],
            )
        ),
        id="upstream",
    )

    class CommittedReaderNode(WorkflowNativeNode):
        def run(self, context):
            artifacts = context.inputs["report"]
            storage_path = artifacts[0].storage.path
            assert storage_path in volume.committed_paths
            return AppRunResult(status=AppRunStatus.SUCCEEDED)

    workflow.add_node(
        CommittedReaderNode(),
        id="downstream",
        inputs={"report": upstream.outputs(kind=ArtifactKind.REPORT)},
    )

    runtime = WorkflowRuntime(
        workflow=workflow,
        volume_root=tmp_path,
        workflow_volume_name="Workflow-outputs",
        workflow_volume=volume,
    )

    result = runtime.run(run_id="run-1")

    assert result.status == AppRunStatus.SUCCEEDED
```

- [ ] **Step 3: Add a regression test for logs from failed nodes**

Add this test in `tests/workflow/test_runtime.py`:

```python
def test_failed_node_logs_are_committed_before_run_returns(tmp_path: Path) -> None:
    workflow = Workflow("demo")
    volume = SnapshotVolume(tmp_path)
    workflow.add_node(
        FakeNode(
            result=AppRunResult(
                status=AppRunStatus.FAILED,
                logs=[
                    AppOutput(
                        name="stderr",
                        kind=ArtifactKind.LOGS,
                        storage=InlineBytes(data=b"missing input\n", filename="stderr.log"),
                    )
                ],
                warnings=["remote file not found"],
            )
        ),
        id="failed",
    )

    runtime = WorkflowRuntime(
        workflow=workflow,
        volume_root=tmp_path,
        workflow_volume_name="Workflow-outputs",
        workflow_volume=volume,
    )

    result = runtime.run(run_id="run-1")

    assert result.status == AppRunStatus.FAILED
    assert (
        "demo/run-1/nodes/failed/attempts/attempt-1/logs/failed-logs-stderr/stderr.log"
        in volume.committed_paths
    )
    assert "demo/run-1/artifacts/failed-logs-stderr.json" in volume.committed_paths
```

- [ ] **Step 4: Add a reload-during-local-node regression test**

Add this test in `tests/workflow/test_runtime.py`. It must fail before the runtime synchronization change because the remote worker can call `reload()` while the orchestrator-local node is still inside `run()`.

```python
def test_remote_reload_waits_for_orchestrator_node_volume_access(
    tmp_path: Path,
) -> None:
    workflow = Workflow("demo")
    local_running = Event()
    local_can_finish = Event()
    local_finished = Event()
    reload_happened_while_local_running = False

    class GuardedVolume(FakeVolume):
        def reload(self) -> None:
            nonlocal reload_happened_while_local_running
            if local_running.is_set() and not local_finished.is_set():
                reload_happened_while_local_running = True
                local_can_finish.set()
            super().reload()

    class SlowLocalNode(WorkflowNativeNode):
        def run(self, context):
            local_running.set()
            local_can_finish.wait(timeout=1)
            local_finished.set()
            return AppRunResult(status=AppRunStatus.SUCCEEDED)

    class FinishingRemoteCall(FakeRemoteCall):
        def get(self, timeout=None):
            local_running.wait(timeout=1)
            return super().get(timeout=timeout)

    volume = GuardedVolume()
    workflow.add_node(SlowLocalNode(), id="local")
    workflow.add_node(
        DirectSubmitNode(
            call=FinishingRemoteCall(
                object_id="fc-remote",
                result=AppRunResult(status=AppRunStatus.SUCCEEDED),
            )
        ),
        id="remote",
    )

    runtime = WorkflowRuntime(
        workflow=workflow,
        volume_root=tmp_path,
        workflow_volume_name="Workflow-outputs",
        workflow_volume=volume,
        max_ready_workers=2,
    )

    result = runtime.run(run_id="run-1")

    assert result.status == AppRunStatus.SUCCEEDED
    assert reload_happened_while_local_running is False
```

- [ ] **Step 5: Run focused tests and confirm the new reload race test fails**

Run:

```bash
rtk uv run pytest tests/workflow/test_runtime.py -q
```

Expected before implementation: the new reload race test fails because `reload_happened_while_local_running` becomes `True`. The committed snapshot tests may already pass; keep them because they lock in the desired commit boundary.

## Task 2: Serialize Volume Sync With Local Volume Access

**Files:**

- Modify: `src/biomodals/workflow/core/runtime.py`

- [ ] **Step 1: Add a runtime lock for mounted workflow-volume access**

In `WorkflowRuntime.__init__`, after `_active_remote_calls_lock`, add:

```python
self._workflow_volume_access_lock = RLock()
```

- [ ] **Step 2: Guard orchestrator-local node execution**

Replace `_dispatch_node()` with:

```python
def _dispatch_node(
    self, node: WorkflowNode, context: NodeRunContext
) -> AppRunResult:
    if node.placement == NodePlacement.REMOTE:
        return self._run_remote_node(node, context)
    with self._workflow_volume_access_lock:
        return node.run(context)
```

This serializes `ORCHESTRATOR` nodes that may read `context.inputs`, read volume-backed artifacts, or write `context.cache_dir`. Remote node waiting can still run concurrently.

- [ ] **Step 3: Guard Modal commit/reload**

Update `_commit_volume()` and `_reload_volume()`:

```python
def _commit_volume(self) -> None:
    if self.workflow_volume is not None:
        with self._workflow_volume_access_lock:
            with self.ledger.closed_for_volume_sync():
                self.workflow_volume.commit()

def _reload_volume(self) -> None:
    if self.workflow_volume is not None:
        with self._workflow_volume_access_lock:
            with self.ledger.closed_for_volume_sync():
                self.workflow_volume.reload()
```

The `RLock` is intentional because finalization will hold this lock while calling `_commit_volume()`.

- [ ] **Step 4: Guard materialization plus ledger finalization as one visible boundary**

Wrap the body of `_finalize_node_result()` in the same lock:

```python
def _finalize_node_result(
    self,
    *,
    node_id: str,
    attempt_id: str,
    attempt_dir: Path,
    result: AppRunResult,
) -> AppRunResult:
    with self._workflow_volume_access_lock:
        materialized = materialize_app_run_result(
            result=result,
            workflow_volume_name=self.workflow_volume_name,
            attempt_dir=attempt_dir,
            artifact_dir=self.ledger.run_root / "artifacts",
            producing_node_id=node_id,
            volume_root=self.volume_root,
        )
        persisted_result = materialized.result
        if result.status in {AppRunStatus.FAILED, AppRunStatus.PARTIAL}:
            print(
                f"[workflow] Node failed: {node_id} attempt={attempt_id}: "
                f"{self._node_error_message(result)}",
                flush=True,
            )
            self.ledger.record_artifacts(materialized.artifacts)
            self.ledger.record_attempt_completed(
                node_id,
                attempt_id,
                NodeStatus.FAILED,
                result=persisted_result,
                error=self._node_error_message(result),
            )
            self._commit_volume()
            return result

        artifacts = materialized.artifacts
        self.ledger.record_artifacts(artifacts)
        self.ledger.mark_node_succeeded(
            node_id,
            [artifact.artifact_id for artifact in artifacts],
        )
        self.ledger.record_attempt_completed(
            node_id,
            attempt_id,
            NodeStatus.SUCCEEDED,
            result=persisted_result,
        )
        self._commit_volume()
    print(
        f"[workflow] Node succeeded: {node_id} attempt={attempt_id} "
        f"artifacts={len(artifacts)}",
        flush=True,
    )
    return result
```

Keep the success `print()` outside the lock so the lock only covers filesystem and ledger visibility.

- [ ] **Step 5: Run focused runtime tests**

Run:

```bash
rtk uv run pytest tests/workflow/test_runtime.py -q
```

Expected after implementation: all runtime tests pass, including the new reload race regression test.

## Task 3: Make Deletions And Run Layout Mutations Explicitly Committed

**Files:**

- Modify: `src/biomodals/workflow/core/runtime.py`
- Modify: `tests/workflow/test_runtime.py`
- [ ] **Step 1: Add a regression test for forced run reset visibility**

Add this test in `tests/workflow/test_runtime.py`:

```python
def test_force_reset_commits_deleted_run_before_recreate(tmp_path: Path) -> None:
    workflow = Workflow("demo")
    volume = SnapshotVolume(tmp_path)
    workflow.add_node(FakeNode(), id="one")
    runtime = WorkflowRuntime(
        workflow=workflow,
        volume_root=tmp_path,
        workflow_volume_name="Workflow-outputs",
        workflow_volume=volume,
    )

    assert runtime.run(run_id="run-1").status == AppRunStatus.SUCCEEDED
    stale_path = tmp_path / "demo" / "run-1" / "nodes" / "stale.txt"
    stale_path.parent.mkdir(parents=True, exist_ok=True)
    stale_path.write_text("stale\n", encoding="utf-8")
    runtime._commit_volume()
    assert "demo/run-1/nodes/stale.txt" in volume.committed_paths

    assert runtime.run(run_id="run-1", force=True).status == AppRunStatus.SUCCEEDED
    assert "demo/run-1/nodes/stale.txt" not in volume.committed_paths
```

- [ ] **Step 2: Add a helper for reset-and-commit**

In `WorkflowRuntime`, add:

```python
def _reset_node_for_rerun(self, node_id: str) -> None:
    with self._workflow_volume_access_lock:
        self.ledger.reset_node(node_id)
        self._commit_volume()
```

Then replace:

```python
self.ledger.reset_node(node_id)
self._commit_volume()
```

with:

```python
self._reset_node_for_rerun(node_id)
```

- [ ] **Step 3: Keep run reset committed before recreate**

The existing `run()` flow already commits after `reset_run()` and after `create_run()`. Leave that ordering in place. If the test in Step 1 fails, wrap the `reset_run()` plus `_commit_volume()` block in `_workflow_volume_access_lock` the same way as node reset:

```python
with self._workflow_volume_access_lock:
    self.ledger.reset_run(definition.name, run_id)
    self._commit_volume()
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
rtk uv run pytest tests/workflow/test_runtime.py tests/workflow/test_ledger.py -q
```

Expected: all tests pass and forced reset removes stale committed files from the fake snapshot.

## Task 4: Keep Artifact And Ledger Helpers Modal-Agnostic

**Files:**

- Modify: `src/biomodals/workflow/core/artifacts.py`
- Modify: `src/biomodals/workflow/core/ledger.py`
- Modify: `src/biomodals/workflow/core/runtime.py`
- [ ] **Step 1: Do not add Modal imports to `artifacts.py` or `ledger.py`**

Confirm these commands return no output:

```bash
rtk rg -n "import modal|modal\\." src/biomodals/workflow/core/artifacts.py src/biomodals/workflow/core/ledger.py
```

- [ ] **Step 2: Add comments only at the caller boundary**

If needed, add this short comment before `_finalize_node_result()` or inside it, not inside `artifacts.py`:

```python
# Materialization writes files under the mounted workflow volume. Keep it
# serialized with reload/commit so other scheduler workers cannot observe a
# partially synchronized volume view.
```

- [ ] **Step 3: Leave app-owned volume commits out of core**

Do not try to commit app-owned output volumes from `core`. `core` only receives `VolumePath` metadata for app outputs and usually does not own the source `modal.Volume` handle. If file-not-found failures point at app-owned paths such as GROMACS or RFdiffusion outputs, fix those app or workflow-specific remote functions separately by calling the app volume `commit()` after writes and `reload()` before reads.

## Task 5: Full Verification

**Files:**

- Verify: `src/biomodals/workflow/core/runtime.py`
- Verify: `src/biomodals/workflow/core/artifacts.py`
- Verify: `src/biomodals/workflow/core/ledger.py`
- Verify: `tests/workflow/test_runtime.py`
- Verify: `tests/workflow/test_artifacts.py`
- [ ] **Step 1: Run focused workflow core tests**

Run:

```bash
rtk uv run pytest tests/workflow/test_runtime.py tests/workflow/test_artifacts.py tests/workflow/test_ledger.py tests/workflow/test_orchestrator.py -q
```

Expected: all selected workflow core tests pass.

- [ ] **Step 2: Run workflow-specific tests that exercise app volume handoffs**

Run:

```bash
rtk uv run pytest tests/workflow/test_shortmd_workflow.py tests/workflow/test_rfd_ligandmpnn_workflow.py -q
```

Expected: ShortMD clone tests still show explicit app-volume reload and commit, and RFdiffusion selector tests still show app-volume reload before reads.

- [ ] **Step 3: Run CLI smoke tests**

Run:

```bash
rtk uv run biomodals workflow list
rtk uv run biomodals workflow help shortmd
rtk uv run biomodals workflow help rfd-ligandmpnn
```

Expected: workflow discovery and help commands complete without import errors.

- [ ] **Step 4: Run pre-commit on changed files**

Run:

```bash
rtk prek run --files src/biomodals/workflow/core/runtime.py tests/workflow/test_runtime.py tests/workflow/test_artifacts.py docs/superpowers/plans/2026-06-08-workflow-volume-sync.md
```

Expected: all hooks pass.

## Rollback Notes

- If runtime throughput regresses too much, keep the `_workflow_volume_access_lock` around `_commit_volume()`, `_reload_volume()`, and `_finalize_node_result()`, but consider making orchestrator-local node serialization opt-in through a node property such as `uses_workflow_volume = True`. Do not start there; the current safer fix should serialize all `ORCHESTRATOR` node execution because those nodes receive volume-backed inputs and cache paths.
- If tests reveal the failure is only app-owned output volume visibility, leave the core runtime synchronization tests in place and fix the specific app/workflow remote function that writes the missing path. Core cannot commit a source app volume unless it owns the `modal.Volume` handle.
- If Modal volume reload behavior changes in future docs, keep the tests as the project contract: no scheduler worker should reload the workflow volume while another local node is accessing it.

## Self-Review

- Coverage: the plan covers every workflow core module that writes or synchronizes mounted data: `runtime.py`, `artifacts.py`, `ledger.py`, and `orchestrator.py`.
- Placeholder scan: no step contains an unspecified implementation placeholder; every step has a concrete test, code change, or command.
- Type consistency: helper names, fake volume names, and existing test helper classes match current `tests/workflow/test_runtime.py` vocabulary.

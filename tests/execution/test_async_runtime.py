"""Asynchronous host facade tests for the execution runtime."""

# ruff: noqa: D101, D102, D103, D107, S106

import asyncio
from dataclasses import dataclass

import pytest

from biomodals.execution import ProviderCallStatus, RunStatus, RunStatusReason
from biomodals.execution.modal import (
    ModalCallObservation,
    ModalCallObservationKind,
    ModalDeploymentUnavailableError,
)
from biomodals.execution.runtime import AsyncExecutionRuntime, ProviderCallSubmission
from biomodals.execution.scheduler import ProviderCallCandidate

from .provider_call_helpers import (
    GPU_BINDING,
    RUN_ID,
    create_repository,
    persist_fixed_policy,
)


@dataclass
class FakeResolvedFunction:
    name: str


class AsyncFakeModalDriver:
    def __init__(self) -> None:
        self.resolve_count = 0
        self.spawn_count = 0
        self.cancelled: list[str] = []
        self.observation = ModalCallObservation(ModalCallObservationKind.RUNNING)

    async def resolve(self, binding):
        self.resolve_count += 1
        return FakeResolvedFunction(binding.function_name)

    async def spawn(self, function, *, args, kwargs):
        self.spawn_count += 1
        return f"fc-{function.name}-{self.spawn_count}"

    async def observe(self, provider_call_handle_id):
        return self.observation

    async def cancel(self, provider_call_handle_id):
        self.cancelled.append(provider_call_handle_id)


def _candidate() -> ProviderCallCandidate:
    return ProviderCallCandidate(
        candidate_key="inference:0",
        node_key="inference",
        node_ordinal=0,
        task_keys=("seed-0",),
        task_ordinal=0,
        binding=GPU_BINDING,
        compatibility_key="af3",
        depth=0,
        unblocking_span=0,
        max_tasks_per_call=1,
    )


def _second_candidate() -> ProviderCallCandidate:
    return ProviderCallCandidate(
        candidate_key="inference:1",
        node_key="inference",
        node_ordinal=0,
        task_keys=("seed-1",),
        task_ordinal=1,
        binding=GPU_BINDING,
        compatibility_key="af3",
        depth=0,
        unblocking_span=0,
        max_tasks_per_call=1,
    )


def test_async_runtime_submits_one_admission_set() -> None:
    async def scenario() -> None:
        repository = create_repository(task_count=2)
        persist_fixed_policy(
            repository,
            ("seed-0", "seed-1"),
            binding=GPU_BINDING,
            compatibility_key="af3",
        )
        driver = AsyncFakeModalDriver()
        checkpoints: list[int] = []
        runtime = AsyncExecutionRuntime(
            repository,
            modal_driver=driver,
            checkpoint=lambda: checkpoints.append(driver.spawn_count),
        )

        calls = await runtime.submit_provider_calls(
            RUN_ID,
            (
                ProviderCallSubmission(_candidate(), "batch-0"),
                ProviderCallSubmission(_second_candidate(), "batch-1"),
            ),
            now=110,
        )

        assert all(call is not None for call in calls)
        assert driver.resolve_count == 1
        assert driver.spawn_count == 2
        assert checkpoints == [0, 2]

    asyncio.run(scenario())


def test_async_runtime_preserves_preclaim_and_result_envelope_boundaries() -> None:
    async def scenario() -> None:
        repository = create_repository(task_count=1)
        persist_fixed_policy(
            repository,
            ("seed-0",),
            binding=GPU_BINDING,
            compatibility_key="af3",
        )
        driver = AsyncFakeModalDriver()
        checkpoints: list[int] = []
        runtime = AsyncExecutionRuntime(
            repository,
            modal_driver=driver,
            checkpoint=lambda: checkpoints.append(driver.spawn_count),
        )

        first = await runtime.submit_fixed_batch(
            RUN_ID,
            _candidate(),
            submission_token="batch",
            kwargs={"seed": 0},
            now=110,
        )
        duplicate = await runtime.submit_fixed_batch(
            RUN_ID,
            _candidate(),
            submission_token="batch",
            kwargs={"seed": 0},
            now=111,
        )
        assert first is not None
        assert first.status == ProviderCallStatus.ATTACHED
        assert duplicate == first
        assert driver.spawn_count == 1
        assert checkpoints == [0, 1]

        driver.observation = ModalCallObservation(
            ModalCallObservationKind.SUCCEEDED,
            result={"path": "/outputs/seed-0"},
        )
        completed = await runtime.reconcile_provider_call(
            first.provider_call_id,
            encode_result=lambda result: {"tasks": {"seed-0": result}},
            now=120,
        )

        assert completed.status == ProviderCallStatus.SUCCEEDED
        assert completed.result_envelope == {
            "tasks": {"seed-0": {"path": "/outputs/seed-0"}}
        }

    asyncio.run(scenario())


def test_async_running_poll_commits_only_the_first_state_transition() -> None:
    async def scenario() -> None:
        repository = create_repository(task_count=1)
        persist_fixed_policy(
            repository,
            ("seed-0",),
            binding=GPU_BINDING,
            compatibility_key="af3",
        )
        driver = AsyncFakeModalDriver()
        local_commits: list[ProviderCallStatus] = []
        runtime = AsyncExecutionRuntime(
            repository,
            modal_driver=driver,
            checkpoint=lambda: None,
            commit_local=lambda: local_commits.append(
                repository.list_provider_calls(RUN_ID)[0].status
            ),
        )
        call = await runtime.submit_fixed_batch(
            RUN_ID,
            _candidate(),
            submission_token="batch",
            now=110,
        )
        assert call is not None

        first = await runtime.reconcile_provider_call(
            call.provider_call_id,
            encode_result=lambda result: result,
            now=120,
        )
        second = await runtime.reconcile_provider_call(
            call.provider_call_id,
            encode_result=lambda result: result,
            now=121,
        )

        assert first.status == ProviderCallStatus.RUNNING
        assert second == first
        assert local_commits == [ProviderCallStatus.RUNNING]

    asyncio.run(scenario())


def test_async_runtime_requests_cancellation_without_inventing_completion() -> None:
    async def scenario() -> None:
        repository = create_repository(task_count=1)
        persist_fixed_policy(
            repository,
            ("seed-0",),
            binding=GPU_BINDING,
            compatibility_key="af3",
        )
        driver = AsyncFakeModalDriver()
        runtime = AsyncExecutionRuntime(
            repository,
            modal_driver=driver,
            checkpoint=lambda: None,
        )
        call = await runtime.submit_fixed_batch(
            RUN_ID,
            _candidate(),
            submission_token="batch",
            now=110,
        )
        assert call is not None

        requested = await runtime.request_provider_call_cancellation(
            call.provider_call_id,
            now=120,
        )

        assert requested.status == ProviderCallStatus.ATTACHED
        assert driver.cancelled == [f"fc-{GPU_BINDING.function_name}-1"]

    asyncio.run(scenario())


def test_async_unattached_returned_call_is_cancelled_and_checkpointed_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        repository = create_repository(task_count=1)
        persist_fixed_policy(
            repository,
            ("seed-0",),
            binding=GPU_BINDING,
            compatibility_key="af3",
        )
        driver = AsyncFakeModalDriver()
        checkpoints: list[ProviderCallStatus] = []
        runtime = AsyncExecutionRuntime(
            repository,
            modal_driver=driver,
            checkpoint=lambda: checkpoints.append(
                repository.list_provider_calls(RUN_ID)[0].status
            ),
        )

        def fail_attachment(*args, **kwargs):
            raise RuntimeError("attachment failed")

        monkeypatch.setattr(repository, "attach_provider_call", fail_attachment)

        with pytest.raises(RuntimeError, match="attachment failed"):
            await runtime.submit_fixed_batch(
                RUN_ID,
                _candidate(),
                submission_token="batch",
                now=110,
            )

        call = repository.list_provider_calls(RUN_ID)[0]
        assert call.status == ProviderCallStatus.OUTCOME_UNKNOWN
        assert driver.cancelled == [f"fc-{GPU_BINDING.function_name}-1"]
        assert checkpoints == [
            ProviderCallStatus.SUBMITTING,
            ProviderCallStatus.OUTCOME_UNKNOWN,
        ]

    asyncio.run(scenario())


def test_async_runtime_fails_a_missing_exact_deployment_before_preclaim() -> None:
    """API-hosted coordination uses the same deployment-loss transition."""

    async def scenario() -> None:
        repository = create_repository(task_count=1)
        persist_fixed_policy(
            repository,
            ("seed-0",),
            binding=GPU_BINDING,
            compatibility_key="af3",
        )

        class UnavailableDriver(AsyncFakeModalDriver):
            async def resolve(self, binding):
                raise ModalDeploymentUnavailableError("version expired")

        runtime = AsyncExecutionRuntime(
            repository,
            modal_driver=UnavailableDriver(),
            checkpoint=lambda: None,
        )

        assert (
            await runtime.submit_fixed_batch(
                RUN_ID,
                _candidate(),
                submission_token="batch",
                now=110,
            )
            is None
        )
        run = repository.get_run(RUN_ID)
        assert run.status_reason == RunStatusReason.DEPLOYMENT_UNAVAILABLE
        assert repository.list_provider_calls(RUN_ID) == ()

    asyncio.run(scenario())


def test_async_unavailable_deployment_does_not_override_cancellation() -> None:
    async def scenario() -> None:
        repository = create_repository(task_count=1)
        persist_fixed_policy(
            repository,
            ("seed-0",),
            binding=GPU_BINDING,
            compatibility_key="af3",
        )

        class CancellingUnavailableDriver(AsyncFakeModalDriver):
            runtime: AsyncExecutionRuntime

            async def resolve(self, binding):
                await self.runtime.cancel_run(RUN_ID, now=109)
                raise ModalDeploymentUnavailableError("version expired")

        driver = CancellingUnavailableDriver()
        runtime = AsyncExecutionRuntime(
            repository,
            modal_driver=driver,
            checkpoint=lambda: None,
        )
        driver.runtime = runtime

        assert (
            await runtime.submit_fixed_batch(
                RUN_ID,
                _candidate(),
                submission_token="batch",
                now=110,
            )
            is None
        )
        run = repository.get_run(RUN_ID)
        assert run.status == RunStatus.CANCEL_REQUESTED
        assert run.cancellation_is_durable
        assert run.status_reason is None

    asyncio.run(scenario())

<!-- markdownlint-disable MD013 -->

# Execution kernel consolidation

Status: accepted on 2026-08-19; implementation pending.

This plan amends [ADR 0006](../adr/0006-unified-execution-kernel.md) and the
[unified scheduler specification](unified-task-scheduler.md). Where an older
statement conflicts with this plan, this plan is authoritative. The older text
remains useful history until the migration is complete.

## Objective

Biomodals will have one provider-neutral execution kernel for apps, workflows,
and API-owned app calls. Provider integrations live below that kernel. Modal is
the only provider integration implemented today.

The consolidation removes two parallel orchestration surfaces:

- `biomodals.helper.app_execution`;
- `biomodals.workflow.core`.

Their reusable behavior moves into `biomodals.execution`. Apps and workflows
retain scientific policy and thin deployment composition roots.

## Domain boundary

The execution kernel owns:

- immutable executable DAGs, Node dependencies, and runtime Task discovery;
- readiness, result-driven pruning, admission rank, and terminal outcomes;
- fixed, weighted-batch, and pull-worker dispatch;
- Run-level total and GPU Provider Call ceilings;
- durable Provider Call ownership, observation, cancellation, and recovery;
- Result Envelopes, execution artifacts, manifests, and publication records;
- coordinator driving, restart lineage, and successful-result lookup;
- the provider-neutral SQLite execution schema and transition contract.

Workloads own:

- scientific DAG construction and Task discovery logic;
- normalized scientific inputs and fingerprints;
- provider operation selection, arguments, and runtime-image identity;
- scientific cache keys, publication identities, and validation;
- result decoding, scientific aggregation, and final interpretation;
- scientifically or measurably justified batching inputs, such as Task weights.

A reusable workload operation is provider-independent. It may perform normal
filesystem and subprocess side effects, but it accepts plain request values,
explicit paths and configuration, returns a provider-neutral result, and does
not import a provider SDK. Provider wrappers own decorators, resource and image
declarations, mounts, Secrets, call handles, and durability synchronization.

The API service owns users, Jobs, authorization, configuration, presentation,
and other non-execution metadata. It embeds the kernel tables in its database
without exposing service state to the kernel.

## App and workflow distinction

App and workflow are authoring and deployment terms, not different execution
models.

- An app packages a reusable scientific capability and its provider operations.
- A workflow composes capabilities into a larger scientific pipeline.
- Both submit an `ExecutionDefinition` to the same kernel.
- Both use the same Node, Task, dispatch, artifact, and recovery contracts.

Workflow discovery, deployment composition, CLI presentation, and DAG display
remain under `biomodals.workflow`. Generic graph execution does not.

## Target module shape

The package should grow only as implementation requires. The intended seams
are:

```text
src/biomodals/execution/
  __init__.py          # small supported provider-neutral interface
  model.py             # Run, Node, Task, result, artifact, identity values
  definition.py        # executable graph and Node interfaces
  scheduler.py         # readiness, ranking, and dispatch policies
  runtime.py           # shared execution lifecycle
  sqlite.py            # repository schema and atomic transitions
  coordinator.py       # provider-neutral drive and resume loops
  artifacts.py         # artifact records and provider-neutral validation
  modal/
    __init__.py        # small supported Modal-host interface
    driver.py          # Modal SDK resolution, calls, and error translation
    host.py            # coordinator, request, lineage, Volume, and run storage
```

Exact internal filenames may change when a deeper module removes an unnecessary
seam. The two supported interfaces are `biomodals.execution` and
`biomodals.execution.modal`; callers should not assemble stores, locks,
checkpoints, or provider adapters themselves.

## Provider seam

Root execution modules use provider-neutral terms. `DeploymentIdentity` keeps
its established name but identifies a provider, deployment namespace/name, and
version without assuming Modal Apps or Functions. Bindings name provider
operations rather than Modal functions.

The Modal integration maps that interface to Modal Environment, App, Function,
Function Call, and Volume behavior. A future provider can add a sibling
`biomodals.execution.<provider>` integration without changing scheduler or
ledger semantics.

For example, a future `biomodals.execution.local` integration could map a
Deployment Identity to an OCI image digest and operation registry, a Provider
Call ID to a durable Docker or Podman container ID, and artifact storage to the
local filesystem. A plain child process is insufficient for recoverable
detached Runs unless another durable host owns and observes it. The local
integration is explicitly deferred; this example constrains the provider seam
without adding speculative implementation.

No universal deployed coordinator app is introduced. Each app or workflow
continues to include a thin, deployment-local Modal composition root so its
decorated functions, images, resources, Volumes, and exact deployment version
remain pinned together.

Existing Modal functions become thin wrappers over provider-independent
operation bodies when that extraction is simple and behavior-preserving. Modal
image definitions remain Modal-owned; future local execution will require its
own OCI image and mount bindings rather than assuming a Modal Image can run
locally unchanged.

## Workload interface

An app or workflow supplies an immutable `ExecutionDefinition` containing
executable Nodes. A Node supplies scientific behavior through a small shared
interface:

- discover an ordered, finite Task set when its inputs become available;
- describe provider binding, compatibility, arguments, and dispatch policy;
- observe and validate existing publications;
- decode a Result Envelope and publish Task outputs;
- finalize Node outputs when aggregation is required.

The kernel owns the lifecycle around those operations. Workloads do not
subclass or reproduce initialization, drive loops, readiness, admission,
submission, polling, restart, or cancellation.

The initial closed dispatch-policy set is:

- one Task per Provider Call;
- fixed or weighted Tasks per Provider Call;
- pull workers claiming bounded Task microbatches.

AlphaFold3 seed balancing and AF3Score length balancing use the weighted policy.
Rosetta uses pull-worker dispatch. Workloads provide scientific weights and
compatibility; they do not implement admission loops.

## Modal host

`biomodals.execution.modal` owns common remote host mechanics:

- immutable execution-request and launch-lineage persistence;
- per-Run Volume-backed SQLite storage;
- Volume commit/reload and lock ordering;
- root and Successor preparation;
- coordinator status, cancellation, resume, and drive methods;
- development and deployed coordinator resolution;
- shared CLI submission, identity reporting, waiting, and terminal validation.

Local Entrypoints remain thin. They parse workload arguments, stage local
scientific inputs, construct the workload request, and call the Modal host.

The API service does not use the Volume-backed CLI host. It uses the same
provider-neutral definition and asynchronous execution runtime with its own
database and process lifecycle, plus the Modal driver from
`biomodals.execution.modal`.

## Source layout

Simple workloads may keep their execution adapter in a sibling
`<name>_execution.py`. A complicated workload may use a sibling package such as
`app/score/af3score/`, mirroring the existing AlphaFold3 layout. The discoverable
`*_app.py` module retains scientific functions, Modal decorators, and thin
entrypoints rather than generic orchestration.

## Compatibility and behavior

There is no compatibility layer for pre-release Python imports or execution
ledger schemas. The migration will:

- bump the execution schema version;
- reject older ledgers explicitly;
- remove obsolete migration and compatibility code;
- provide no aliases for `helper.app_execution`, `workflow.core`, or renamed
  shared execution types.

The refactor preserves scientific DAGs, fingerprints, cache identities, output
layouts, provider decorators, public CLI arguments, and observable execution
behavior. OligoFormer's custom per-Node admission is removed in favor of kernel
admission. Any other scientifically necessary, performance-motivated, or
behavior-changing exception requires a separate explicit decision.

Local-container execution, parallel provider image definitions, and a generic
provider registry are outside this refactor. The code should leave a real seam
for them without implementing or testing hypothetical providers.

## Implementation sequence

Each step is an independently reviewable commit with focused tests kept green:

1. Record this domain and interface amendment.
2. Introduce provider-neutral names and the `execution.modal` package.
3. Move and deepen `helper.app_execution`, then delete it.
4. Move generic graph, artifact, runtime, and coordinator behavior from
   `workflow.core`, then delete it.
5. Establish the shared `ExecutionDefinition`, Node, artifact, and dispatch
   interfaces.
6. Migrate simple direct apps and remove equivalent lifecycle code.
7. Migrate specialized apps while preserving their scientific hooks.
8. Migrate ShortMD, RFD-LigandMPNN, and PPIFlow.
9. Make API app calls consume the same app-owned definitions.
10. Remove obsolete names, schema migration code, tests, and documentation.
11. Run full local verification and report per-workload and total LOC changes.

## Completion criteria

The consolidation is complete when:

- `helper.app_execution` and `workflow.core` no longer exist;
- apps and workflows use one executable graph and artifact model;
- no workload reproduces standard initialization or admission;
- common root, restart, and coordinator submission flows have one owner;
- provider-neutral modules import no Modal SDK code;
- Modal-specific behavior is confined to `execution.modal` and thin decorated
  deployment roots;
- shared workload operation bodies do not import Modal APIs or depend on Modal
  handles;
- direct CLI and service calls for the same app reuse one scientific definition;
- the full local test suite, lint, type guidance, and Python 3.11 task-image
  import checks pass;
- no cost-incurring provider run occurs without explicit approval.

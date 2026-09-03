<!-- markdownlint-disable MD013 -->

# Execution kernel consolidation

Status: implemented on 2026-08-20. A local-container provider remains
deferred.

This specification amends
[ADR 0006](../adr/0006-unified-execution-kernel.md) and the
[unified scheduler specification](unified-task-scheduler.md). Where an older
statement conflicts with this specification, this specification is
authoritative. Older text is retained only as labeled history.

## Objective

Biomodals has one provider-neutral execution kernel for apps, workflows, and
API-owned app calls. Provider integrations live below that kernel. Modal is the
only provider integration implemented today.

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

The implemented package has these maintained seams:

```text
src/biomodals/execution/
  __init__.py          # small supported provider-neutral interface
  model.py             # Run, Node, Task, call, and identity values
  definition.py        # executable graph and Node interfaces
  definition_plan.py   # immutable graph-to-plan conversion
  definition_runtime.py # definition-owned runtime integration
  scheduler.py         # readiness, ranking, and dispatch policies
  runtime.py           # shared execution lifecycle
  sqlite.py            # repository schema and atomic transitions
  coordinator.py       # provider-neutral drive and resume loops
  artifacts.py         # artifact records and provider-neutral validation
  artifact_availability.py # reusable tri-state availability checks
  artifact_store.py    # execution artifact persistence
  hashing.py           # stable scientific and artifact hashing
  provider.py          # provider driver protocol and observations
  store.py             # host-supplied Run storage boundaries
  pull_worker.py       # provider-neutral pull-worker loop
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

Provider-independent does not mean mathematically pure. An operation may read
and write files, invoke subprocesses, and use devices exposed inside its
container. It must receive paths and configuration explicitly, return the
shared execution result contract, and leave provider object resolution and
container lifecycle to its wrapper and provider integration.

Provider support is declared per operation. An Execution Definition can run
through a provider only when every remote operation selected by that graph has
a provider binding and suitable packaging. Adding a local provider therefore
does not make every existing app local automatically: already-explicit
operations need only a local wrapper and OCI image, while operations coupled to
Modal lifecycle or implicit mounts must first extract those concerns. The DAG,
cache identity, scheduler policy, and scientific implementation remain shared.

A provider binding maps one logical operation name to a provider entrypoint,
container packaging, resources, mounts, environment, Secrets, and argument and
result translation. These bindings belong to the app or workflow composition
root because they package workload code; generic submission, observation,
cancellation, and durable call identity belong to the provider subpackage.

The provider host must validate before admission that every operation selected
by an Execution Definition has a binding. A partially migrated app therefore
remains unsupported by that provider rather than switching providers partway
through a Run.

The portable operation body must be directly testable without importing or
hydrating a provider SDK object. It may use files, subprocesses, accelerators,
and ordinary side effects, but provider mount roots and configuration are
explicit inputs. Provider wrappers may translate those roots and result
envelopes; they must invoke the same scientific body and preserve its output
layout and scientific identity.

Provider subpackages implement generic host mechanics only. They do not own a
workload registry or app-specific images. Each app or workflow composition root
maps the logical operations in its Execution Definition to that provider's
executable, packaging, resources, mounts, and configuration. This keeps adding
`biomodals.execution.local` independent from migrating every workload to it.

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
provider-neutral definition and synchronous `ExecutionRuntime` behind its
asynchronous service boundary, with its own database and process lifecycle. Its
provider adapter uses `AsyncModalCallDriver` from
`biomodals.execution.modal`.

## Source layout

Simple workloads may remain in a discoverable `<name>_app.py`. Multi-module
workloads use a discoverable `<name>/app.py` composition root with execution
adapters and domain modules in the same package. The composition root retains
scientific functions, Modal decorators, and thin entrypoints rather than
generic orchestration.

## Compatibility and behavior

There is no compatibility layer for pre-release Python imports or execution
ledger schemas. The migration:

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

## Implemented sequence

The consolidation was delivered as independently reviewable commits with
focused tests kept green:

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

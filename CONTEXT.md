<!-- markdownlint-disable MD013 -->

# Biomodals

Biomodals runs bioinformatics tools as Modal apps and composes them into reusable computational workflows.

## Language

### Execution scheduling

**Execution Run**:
One invocation of an immutable execution plan, whether started by an API Job,
a workflow, or an app entrypoint. It has one opaque Execution Run ID that is
independent of workload naming and scientific identity.
_Avoid_: API Job, workflow definition

**Execution Run ID** [planned]:
A kernel-generated UUID identifying exactly one Execution Run. It is generated
before admission and used for repository keys, coordinator routing, lineage,
and ledger paths.
_Avoid_: Workload Run Key, Service Job ID, display name

**Workload Run Key** [planned]:
An optional workload-owned name or scientific key that influences provider
arguments, output paths, or publication identity. Successor Execution Runs may
reuse it, and the kernel stores it only as immutable workload plan input.
_Avoid_: Execution Run ID, ledger path, display label alone

**Workload Plan Fingerprint** [planned]:
A stable digest of every normalized result-affecting input and declared
scientific tool, model, adapter, or schema version for an Execution Run.
Successor Execution Runs require the same fingerprint. File contents are
represented by content digests; operational concurrency, batching, resources,
and Deployment Identity are excluded.
_Avoid_: command-text hash, file path alone, Execution Run ID, deployment version alone

**Execution Run Status** [planned]:
The kernel lifecycle value for an Execution Run: `pending`, `running`,
`cancel_requested`, `suspended`, `state_unknown`, `succeeded`, `partial`,
`failed`, or `cancelled`. The first five are nonterminal and the final four are
terminal. Structured reasons refine a status without creating another status.
_Avoid_: queued, finalizing, blocked, interrupted

**Execution Run Status Reason** [planned]:
The optional stable, machine-readable `status_reason` code that explains the
current Execution Run Status. An optional human-readable `status_message`
provides diagnostics but is never used for control flow. Task- and
Node-specific failures remain canonical on those records; the Run fields only
summarize why its lifecycle changed. The initial reason vocabulary is:
`coordinator_error`, `result_validation_unknown`,
`submission_outcome_unknown`,
`provider_outcome_unknown`, `cancellation_outcome_unknown`,
`required_work_failed`, and `deployment_unavailable`.
_Avoid_: status-specific reason columns, free-text state machine, copied stack trace

**Successor Execution Run** [planned]:
A new Execution Run created by an explicit restart of an eligible terminal
Run. It records its predecessor, uses a newly resolved Deployment Identity and
Execution Run ID, reuses the Workload Run Key when applicable, revalidates
Workload Publications, and schedules only missing work whose predecessor
Provider Call is conclusively terminal. It is the only way to retry failed
provider work.
_Avoid_: in-place migration, same-run retry, provider redelivery

**Successor Repair Closure** [planned]:
The union of the backward ancestor closures needed to repair predecessor
terminal Nodes that were `partial`, `failed`, `skipped`, or `cancelled`, plus
any previously successful terminal whose publication no longer validates.
Traversal stops at complete reusable publications. Within the closure,
successful Task publications are reused and only conclusively unowned missing
work may be submitted in the Successor Execution Run.
_Avoid_: rerun whole DAG, reopen predecessor, accept partial as complete

**Suspended Run** [planned]:
A nonterminal Execution Run whose coordinator stopped after an application
error or an unknown Node or Task publication observation. It admits no new
work until explicit resume reconciles durable state and repeats any unresolved
validation.
_Avoid_: failed Task, provider state unknown, Modal preemption

**State-Unknown Run** [planned]:
A nonterminal Execution Run whose provider submission, call state, or
cancellation outcome cannot be established. It preserves ownership and forbids
replacement work until explicit reconciliation or administrative resolution.
_Avoid_: suspended Run, missing publication, failed Run

**Execution Node**:
A fixed semantic stage in an execution DAG that may discover one or more Tasks
when it becomes ready. Its identity and dependencies do not change when
provider batch size or concurrency changes.
_Avoid_: Modal function, dynamic task, provider call

**Execution Node Status** [planned]:
The kernel lifecycle value for an Execution Node: `pending`, `running`,
`succeeded`, `partial`, `failed`, `cancelled`, or `skipped`. Only `pending`
and `running` are nonterminal. Readiness is derived rather than persisted, and
cache reuse is Task provenance rather than a Node status. The optional
`status_reason=result_already_satisfied` distinguishes an ancestor pruned by a
complete terminal result from ordinary dependency failure or explicit Run
cancellation.
_Avoid_: ready, cached, suspended, state unknown

**Terminal Execution Node** [planned]:
An Execution Node with no downstream dependency. Terminal Execution Nodes
collectively define an Execution Run's scientific result boundary. Their
validated publications determine the Run outcome; upstream Node history does
not override a complete terminal result.
_Avoid_: last executed node, designated result node, all-Node aggregation

**Node Result Observation** [planned]:
The workload-owned `available`, `missing`, or `unknown` answer from validating
one Execution Node's complete publication before its dependencies or Tasks
run. `available` prunes unnecessary ancestors, `missing` expands the backward
repair closure, and `unknown` blocks new work. A partial publication is not an
available complete Node result.
_Avoid_: boolean cache hit, fake terminal Task, kernel-owned scientific validator

**Partial Dependency Acceptance** [planned]:
The immutable `accept_partial` boolean on one Node dependency edge. Success
always satisfies the edge; a partial upstream Node satisfies it only when the
edge opts in. Failed, cancelled, and skipped outcomes never satisfy it.
_Avoid_: global partial mode, implicit best effort, accepted-status set

**Node Aggregation Policy** [planned]:
The immutable `fail_fast`, `collect_all`, or `allow_partial` rule by which one
Execution Node admits Tasks and derives its outcome. Fail-fast stops new
admission without cancelling owned work; collect-all is strict after every
Task finishes; allow-partial succeeds partially only when at least one Task
succeeds.
_Avoid_: same-Run retry policy, implicit cancellation, arbitrary success threshold

**Explicit Empty Result** [planned]:
A workload-published and validated complete Node result created after a
`NodePlan` with `allow_empty_result=True` discovers zero Tasks. The boolean is
part of the Workload Plan Fingerprint. An empty in-memory collection alone is
never evidence of scientific completion.
_Avoid_: vacuous success, implicit empty cache hit, zero Tasks means succeeded

**Task**:
The smallest independently scheduled and validated work item in an Execution
Node whose cached publication and outcome can be observed. Every Task belongs
to exactly one Node.
_Avoid_: workflow node, thread, untracked work item

**Task Fingerprint** [planned]:
The kernel-computed SHA-256 digest of compact canonical JSON containing the
Workload Plan Fingerprint, Node key, Task key, and workload-normalized
scientific payload. It is calculated once at discovery and persisted.
Operational execution payloads, provider kwargs, paths, batching, resources,
and call identity are excluded.
_Avoid_: workload-supplied opaque digest, whole-file hashing, polling-time recomputation

**Task Discovery Checkpoint** [planned]:
The atomic per-Node repository transition that validates and inserts the
complete finite set of `TaskPlan` records with unique stable Node-local keys,
then marks discovery complete. The host durability boundary must be crossed
before any Task can acquire an execution owner. Recovery either rediscovers
the whole set or reloads the whole set, never a partial queue.
_Avoid_: incremental Task creation, half-discovered queue, worker-side discovery

**Task Status** [planned]:
The kernel lifecycle value for a Task: `pending`, `running`, `succeeded`,
`failed`, `cancelled`, or `skipped`. Only `pending` and `running` are
nonterminal. A Task becomes `running` when durable local or provider ownership
is assigned and retains that status while its owner outcome is unknown.
Partiality belongs to Node aggregation, and cache reuse is success provenance.
The optional `status_reason=result_already_satisfied` identifies an unowned
Task skipped or an owned Task cancelled because its terminal result no longer
needed that ancestor work.
_Avoid_: partial, cached, submitting, attached, state unknown

**Single-Submission Rule** [planned]:
Within one Execution Run, the kernel schedules each Task once and submits at
most one Provider Call or Worker Assignment for it. Provider redelivery may
re-execute that same call, so this rule does not claim exactly-once execution.
A conclusive failure terminates the Task; retry requires a Successor Execution
Run.
_Avoid_: exactly-once execution, same-run retry, attempt counter

**Submission Preclaim** [planned]:
The atomic repository operation that creates a Provider Call in `submitting`
and durably assigns its Tasks before the external spawn side effect. Only the
caller that created the row and crossed its host durability boundary receives
permission to invoke spawn; duplicate requests and recovered coordinators
preserve the existing owner without spawning.
_Avoid_: retry token, timeout lease, Task Attempt

**Provider Call**:
One concrete remote worker invocation submitted to a compute provider and
identified for later observation, recovery, logging, or cancellation. It
belongs to exactly one Node and may own zero or more of that Node's Tasks. A
Modal Function Call is the current provider-specific implementation.
_Avoid_: Task, Node, Coordinator Attempt, provider redelivery, app

**Provider Call Status** [planned]:
The kernel lifecycle value for a Provider Call: `submitting`, `attached`,
`running`, `outcome_unknown`, `state_unknown`, `succeeded`, `failed`, or
`cancelled`. The first five are nonterminal and preserve ownership; the final
three are terminal. `outcome_unknown` has no durably attached provider call ID,
whereas `state_unknown` does.
_Avoid_: planned, expired, retrying

**Dispatch Batch** [planned]:
A durable grouping of Tasks from one Node offered together to one Provider
Call or to a shared pull worker pool. It records intended dispatch without
claiming that a particular worker performed a Task before a Worker Assignment
is committed.
_Avoid_: workflow node, provider call, scientific batch

**Worker Assignment** [planned]:
A durable SQLite record linking one Task to the Provider Call and worker claim
responsible for it. The coordinator commits and checkpoints the assignment
before returning its payload. Repeating the same claim request returns the
same assignment.
_Avoid_: queue item, timeout lease, retry, publication

**Task Claim Request** [planned]:
An idempotent request from a pull worker for a bounded set of ready Tasks.
Its stable request ID lets a replacement worker recover the same committed
Worker Assignments after a lost response or provider restart.
_Avoid_: automatic retry, timeout lease

**Execution State Repository**:
A durable record of Execution Runs, Nodes, Tasks, Dispatch Batches, Worker
Assignments, and Provider Calls governed by the execution kernel's transition
contract. Each durable coordinator may use a separate physical repository.
_Avoid_: universal service database, scientific cache

**App Run Ledger** [planned]:
The physical per-run SQLite Execution State Repository for a Direct CLI App
Run, stored at
`.biomodals/execution/runs/<execution-run-id>/ledger.sqlite3` in that app
deployment's configured durable Volume.
_Avoid_: scientific output directory, Workflow Ledger, shared execution Volume

**Execution Coordinator** [planned]:
The logical scheduling authority that serializes transitions in an Execution
State Repository and advances active Execution Runs independently of their
launching clients. It outlives any process or container temporarily performing
that work.
_Avoid_: worker, SQLite writer container, Provider Call

**Coordinator Attempt** [planned]:
One continuous tenure in which a process or container actively advances an
Execution Coordinator. An interruption ends the Attempt without cancelling
the Coordinator's Runs or child Provider Calls.
_Avoid_: Task, provider retry, Execution Run

**Coordinator Interruption** [planned]:
A non-user-requested loss or shutdown of the current Coordinator Attempt that
requires a replacement Attempt to recover durable execution state.
_Avoid_: Job cancellation, Task failure, Provider Call cancellation

**Run-Scoped Coordinator Pool** [planned]:
A provider-routed container pool created by a Deployment Coordinator Adapter
and identified by an Execution Run ID and Deployment Identity. It admits at
most one coordinator container for that identity; concurrent control requests
submit commands to that container's single SQLite writer.
_Avoid_: worker pool, timeout lease, service database

**Deployment Coordinator Adapter** [planned]:
A thin Modal binding included in each app or workflow deployment. It binds the
shared execution kernel to that deployment's workload hooks, Volumes, and
configuration without introducing a universal coordinator service.
_Avoid_: execution kernel, workload registry, API service

**Deployment Identity** [planned]:
The Modal Environment, deployed app or workflow name, and exact numeric
deployment version selected and persisted before an Execution Run admits work.
An explicit CLI version wins; otherwise the CLI resolves current deployment
history once and pins the result.
_Avoid_: floating latest handle, semantic app version, source revision alone

**Deployed CLI Run** [planned]:
A top-level app or workflow Execution Run submitted by the Biomodals CLI to an
exact Deployment Identity. It may be observed or resumed across local CLI
processes through its remote Run-Scoped Coordinator Pool.
_Avoid_: ephemeral development run, API Job, Child App Call

**Direct CLI App Run** [planned]:
A Deployed CLI Run initiated through `biomodals app run`. Its durable
repository lives remotely; the user's machine does not create or own a run
database. Repeating a launch without predecessor identity creates a new root
Run; `--restart-from <execution-run-id>` explicitly creates a Successor
Execution Run and is a convenience over the generic restart command.
_Avoid_: Child App Call, local scheduler, API Job

**Development CLI Run** [planned]:
An explicitly requested source-backed app or workflow run using an ephemeral
Modal deployment. It may use the remote kernel but promises no
cross-invocation resume after that deployment expires.
_Avoid_: Deployed CLI Run, production deployment, dry run

**Service Job**:
A user-facing API service record for ownership, admission, configuration,
result delivery, and presentation that refers to an Execution Run without
persisting a duplicate compute state.
_Avoid_: Execution Run, Task, provider call

**Job State Projection**:
The user-facing Job state and timeline derived from an Execution Run together
with Service Job result-delivery metadata.
_Avoid_: persisted scheduler state, duplicate task status

**Workload Publication**:
Workload-owned durable evidence that a Task's scientific output is complete
and reusable.
_Avoid_: provider success, build claim, database status alone

**Workflow Artifact**:
A durable record of data produced or consumed by a workflow step, including its data category, storage location, and metadata needed by downstream steps.
_Avoid_: raw app output, untyped file path, loose tarball

**Artifact Availability**:
The observed state of a workflow artifact as available, missing, or unknown; unknown means verification could not establish presence or absence.
_Avoid_: boolean existence, checker success

**Inline Byte Output**:
A workflow-compatible app output whose bytes are small enough to serialize directly in a Pydantic JSON payload before materialization.
_Avoid_: large archive, arbitrary binary bytes

**Workflow Node**:
A semantic step in a workflow DAG that consumes workflow artifacts and produces workflow artifacts.
_Avoid_: Modal function, app function

**Terminal Workflow Node**:
A workflow node with no downstream dependencies in a validated workflow DAG.
_Avoid_: final node, last node

**App**:
A deployed Modal app that owns tool runtime, images, volumes, and exported app functions.
_Avoid_: workflow node, app node

**Shard Build Recipe**:
A versioned deterministic transformation from a reference-database monolith into the shard layout of a Sharded Database Profile.
_Avoid_: ad hoc split, shard command

**Sharded Database Profile**:
An immutable, manifest-validated publication of every shard for one logical MSA database. Its manifest preserves the source FASTA identity and construction recipe, but the published profile does not retain a duplicate source FASTA.
_Avoid_: source database directory, incomplete shard staging, arbitrary shard set

**Profile ID**:
A code-owned identifier for one immutable Sharded Database Profile, fixing its source database generation, shard count, and build recipe. It is selected by the Supported Database Specification rather than through a mutable runtime alias.
_Avoid_: current profile, database ID, manifest digest

**Profile Build Claim**:
A minimal, append-only Modal Dict election record allowing one invocation to construct one Profile ID. Conflicts fail fast; a later explicit invocation may advance beyond a failed or conservatively stale generation.
_Avoid_: published profile, search build claim, polling lock service

**Source FASTA Policy**:
The explicit post-publication choice to keep, round-trip-verify and archivally compress, or delete an original database FASTA after its Sharded Database Profile is durably validated. A compressed source must be restored manually before another profile build.
_Avoid_: temporary builder cleanup, shard compression, implicit retention, automatic source restore

**Database Search Space**:
The full unsharded database size used by HMMER to scale hit E-values across a Sharded Database Profile. It is the exact sequence count for protein searches and the exact nucleotide count expressed in megabases for RNA searches.
_Avoid_: shard count, FASTA byte size, per-shard record count

**Supported Database Specification**:
The code-owned production definition of one official AlphaFold MSA database, including its logical identifier, source filename and molecule type, accepted shard count, and expected source statistics.
_Avoid_: runtime override, profile manifest, arbitrary FASTA

**MSA Search Subject**:
A unique biological sequence that requires database-generated MSA evidence and may be referenced by one or more input chains.
_Avoid_: chain identifier, duplicate homomer chain

**Polymer Cache Namespace**:
The top-level `Protein/` or `RNA/` directory that separates MSA cache entries for identical sequence hashes interpreted as different polymer types.
_Avoid_: database ID, polymer-aware sequence hash, chain type suffix

**Raw Database MSA**:
A validated result of searching one MSA Search Subject against one reference database profile. It is independently complete and reusable before AlphaFold constructs combined unpaired or paired MSAs.
_Avoid_: combined unpaired MSA, paired MSA, search log

**Combined Unpaired MSA**:
The AlphaFold-ready unpaired alignment assembled from validated Raw Database MSAs in pinned upstream order, with duplicate aligned sequences removed after ignoring lowercase insertions. Protein order is UniRef90, small BFD, then MGnify; RNA order is RFam, RNAcentral, then NT-RNA.
_Avoid_: raw database MSA, simple FASTA concatenation

**Combined Paired MSA**:
The AlphaFold-ready paired protein alignment assembled from the UniProt Raw Database MSA without deduplication. RNA inputs have no paired MSA.
_Avoid_: combined unpaired MSA, RNA MSA

**Combined MSA Publication**:
The latest validated sequence-root `unpaired.a3m` and, for protein, `paired.a3m` derived from the currently selected Raw Database MSAs. Its completion manifest, not file existence, proves which raw results and merge semantics it represents.
_Avoid_: raw database MSA, unmarked legacy file, versioned assembly archive

**Template Search Result**:
A validated AlphaFold-ready list of selected protein templates derived from one Combined Unpaired MSA and the Immutable Template Store. The sequence root retains only the latest publication, and an empty list is a complete result distinct from an unfinished search.
_Avoid_: PDB/mmCIF reference store, raw database MSA, missing result

**Immutable Template Store**:
The fixed upstream PDB seqres file and `mmcif_files/` directory used for protein template search. Their contents are treated as immutable infrastructure and do not receive a separate cache identity or digest.
_Avoid_: sharded database profile, template search result, versioned reference

**Enriched AlphaFold Input**:
An AlphaFold input whose protein and RNA search fields are all explicit, using searched evidence, non-empty caller-supplied values, or deliberate empty sentinels according to the selected search policy. It is ready for inference without rerunning AlphaFold's data pipeline.
_Avoid_: raw input, model features, partially populated search fields

**Caller-Supplied Search Evidence**:
Non-empty MSA or template data supplied for one chain in one request, either inline or materialized from a caller path. It may contribute to that chain's Enriched AlphaFold Input but is neither propagated to identical sibling chains nor made shared canonical search evidence.
_Avoid_: raw database MSA, combined MSA publication, template search result

**Staged Custom Template**:
A caller-supplied mmCIF made remotely accessible at a content-addressed path beneath one AlphaFold Run Root. Its content digest and residue mappings define biological identity; its original and staged paths do not.
_Avoid_: template search result, local template path, shared template cache

**Staged Inference Input**:
The marker-complete canonical request input, run identity, and path-backed custom templates stored beneath one AlphaFold Run Root. GPU workers and summary finalizers load and re-derive its run and request identities instead of accepting caller-supplied inference JSON.
_Avoid_: unchecked worker payload, enriched local input, seed prediction

**Search Field Resolution**:
The request-scoped selection or generation of each MSA and template field independently. A populated field neither authorizes its replacement nor implies that a missing sibling field is resolved.
_Avoid_: chain-wide search, all-or-nothing data pipeline

**Partial Search Resolution**:
A Search Field Resolution with reusable canonical database or template results but at least one required field still incomplete after a surfaced CPU-task failure. It cannot produce an Enriched AlphaFold Input.
_Avoid_: empty search result, enriched input, failed request cache

**AlphaFold Run Root**:
The seed-independent `/{run_id[:2]}/{run_id}/` directory at the AlphaFold3 output-Volume root. It makes staged caller inputs available to remote functions and durably owns inference outputs, requests, logs, and completion state.
_Avoid_: MSA cache root, database Volume, local download directory

**Inference Run Identity**:
The stable digest that groups predictions for one Enriched AlphaFold Input and one set of seed-independent scientific inference settings. Model seeds and accelerator class are deliberately excluded so additional seeds and operationally different GPU deployments can share the same AlphaFold Run Root.
_Avoid_: sequence hash, model seed, GPU accelerator class, display name

**Inference Identity View**:
The normalized Enriched AlphaFold Input used to derive Inference Run Identity after removing display name and seeds and replacing operational template paths with content digests. It conservatively retains every other validated input field.
_Avoid_: request input, selected-field whitelist, upstream worker JSON

**Declared Model Identity**:
The code-owned checkpoint label and pinned AlphaFold/app version used in Inference Run Identity without hashing the model file. It assumes the checkpoint is not replaced in place.
_Avoid_: checkpoint digest, model Volume path alone, model seed

**Canonical Output Name**:
The deterministic, run-derived name given to upstream inference so every durable output filename is stable across caller display names. It uses `af3-{run_id[:16]}` and is not a user-facing run label.
_Avoid_: display name, Execution Run ID, local archive name

**Presentation Output Name**:
The sanitized caller display name applied to filenames only while creating a Request Retrieval Archive. It never renames or identifies durable prediction artifacts.
_Avoid_: canonical output name, inference run identity, Volume path

**Seed Prediction**:
The independently complete inference output for one model seed beneath an AlphaFold Run Root. Its diffusion samples use upstream's `seed-{seed}_sample-{sample_index}` directories; optional embeddings and distogram directories belong to the same Seed Prediction.
_Avoid_: inference run identity, container part, combined top-level output

**Seed Completion Marker**:
The minimal authoritative record that a Seed Prediction's upstream process succeeded and its promoted output was committed. It carries sample ranking scores for summary derivation, while readers trust completion without revalidating individual prediction artifacts.
_Avoid_: artifact inventory, worker exit alone, seed build claim

**Inference Worker Staging**:
An exclusive, temporary subtree on the AlphaFold3 output Volume where one GPU container lets upstream write results for its disjoint seed list without colliding with another container's shared files.
_Avoid_: seed prediction, canonical output tree, local scratch directory

**Inference Run Summary**:
The single finalized set of upstream-style top-level data, ranking, and best-prediction files derived from the accumulated union of every validated Seed Prediction beneath one AlphaFold Run Root.
_Avoid_: worker-local ranking, seed sample directory, completion marker

**Prediction Ranking Order**:
The deterministic ordering shared by request-scoped and global prediction summaries: descending ranking score, then ascending model seed, then ascending sample index. It makes equal-score best-sample selection independent of worker completion order.
_Avoid_: worker completion order, submission order, arbitrary equal-score winner

**Inference Request**:
The normalized model-seed set requested against an Inference Run Identity. Its request ID ignores submitted order, duplicate seeds, and display name, and it may reuse existing Seed Predictions without changing the shared run identity.
_Avoid_: inference run identity, request view, GPU worker assignment

**Inference Request View**:
An identity-stable durable presentation of one Inference Request, identified from its request ID, submitted seed order and duplicates, and display name. It references all requested canonical Seed Predictions plus request-specific ranking and best files without recording mutable invocation outcomes or duplicating seed artifacts.
_Avoid_: inference request, complete run archive, accumulated run summary

**Partial Inference Request**:
An Inference Request for which at least one normalized requested seed has a Seed Completion Marker and at least one remains unmarked after a surfaced failure. It retains reusable seeds and diagnostics but has no successful Inference Request View.
_Avoid_: failed seed prediction, completed request, empty run

**Request Retrieval Archive**:
A self-contained local `.tar.zst` materialization of one Inference Request View, assembled by downloading only its manifest-declared canonical artifacts and referenced Staged Custom Templates. Its input copy uses archive-relative template paths, and its filename combines the Presentation Output Name with a view-ID prefix.
_Avoid_: inference request view, global run archive, remote function payload

**Seed Build Claim**:
An atomic, generation-scoped coordination record granting one request ownership of computing one missing Seed Prediction. It is never evidence that the prediction completed; only the validated seed publication is authoritative.
_Avoid_: seed completion marker, inference request, cache entry

**Summary Build Claim**:
An atomic, generation-scoped coordination record granting one finalizer ownership of rebuilding the mutable Inference Run Summary. It serializes publication but never proves which seeds the current summary contains.
_Avoid_: seed build claim, run-summary marker, inference request

**Search Build Claim**:
An atomic, generation-scoped coordination record granting one request ownership of producing one missing Raw Database MSA or publishing one sequence-root combined-MSA or template result. Claims follow the exclusive output path they protect; the validated publication, not the claim, is the reusable scientific evidence.
_Avoid_: search identity, completion marker, database shard

**Search Worker Budget**:
The request-wide maximum number of active CPU search containers across database-MSA and protein-template phases. It bounds operational fanout independently of shard concurrency inside a database worker.
_Avoid_: MSA-only worker limit, shard count, HMMER thread count, number of input chains

**Search Identity**:
A digest of the result-affecting inputs for one Raw Database MSA, stored beneath the full sequence hash. It includes semantic source/shard profile content, scientific search parameters, and pinned tool versions, but excludes build timestamps, thread counts, CPU allocation, and container layout.
_Avoid_: sequence hash, full profile-manifest digest, resource configuration

**Canonical Search Result**:
The immutable production-cache publication for one Raw Database MSA, stored at its Search Identity root only after validation. It preserves the merged database alignment and compact provenance needed for reuse; per-shard merge evidence is transient execution data.
_Avoid_: arbitrary search output, unvalidated sample, shard tblout

**Sequence Hash Prefix**:
The first two hexadecimal characters of the full sequence hash, used only to fan out directories. It is not a sequence identifier by itself.
_Avoid_: sequence hash, search identity

**App Function**:
A callable Modal remote function exposed by a Biomodals app or another Modal app and invoked by a workflow node.
_Avoid_: workflow node

**Child App Call**:
A Modal app function call submitted by an Execution Coordinator as part of a
larger Execution Run. It uses its parent Run's execution state rather than
creating another coordinator repository.
_Avoid_: workflow node, execution run, Direct CLI App Run

**Local Entrypoint**:
A CLI-facing Modal entrypoint that parses and stages local user inputs,
submits a Deployed CLI Run to a remote coordinator, then observes, downloads,
or reports its outputs. It does not own an Execution State Repository.
_Avoid_: workflow entrypoint, execution coordinator

**CLI Namespace**:
A top-level `biomodals` command group that separates app commands from workflow commands.
_Avoid_: mixed app/workflow command collection

**Workflow-Compatible App Function**:
An app function with standardized workflow input and output schemas suitable for app-backed workflow nodes.
_Avoid_: local entrypoint, submit function

**Partial App Run**:
A terminal app-run outcome in which some requested candidate work succeeded and some failed, with successful outputs and failure diagnostics both preserved.
_Avoid_: best-effort success, warning-only success

**siRNA Candidate Set**:
A ranked collection of small interfering RNA candidates designed or scored together for one target mRNA.
_Avoid_: loose siRNA list, unranked outputs

**siRNA Candidate Identity**:
A stable per-candidate identity used to join efficacy, off-target evidence, toxicity, and final selection results for the same siRNA candidate.
_Avoid_: top-N rank, sorted row number, incidental list position

**Off-Target Reference Set**:
A collection of non-target transcript regions used to estimate unintended siRNA binding during off-target prediction.
_Avoid_: background FASTA, all-human mode

**siRNA Candidate Batch**:
A subset of a siRNA candidate set that is scored together against one or more off-target reference shards.
_Avoid_: single siRNA job, candidate chunk

**Off-Target Reference Shard**:
A transcript-aligned subset of an off-target reference set that preserves the UTR, ORF, and related transcript-region records needed to score candidates against that subset.
_Avoid_: loose FASTA split, UTR-only shard

**Off-Target Scoring Tile**:
The pair of one siRNA candidate batch and one off-target reference shard whose partial off-target evidence can be combined with other tiles for the same candidate set.
_Avoid_: worker job, Modal task, queue item

**Off-Target Tile Manifest**:
A run-level manifest of typed off-target scoring tiles that defines the finite TargetScan and PITA work to run for one siRNA candidate set.
_Avoid_: worker queue, dynamic task list

**Off-Target Evidence Table**:
A transcript-level table of partial TargetScan or PITA off-target evidence that can be merged across scoring tiles before candidate-level filtering or ranking.
_Avoid_: final score table, shard output

**Upstream Equivalence**:
Agreement between Biomodals-produced tables and tables produced by invoking the wrapped upstream tools directly, after canonicalizing table order and applying defined numeric tolerances.
_Avoid_: raw byte identity, same app implementation comparison

**Generated Scaffold Segment**:
A de novo structure segment introduced by RFdiffusion from a numeric contig segment rather than copied from the input PDB.
_Avoid_: generated position, inpainted position

**Fixed Motif Segment**:
A structure segment copied from the input PDB by RFdiffusion from a chain-qualified contig segment.
_Avoid_: fixed position, reference position

**LigandMPNN Redesign Set**:
The output-PDB residues that LigandMPNN should sequence-design after RFdiffusion, excluding residues that belong to fixed motif segments.
_Avoid_: all scaffold residues, RFdiffusion positions

**RFdiffusion Output Mapping**:
RFdiffusion metadata that relates input-PDB residues copied from contigs to their residue labels in a generated output PDB.
_Avoid_: inferred contig positions, guessed residue ranges

**RFdiffusion Trajectory**:
One independent RFdiffusion inference call for an input PDB and contig specification.
_Avoid_: RFdiffusion replicate, RFdiffusion node count

**RFdiffusion Design**:
One output structure emitted by a single RFdiffusion trajectory.
_Avoid_: trajectory, LigandMPNN design

**LigandMPNN Sequence Batch**:
The set of sequences generated by LigandMPNN for one RFdiffusion design under the requested seeds, batch size, and number of batches.
_Avoid_: RFdiffusion design, trajectory

**PPIFlow Candidate Set**:
The complete collection of structures produced by one PPIFlow design stage and carried together through downstream design, packing, scoring, refolding, and comparison steps.
_Avoid_: first structure, representative structure, single design

**PPIFlow Candidate Identity**:
The stable provenance that relates one PPIFlow candidate to every derived structure, score, and comparison result produced for that candidate.
_Avoid_: list position, incidental filename ordering

**PPIFlow Candidate Manifest**:
A stage-level record that lists each PPIFlow candidate, its candidate identity, parent candidate identity, artifact locations, and per-candidate outcome for one workflow stage.
_Avoid_: sorted file list, implicit output directory

**PPIFlow Candidate Join**:
An alignment of PPIFlow structures, scores, sequence tables, manifests, and reports by candidate identity.
_Avoid_: filename-order pairing, silent row dropping

**PPIFlow Candidate Filter**:
A workflow stage that narrows the active PPIFlow candidate set while retaining rejected-candidate status for audit.
_Avoid_: score-row subset, deleted candidates

**PPIFlow Candidate Attrition**:
The stage-by-stage record of candidates retained, rejected, failed, or skipped through a PPIFlow run.
_Avoid_: final count only, lost rejected candidates

**PPIFlow Sequence Table**:
A candidate-keyed table of LigandMPNN-designed sequences and provenance used by PPIFlow stages.
_Avoid_: unkeyed FASTA, app-specific sequence dump

**PPIFlow MPNN Mode**:
A PPIFlow stage configuration that selects binder-MPNN or AbMPNN behavior while preserving the same workflow-node contract.
_Avoid_: separate node type, hidden app variant

**PPIFlow Interface Energy Analysis**:
Workflow-owned Rosetta post-processing that produces residue-level interface energy tables for deriving fixed-position constraints in PPIFlow stage 2.
_Avoid_: generic Rosetta app contract, ad hoc log parsing

**PPIFlow Rosetta Job Manifest**:
A workflow-owned record of PPIFlow Rosetta inputs, commands, queue entries, expected outputs, and per-candidate Rosetta outcomes.
_Avoid_: generic Rosetta queue state, worker log scraping

**ReFold Quality Metrics**:
Candidate-keyed confidence and ranking metrics for structures produced by PPIFlow's ReFold stage.
_Avoid_: unkeyed AlphaFold3 JSON, structure-only refold output

**Shared Schema**:
A stable Pydantic contract used across Biomodals packages without depending on app or workflow implementation modules.
_Avoid_: app config, internal model

**App Configuration Schema**:
The pure Pydantic fields and validators that describe a Biomodals app's metadata and runtime settings.
_Avoid_: Modal volume factory, image helper

**App Run Layout**:
A standard per-run directory contract for Biomodals apps that defines where inputs, outputs, logs, failures, metrics, and completion markers live under an app's local scratch directory or mounted output volume.
_Avoid_: ad hoc workdir, loose output folder, app-specific run paths

**Canonical Run Name**:
A stable app-run identifier that contains no path traversal or separator semantics and maps to exactly one directory below an app's run root.
_Avoid_: raw user path, silently normalized cache key

**App-Backed Node**:
A workflow node implemented by calling one or more app functions.
_Avoid_: app node, runner node

**Workflow-Native Node**:
A workflow node implemented directly in workflow code for orchestration, transformation, selection, ranking, packaging, or reporting.
_Avoid_: runtime node, orchestrator node

**Workflow Builder**:
A Python interface for declaring workflow nodes, dependencies, artifact selectors, and execution settings before a workflow run.
_Avoid_: workflow YAML, scheduler config

**Artifact Selector**:
A named input reference that selects upstream workflow artifacts by kind, file role, path pattern, metadata, or producing node.
_Avoid_: raw input path, wildcard-only dependency

**Control Edge**:
A dependency between workflow nodes that enforces execution order without passing workflow artifacts.
_Avoid_: dummy artifact

**Dynamic Task Fan-Out** [planned]:
A workflow node execution pattern where the DAG node is fixed but the number of per-input tasks is determined from upstream artifacts at runtime.
_Avoid_: dynamic DAG

**Worker Pool**:
A bounded process or thread pool from `concurrent.futures` that limits concurrent task execution within one workflow node.
_Avoid_: server pool, runner server

**Workflow Node Parallelism**:
The number of ready workflow nodes the workflow runtime may start concurrently in one scheduler wave.
_Avoid_: global Modal container limit, child app concurrency

**App-Local Scheduler**:
A tool-specific queue, worker pool, pod pool, or fan-out loop that directly
coordinates an app's concurrent Tasks.
_Avoid_: execution kernel, provider autoscaler

**Run-Level Task Budget**:
A coordinator-enforced concurrency budget for Tasks and Provider Calls in one
Execution Run.
_Avoid_: service admission limit, Modal deployment limit, cross-run global cap

**Workflow Runtime**:
The reusable library that validates a workflow DAG, schedules workflow nodes, tracks durable run state, and materializes workflow artifacts.
_Avoid_: engine

**Runtime Diagnostics**:
In-memory inspection data produced by the workflow runtime for the most recent run, including scheduler decisions and scheduled node waves.
_Avoid_: public scheduler API, debug-only list

**Durable Node Completion**:
The committed state in which a node's processed result, materialized files,
artifact manifests, Task status, node status, and Provider Call status agree.
_Avoid_: returned function result, partially recorded success

**Workflow Orchestrator**:
A Modal-hosted coordinator that owns one workflow run, hosts the workflow runtime, records durable run state, and uses Modal lifecycle hooks to reconcile interrupted work.
_Avoid_: workflow node, runner

**Workflow Ledger**:
The physical per-run SQLite database in which a workflow orchestrator hosts
the shared Execution State Repository alongside workflow-specific artifact
records.
_Avoid_: separate workflow execution state machine, worker-owned database

**Workflow Artifact Store**:
The workflow-specific persistence for artifact manifests, files, node
inputs, and node outputs, colocated with shared execution tables in the
Workflow Ledger.
_Avoid_: execution repository, scientific publication

**Node Placement**:
The execution location for a workflow node, either inline in the workflow orchestrator or in a separate remote Modal function.
_Avoid_: runner location, execution site

**Durable Node Cache** [planned]:
Volume-backed intermediate checkpoint state that workload code may use when
Modal redelivers the same provider input or a Successor Execution Run schedules
still-missing work.
_Avoid_: temporary scratch, local cache

## Flagged ambiguities

- "artifact" can mean either inline app bytes or remote files. Resolved: an **Inline Byte Output** is a small app output before materialization; a **Workflow Artifact** is durable volume-backed state after materialization.
- "step" can mean either a semantic workflow operation or one callable remote function. Resolved: use **Workflow Node** for the semantic DAG unit and **App Function** for a Modal remote callable.
- "app node" can mean either a Modal deployment unit or a DAG vertex backed by that app. Resolved: use **App** for the deployment unit and **App-Backed Node** for the DAG vertex.
- "workflow entrypoint" can be confused with Modal's local entrypoint. Resolved: use **Workflow-Compatible App Function** for reusable remote app functions and **Local Entrypoint** for CLI wrappers.
- "parallelism" can mean ready workflow nodes, child app calls, tool pods, or CPU workers. Resolved: use **Workflow Node Parallelism** for scheduler waves, **Run-Level Task Budget** for coordinator-scoped child-work limits, **Child App Call** for submitted app functions, **App-Local Scheduler** for tool-owned queues, and **Worker Pool** for local thread or process pools.
- "dynamic workflow" can mean changing the DAG at runtime or changing only the task count. Resolved: first-version workflows use static DAGs with **Dynamic Task Fan-Out** only.
- "scheduler database" can mean either the common execution-state contract or one shared physical database. Resolved: the kernel governs the **Execution State Repository** contract, while each durable coordinator may persist it separately and **Workload Publications** remain authoritative for scientific completion.
- "job" can mean a user-facing service request or actual scheduled work. Resolved: a **Service Job** holds service metadata and refers one-way to an **Execution Run**; the execution kernel knows only the Run and its work.
- "Job state" can mean either persisted compute state or the API's user-facing summary. Resolved: compute state exists only in the Execution State Repository; the service exposes a **Job State Projection**.
- "Workflow Ledger" can mean either the physical per-run database or a workflow-specific implementation of execution state. Resolved: it names the physical database; shared execution tables come from the execution kernel, while the **Workflow Artifact Store** owns only workflow artifact records.
- "positions marked for RFdiffusion to generate scaffolds for" can mean every RFdiffusion output residue or only de novo contig residues. Resolved: use **LigandMPNN Redesign Set** for de novo output residues and exclude copied motif residues.

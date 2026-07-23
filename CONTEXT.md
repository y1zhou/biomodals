<!-- markdownlint-disable MD013 -->

# Biomodals

Biomodals runs bioinformatics tools as Modal apps and composes them into reusable computational workflows.

## Language

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

**MSA Benchmark App**:
An experimental, self-contained Biomodals app that produces comparable AlphaFold3 MSA-search measurements while remaining outside production prediction and workflow behavior. It may copy proven environment and image setup, but owns separate Modal resources and never imports the production AlphaFold3 app module. Profile preparation, storage scanning, and search benchmarking are separately invoked operations; later operations require a valid Published Database Profile. Local commands are plan-only by default and require an explicit `--submit` flag before making remote calls. The initial implementation is limited to small-BFD profile preparation, Volume scans, smoke validation, and the Phase 1 search matrix; four-database/per-sequence fanout is designed only after Phase 1 measurements select candidate layouts.
_Avoid_: temporary AlphaFold3 app, production AlphaFold3 app

**Benchmark Database Profile**:
A manifest-identified set of scientifically equivalent physical layouts of one genetic reference database used for controlled MSA-search comparisons.
_Avoid_: shard directory, test database, database copy

**Published Database Profile**:
A Benchmark Database Profile whose source identity, shard equivalence, and balance have passed validation and whose readiness manifest has been published.
_Avoid_: complete directory, prepared database

**Profile Preparation**:
The explicit pre-measurement operation that creates, validates, and publishes a Benchmark Database Profile.
_Avoid_: benchmark setup, lazy shard creation

**Shard Build Recipe**:
A versioned deterministic transformation from a reference-database monolith into the shard layout of a Benchmark Database Profile.
_Avoid_: ad hoc split, shard command

**Benchmark Run**:
A controlled MSA-search execution that reads a published Benchmark Database Profile without modifying it and produces performance and scientific evidence.
_Avoid_: profile preparation, production prediction

**Storage Scan**:
A Benchmark Run that reads one complete physical database layout without performing sequence search, establishing the storage-delivery envelope.
_Avoid_: MSA search, cold-cache benchmark

**Storage Reader Topology**:
The allocation of concurrent database read streams across benchmark containers during a Storage Scan.
_Avoid_: shard count, HMMER CPU layout

**Isolated Database Search**:
A Benchmark Run that queries exactly one genetic database through the pinned AlphaFold MSA implementation, excluding other databases, templates, and inference.
_Avoid_: full data pipeline, prediction benchmark

**Instrumented Search Adapter**:
A benchmark-only wrapper around the pinned AlphaFold database search that retains raw evidence and timings without changing search or merge behavior.
_Avoid_: AlphaFold fork, alternative search implementation

**Validated Search Result**:
A database-search result whose completion marker proves its scientific identity and the integrity of its declared artifacts.
_Avoid_: existing A3M file, benchmark cache hit

**Operational Search Baseline**:
An Isolated Database Search using the current unsharded AlphaFold search configuration, used to quantify change from existing behavior.
_Avoid_: scientific oracle, full production latency

**Monolithic Search Oracle**:
An unsharded Isolated Database Search with full-database statistical scaling fixed explicitly, used to evaluate sharded scientific evidence.
_Avoid_: operational baseline, byte-identical output oracle

**HMMER CPU Layout**:
The combination of tool threads per shard process and simultaneously active shard processes within an Isolated Database Search.
_Avoid_: Modal CPU allocation, total database shard count

**Screening Query**:
The pinned representative protein sequence evaluated across every initial Isolated Database Search configuration.
_Avoid_: example input, stress query

**Stress Query**:
The pinned protein sequence with high expected search and merge demand used to validate promoted search configurations.
_Avoid_: screening query, arbitrary long sequence

**Benchmark Block**:
One sequential pass through a defined set of search configurations in a controlled order, providing one measured sample per configuration.
_Avoid_: concurrent batch, benchmark case

**Promoted Search Layout**:
A sharded HMMER CPU Layout that passes screening gates and advances to stress-query or integrated-pipeline evaluation.
_Avoid_: fastest observation, presumed production default

**Scientific Promotion Gate**:
The evidence requirements an Isolated Database Search must satisfy against the Monolithic Search Oracle before its performance can justify further evaluation.
_Avoid_: byte-identity requirement, final-model score check

**Performance Promotion Gate**:
The latency and cost requirements a scientifically valid search layout must satisfy before advancing beyond isolated-database evaluation.
_Avoid_: fastest sample, scientific gate

**Benchmark Evidence Set**:
The durable raw search outputs, measurements, provenance, and completion state produced by one defined benchmark execution. At the campaign root, `plan.json` fixes the work, `results.parquet` indexes one row per sample, and `summary.md` reports comparisons and gates; large/raw artifacts remain in their sample directories and are referenced rather than copied.
_Avoid_: MSA cache, database profile, downloaded summary

**Resource Trace**:
A durable time series of container resource use and active search work aligned to the phase boundaries of a Benchmark Run.
_Avoid_: transient platform log, aggregate runtime only

**Benchmark Campaign**:
A finite, reproducible collection of Benchmark Runs performed to select an MSA Sharding Strategy. Non-sequence evidence such as storage scans and campaign summaries lives under `/benchmarks/{campaign_id}/`; it is never assigned a synthetic sequence hash. The immutable initial campaign ID is `small-bfd-phase1-v1`; changing its plan requires a new ID. This campaign permits one smoke invocation and, after it passes, one measured matrix invocation containing the agreed three blocks. Completion markers make accidental reruns submit no work.
_Avoid_: benchmark service, production workload

**MSA Sharding Strategy**:
A scientifically valid database layout and search-resource configuration selected for possible incorporation into the production AlphaFold3 app. Benchmarking may rank candidates but never migrates one automatically; production promotion is a separate, explicitly approved implementation step.
_Avoid_: fastest sample, benchmark harness

**MSA Search Subject**:
A unique biological sequence that requires database-generated MSA evidence and may be referenced by one or more input chains.
_Avoid_: chain identifier, duplicate homomer chain

**Raw Database MSA**:
A validated result of searching one MSA Search Subject against one reference database profile. It is independently complete and reusable before AlphaFold constructs combined unpaired or paired MSAs.
_Avoid_: combined unpaired MSA, paired MSA, benchmark summary

**Combined Unpaired MSA**:
The AlphaFold-ready unpaired alignment assembled from validated Raw Database MSAs in pinned upstream order, with duplicate aligned sequences removed after ignoring lowercase insertions. Protein order is UniRef90, small BFD, then MGnify; RNA order is RFam, RNAcentral, then NT-RNA.
_Avoid_: raw database MSA, simple FASTA concatenation

**Combined Paired MSA**:
The AlphaFold-ready paired protein alignment assembled from the UniProt Raw Database MSA without deduplication. RNA inputs have no paired MSA.
_Avoid_: combined unpaired MSA, RNA MSA

**Search Identity**:
A digest of the result-affecting inputs for one Raw Database MSA, stored beneath the full sequence hash. It includes the database-profile manifest, scientific search parameters, and pinned tool versions, but excludes operational benchmark settings such as CPU allocation and container layout.
_Avoid_: sequence hash, benchmark sample ID, resource configuration

**Canonical Search Result**:
The production-cache artifact published at a Search Identity root only after its search strategy has passed the scientific and performance gates. Benchmark runs never publish this artifact; they write only Benchmark Samples beneath `samples/`.
_Avoid_: arbitrary benchmark result, fastest unvalidated sample

**Benchmark Sample**:
One measured execution of a search case within a Benchmark Campaign, identified by a human-readable sample ID such as `screen-S3-block-01`. Multiple Benchmark Samples may share one Search Identity while measuring different operational layouts or repetitions. Query-derived evidence always lives beneath `/{sequence_hash_prefix}/{sequence_hash}/raw-msa/`.
_Avoid_: scientific cache identity, unique biological sequence

**Sequence Hash Prefix**:
The first two hexadecimal characters of the full sequence hash, used only to fan out directories. It is not a sequence identifier by itself.
_Avoid_: sequence hash, search identity

**Duplicate-Tail Difference**:
A sharded-search discrepancy attributable only to cross-shard duplicate hits and their effect at the truncated end of an MSA result.
_Avoid_: unexplained hit difference, scientific equivalence

**Benchmark CPU Floor**:
The minimum Modal CPU reserved throughout a Benchmark Run, balancing guaranteed compute availability against cost during low-utilization phases.
_Avoid_: HMMER CPU layout, CPU limit

**Search Wall Time**:
Elapsed time from starting the pinned database query through completion of its merged A3M result.
_Avoid_: sample wall time, remote call wall time

**Sample Wall Time**:
Elapsed time from benchmark-function entry through durable publication of that sample's evidence.
_Avoid_: search wall time, remote call wall time

**Remote Call Wall Time**:
Elapsed time from submitting a benchmark function call through observing its completion, including platform scheduling and container startup.
_Avoid_: search wall time, sample wall time

**Volume-Resident Reference Data**:
Genetic reference-database payloads that remain on persistent Modal Volumes throughout preparation and measurement rather than being fully staged on container-local storage.
_Avoid_: local database staging, SSD-cached database

**App Function**:
A callable Modal remote function exposed by a Biomodals app or another Modal app and invoked by a workflow node.
_Avoid_: workflow node

**Child App Call**:
A Modal app function call submitted inside an app-backed workflow node or an app-local entrypoint as part of a larger semantic operation.
_Avoid_: workflow node, scheduler node

**Local Entrypoint**:
A CLI-facing Modal entrypoint that parses local user inputs, submits app functions, downloads or reports outputs, and returns no workflow contract.
_Avoid_: workflow entrypoint

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
A tool-specific queue, worker pool, pod pool, or fan-out loop inside an app function or local entrypoint.
_Avoid_: workflow runtime, DAG scheduler

**Run-Level Task Budget**:
A shared concurrency budget for child app calls and local workers participating in one user-submitted run.
_Avoid_: max_parallel, workflow node parallelism

**Workflow Runtime**:
The reusable library that validates a workflow DAG, schedules workflow nodes, tracks durable run state, and materializes workflow artifacts.
_Avoid_: engine

**Runtime Diagnostics**:
In-memory inspection data produced by the workflow runtime for the most recent run, including scheduler decisions and scheduled node waves.
_Avoid_: public scheduler API, debug-only list

**Stale Node Attempt** [planned]:
A durable node attempt from a previous failed orchestrator session. On next session the runtime should either rerun the node as a fresh attempt or recover from the previous run's recorded state and remote-call identity.
_Avoid_: active node, pending node

**Durable Node Completion**:
The committed state in which a node's processed result, materialized files, artifact manifests, attempt status, node status, and remote-call status agree.
_Avoid_: returned function result, partially recorded success

**Workflow Orchestrator**:
A Modal-hosted coordinator that owns one workflow run, hosts the workflow runtime, records durable run state, and uses Modal lifecycle hooks to reconcile interrupted work.
_Avoid_: workflow node, runner

**Workflow Ledger**:
A per-run SQLite database written by the workflow orchestrator that records run, node, attempt, remote-call, fan-out task, and artifact state for recovery and manual debugging.
_Avoid_: scattered JSON state files, worker-owned database

**Node Placement**:
The execution location for a workflow node, either inline in the workflow orchestrator or in a separate remote Modal function.
_Avoid_: runner location, execution site

**Node Execution Policy**:
The restart and recovery contract for an incomplete workflow node when Modal interrupts or retries the node.
_Avoid_: runner tag, retry hint

**Durable Node Cache** [planned]:
Volume-backed intermediate checkpoint state that lets a long-running workflow node with `RESUME` execution policy restore progress after interruption or restart.
_Avoid_: temporary scratch, local cache

## Flagged ambiguities

- "artifact" can mean either inline app bytes or remote files. Resolved: an **Inline Byte Output** is a small app output before materialization; a **Workflow Artifact** is durable volume-backed state after materialization.
- "step" can mean either a semantic workflow operation or one callable remote function. Resolved: use **Workflow Node** for the semantic DAG unit and **App Function** for a Modal remote callable.
- "app node" can mean either a Modal deployment unit or a DAG vertex backed by that app. Resolved: use **App** for the deployment unit and **App-Backed Node** for the DAG vertex.
- "workflow entrypoint" can be confused with Modal's local entrypoint. Resolved: use **Workflow-Compatible App Function** for reusable remote app functions and **Local Entrypoint** for CLI wrappers.
- "parallelism" can mean ready workflow nodes, child app calls, tool pods, or CPU workers. Resolved: use **Workflow Node Parallelism** for scheduler waves, **Run-Level Task Budget** for shared child-work limits, **Child App Call** for submitted app functions, **App-Local Scheduler** for tool-owned queues, and **Worker Pool** for local thread or process pools.
- "dynamic workflow" can mean changing the DAG at runtime or changing only the task count. Resolved: first-version workflows use static DAGs with **Dynamic Task Fan-Out** only.
- "positions marked for RFdiffusion to generate scaffolds for" can mean every RFdiffusion output residue or only de novo contig residues. Resolved: use **LigandMPNN Redesign Set** for de novo output residues and exclude copied motif residues.

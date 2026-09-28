# Protein sequence optimization

Status: implemented and under review; native model/GPU validation remains a rollout gate. This document records the accepted product decisions and their implementation. The user-supplied research proposal is local background, not a repository dependency or an additional source of product requirements.

## Requested outcomes

1. A reusable Biomodals TabPFN app that fits a labeled table and predicts another table with the same features.
2. A reusable Biomodals ridge app that accepts a `mutations,label` table, constructs mutation one-hot features, and scores all compatible combinations through a user-selected mutation count. Full parental sequences are not required for this model.
3. A protein-optimization service and website that turn experimental measurements into suggested combinations or new variants, including but not limited to antibodies.

The current branch is `feat/protein-optimization`. Research and this interview do not authorize model downloads, paid inference, deployment, or server changes.

## Accepted product decisions — updated September 28, 2026

- Include both Combination and Exploration in the first release, delivering the combination path as the first implementation milestone. Each website submission selects exactly one of the two modes; there is no Both mode. Preselect Combination, retain uploaded data when switching modes, and never start computation automatically. Explain their differences in gray descriptive website text: Combination recombines experimentally supported substitutions using additive ridge; Exploration proposes previously unmeasured substitutions and scores complete variants using sequence features and TabPFN. Neither prediction is experimental confirmation.
- Support arbitrary proteins and mutations across user-identified chains of a multimer, not only VH/VL or VHH. Each run uses fixed original-residue identities at measured sites; Exploration additionally uses full parental chains. Do not infer chain roles or apply antibody-specific trimming, imputation, or numbering rules.
- Start by uploading the measurement table. Mutation tokens use `{chain_id}:{original_aa}{position}{mutated_aa}`, with commas separating multiple mutations in a row. Both modes reject conflicting original-residue claims at a site. Only Exploration requests parental FASTA and validates every original residue and position against its chain; missing chains or out-of-range positions must not reach Exploration fitting. Combination uses observed sites only, without fabricating residues for unobserved positions. Positions are one-based raw sequence coordinates.
- Combination features cover only exact substitutions present in usable measured input, never invariant positions or unobserved alternatives at a tested position. Encode a variant as a binary vector over those substitutions; distinct alternative residues at one position remain distinct features and cannot coexist in one candidate.
- Accept an arbitrary numeric experimental label with explicit higher/lower-is-better direction. All uploaded labels are treated as already normalized; users must correct batch/plate effects before upload. For example, use `log10(mutant KD) - log10(parent KD)` with the parent control on each plate and select lower-is-better. Do not presume KD, require affinity units, or automatically transform labels. Use the supplied numerical scale.
- Return all compatible novel combinations through the requested mutation count and reject oversized exhaustive requests before compute rather than silently sampling or truncating. Exclude the unchanged parent and variants already represented in the uploaded measurement snapshot from the candidate CSV/table. The input measurements remain training/validation data, not candidate-result rows.
- The generic TabPFN app is regression-only initially, with one numeric target and numerical/categorical features selected explicitly; identifiers are not automatically features.
- Fit and infer in the same invocation. No saved fitted-model reuse, prediction-only follow-up using an earlier fit, model registry, or reusable fitted-context cache in this release. Standard Job inputs/results and pinned foundation-weight provisioning remain distinct from fitted-model storage. Runtime speed still needs measurement, especially embedding extraction and large candidate pools.
- Each successive wet-lab round is a fresh immutable Job. Users manually assemble and upload the next measurement snapshot; no campaign CRUD, automatic historical-measurement merge, or model update operation.
- Start with the latest stable model/software releases at implementation time and pin exact versions/checkpoint identities like other apps; do not preserve the paper's older baseline solely for reproduction. Reconfirm published releases, checkpoint access, and compatible dependencies when implementing. Use ESMC600M specifically, with the latest stable compatible release and an exact checkpoint revision/digest.
- Target hundreds to a few thousand unique measured variants initially. Use 5,000 exploration candidates and 1,000,000 combinations as initial budget defaults, configurable within operator resource ceilings that must be verified before release. An exhaustive request above its selected combination budget is rejected, never sampled. Explain that additive combination scoring is computationally cheap and supports larger candidate counts, while acknowledging enumeration/output size; defer exploration cost/runtime guidance until benchmarks exist.
- Present novel candidates sorted by predicted label in the requested improvement direction, with filtering and manual selection. No measured-reference rows, automatic diversity selector, or experiment quotas in this release.
- The user-facing scientific output is the scored-candidate CSV, also displayed as a bounded sortable webpage table. Permit downloading all candidates or a manually selected subset using the same schema; selection survives pagination, sorting, and filtering through stable candidate identities. Do not add separate FASTA, measurement-return template, coefficient export, validation-report download, or fitted-model bundle. Internal request/provenance and the compact validation summary are distinct from downloadable scientific outputs.

### Measurement input and parental chains

The input is CSV with required `mutations,label` columns and optional `id`. A cell containing multiple comma-separated substitutions is quoted using ordinary CSV rules:

```csv
id,mutations,label
variant_1,A:Y52F,-0.2
variant_2,"A:Y52F,B:S30A",-0.4
```

No parent row is needed for normalization: it is the user's responsibility before upload. An explicitly supplied empty mutation cell still denotes a measured unchanged parent and is accepted on the same supplied scale. Its absence is not a missing-input warning.

Repeated rows describing the same canonical variant are experimental replicates. Aggregate their finite labels by arithmetic mean on the supplied scale, retain raw observations, counts and spread, and keep their shared variant identity together in validation splits. Flag malformed, missing, nonfinite, and censored labels such as `>1000` for explicit correction or removal; do not silently discard rows or convert bounds to exact values.

Only in Exploration, show the Parental chains box beneath Design settings and accept a multi-record parental FASTA with matching chain-ID headers and per-chain previews. Require every referenced chain and permit additional explicitly named chains for exploration of previously unmutated chains. Unchanged partners are optional; do not invent missing chains or claim multimer structure/stoichiometry modeling. Mode switches retain drafts locally, but the website omits FASTA from Combination requests. The standalone ridge/API optional FASTA can still validate sites and supply full-chain outputs when explicitly provided; it never changes ridge features or scores. Table-only Combination CSVs have no sequence columns and expose an empty `chain_columns` mapping.

### Design space and exploration representation

Exploration defaults to positions present in the measurement table. Users can explicitly add chain positions/ranges and exclusions; new amino-acid alternatives may be proposed at those positions. Combination remains restricted to exact experimentally supported substitutions. No antibody-specific CDR/framework restriction is inferred.

Exploration's allowed amino acids are configured independently per chain position. The default set is `ADEFGHIKLNPQRSTVWY`: the 18 standard amino acids excluding C and M, with explicit user overrides. Every proposed replacement in Exploration must satisfy its position's allowed set, including a previously measured substitution used as background. A proposed substitution must differ from the parental residue; leaving a position unchanged does not require that parental residue to be in the allowed replacement set. The restriction is not a global ban on cysteine/methionine in inputs or final sequences. It does not apply to Combination, which can recombine measured C/M substitutions. Neither mode drops training measurements merely because their residues would be excluded from Exploration proposals.

Default maximum distance is two substitutions from the fixed Optimization Parent, summed across chains. Exploration defaults to at most one unmeasured substitution per candidate, with an explicit control to increase that count. The exploration evaluation budget is separate from exhaustive combination-space size: deterministically sample an oversized exploration space with a recorded seed, while an oversized combination request is rejected rather than sampled. The initial configurable budgets are recorded above; final operator ceilings require resource verification.

For oversized Exploration spaces, divide evaluations approximately evenly across feasible mutation-count strata, reassign unused capacity from exhausted strata, and sample reproducibly without replacement within each stratum. Every proposed Exploration candidate contains at least one exact substitution absent from the usable measured catalog and obeys the maximum new-substitution count. Allocation is over candidates to evaluate, not experimental panel quotas; no measured-best seed/search machinery is required.

For exploration features, independently encode supplied chains with frozen ESMC600M, mean-pool residue representations, and concatenate in a stable recorded chain order. This retains chain identity but does not model joint interchain attention or the multimer's structure. Any needed dimensionality reduction is fitted on training data only, separately within validation folds. Exact package/checkpoint pins, resource bounds, and compression settings require verification against the selected latest stable release, not the older paper environment's limits.

### Validation and final fit

Validation uses up to five reproducible support-aware splits. Display a compact summary above the CSV table, including evaluated variant counts, error/ranking metrics when defined, and limitations. If meaningful validation cannot be constructed, allow otherwise valid predictions with a clear warning; do not silently fall back to random row splits or add a separate report download.

Combination validation holds out complete measured multi-mutant variants only when every exact constituent substitution retains usable support in the remaining training partition. Preserve measured parent/singles as training anchors. Other combinations may provide support, but occurrence alone does not establish independently identifiable effects; report co-occurrence confounding. Check support for the complete fold. A singles-only dataset has no empirical combination validation, and validation on doubles does not establish higher-order accuracy.

Exploration validation holds out an exact substitution when another amino-acid alternative at the same chain position remains represented in training. Remove every variant containing the held-out substitution from training, including multi-mutants and all replicates. Track extra unsupported substitutions in held-out variants rather than mislabel them as a clean one-new-substitution test. This evaluates new alternatives at known positions, not entirely unmeasured positions or chains.

Keep canonical variants/replicates together; construct splits without using outcome labels; recompute support and learned preprocessing inside training folds; and keep evaluation labels out of regularization selection. Undefined metrics and insufficient evaluation coverage remain explicit.

**Final scoring always refits on all validated input data**, after the agreed replicate aggregation. Validation models are temporary and never become the final candidate scorer merely because one split performed well. Refit learned preprocessing, including any PCA, on the complete final training set as well. Held-out measurements return to final training; candidate mutation-count bounds constrain proposed candidates, not which otherwise valid measured variants are allowed into the final fit. Frozen foundation-model weights are not fine-tuned on these measurements.

## Evidence and integration boundaries

The [model API audit](../research/protein-optimization-model-apis.md) records official sources, version drift, native fit/predict/persistence behavior, and unresolved installation/performance checks. The supplied study's main 0.767 versus 0.663 comparison uses ESMC features for both TabPFN and ridge under random folds; it does not compare TabPFN to mutation one-hot ridge or establish prospective antibody performance. See [the paper, Table 1](https://arxiv.org/html/2606.31126v2).

No TabPFN, mutation-ridge, or ESMC extractor exists under `src/biomodals` at the investigated backend HEAD `2a9b024`. Exploration therefore requires a new sequence-feature stage; the tabular estimator alone cannot consume raw antibodies or infer meanings for unseen mutation columns.

Relevant existing seams:

- Scientific apps follow [app development standards](../../.agents/skills/biomodals-app-development/references/quick-app.md). Multi-file apps expose `app.py`; they retain independent CLI and workflow-compatible functions. Candidate enumeration belongs with the ridge app where it is part of that app's standalone contract.
- A composed optimization workflow can follow [workflow development standards](../../.agents/skills/biomodals-workflow-development/references/workflow-development.md), using the existing execution kernel, immutable requests, pinned deployments, artifact publication, and total/GPU admission limits. No new scheduler or `abaffinity` CLI is needed.
- [ToolAdapter](../../src/biomodals/service/tool_runtime.py) owns only request staging, pending cleanup, and result preparation. [Service assembly](../../src/biomodals/service/api.py) and [Tool definitions](../../src/biomodals/service/tools.py) register each Tool explicitly. Existing ownership, idempotency, cancellation, download, result recovery, and deletion apply to a new Job-producing Tool.
- [Table archive construction](../../src/biomodals/service/table_archive.py) is a pattern, not the chosen output contract: the user has selected CSV-only delivery rather than a ZIP of scientific files. Its current browser reader assumes a ZIP member `selection.csv` and a 32 MiB member limit. Use shared artifact/cache/download mechanisms without forcing a new tool's CSV into that archive-specific reader. Exhaustive outputs must not cause complete large-file rereads per page.
- [Humanization table presentation](../../src/biomodals/service/selection.py) is not a generic prediction table: it assumes parental grouping, evaluation-completion flags, and nativeness ranges. Reuse bounded paging/sorting patterns, not these scientific fields or ranks.
- Frontend `src/tools.ts`, `src/App.tsx`, and `src/pages/JobDetailPage.tsx` provide explicit catalog/routes and a lazy tool-specific result boundary. `src/lib/csv.ts` supplies CSV framing, not arbitrary column mapping. The existing shared sequence dialog can inspect a candidate against its exact parent. Optimization should have a dedicated form/result panel rather than another mode flag in `HumanizationResults`.

## Implementation architecture

The product responsibilities, module layout and execution decomposition are approved. Exact interfaces below remain implementation work until verified.

| Surface | Scientific responsibility | Outputs |
| --- | --- | --- |
| TabPFN app | Validate declared features and numeric target; fit native estimator; score an inference table in the same invocation | Predictions, ordered feature/runtime identity, and diagnostics; no reloadable fitted bundle |
| Mutation ridge app | Validate chain-relative substitutions; fit additive ridge; enumerate compatible known substitutions in batches | Complete scored-combination CSV; internal fitting/support evidence as needed |
| Protein optimization workflow | Validate parental chains and measurement semantics; run the selected mode and evaluate | Novel scored-candidate CSV and bounded presentation/validation metadata |
| Service and frontend | Review inputs/settings, admit an immutable Job, present bounded results and downloads | Existing Job lifecycle and owner-scoped scientific artifacts |

Keep generic table fitting separate from protein design policies. If ESMC is only needed here, start with a workflow-owned GPU feature stage; make it a separate app only if independent CLI reuse is required. Fit/checkpoint dependencies belong in their compute images, not the ordinary API environment. Avoid introducing the research proposal's twelve-component package, separate campaign database, generalized model registry, or independent CLI. Any within-run reuse of identical chain embeddings is an execution optimization, not a user-facing fitted-model cache; cross-run embedding caching is not yet approved.

## Scientific and operational constraints to preserve

- Mutation coordinates use a chain plus one-based raw sequence position and checked reference/alternative residue, such as `H:Y52F`; antibody numbering, if displayed for antibodies, is a separate annotation. Do not trim or impute experimentally measured sequences silently.
- Ridge's additive coefficients do not estimate interactions. Mutations observed only together may have confounded effects; regularization can produce coefficients without identifying separate biological contributions.
- An unobserved substitution cannot silently become the parental all-zero vector. Combination vocabulary comes from usable experimental labels, not model predictions.
- At most one alternative can occupy a position. If position `i` has `a_i` permitted alternatives, the exact count through `n` substitutions is the sum of coefficients through degree `n` in the polynomial `product_i(1 + a_i t)`, including the parent at degree zero. Count before allocating candidates. For 100 positions with one alternative each, this is 166,751 through three mutations and 79,375,496 through five.
- The standalone app's promise to score all combinations is distinct from selecting a small experimental panel. Do not silently substitute sampling or top-k output for an exhaustive request.
- Keep labels and direction explicit. KD, EC50, enrichment, and arbitrary scores are not interchangeable; bounds such as `>1000` are not exact measurements. Repeated observations of one sequence must not leak across evaluation partitions.
- Fit PCA/scalers and select hyperparameters within training partitions. Random row splits alone do not validate transfer to new substitutions or positions. Report unavailable validation regimes rather than fabricate evidence.
- Freeze checkpoint, feature schema/order, preprocessing, precision, seed, and data identity. The TabPFN default inspected in the research differs from the proposal's V3 baseline. Pin deliberately and test fit/predict and prediction batching before promising limits.
- Do not add native fitted-model persistence or accept uploaded model pickles or arbitrary checkpoint paths. Provisioning pinned foundation weights is still necessary and is separate from storing a user's fitted model.
- General multichain inputs affect exploration feature width: concatenated per-chain embeddings grow with chain count. A stable chain order, a bounded supported chain/feature size, and a declared training-only compression policy need resolution. Separate-chain embeddings do not supply joint structural or interchain attention, and arbitrary partner-chain uploads must not be described as structural conditioning.
- Score candidates in bounded batches and retain authoritative downloadable results. Candidate count, training rows, feature width, artifact size, and browser page size are distinct resource limits.

## Website controls and candidate CSV

Main controls are mode, improvement direction, maximum total mutations, and candidate budget. Advanced exposes Exploration's editable positions, per-position allowed amino acids, maximum new substitutions, and random seed. Keep raw ridge regularization and TabPFN technical settings out of the ordinary website form; maintain reproducible app-level controls and document final fitting defaults in the implementation plan.

The compact candidate schema is `id`, `mutations`, `predicted_label`, `n_mutations`, `n_new_mutations`, `warnings`, and one full-sequence column per supplied chain. Because every candidate is untested relative to the supplied measurement snapshot, `measured_label` and `is_measured` are unnecessary. This does not change final fitting on all validated input data. Deduplicate/exclude by canonical variant identity (fixed chain mapping and substitutions/sequences), not uploaded display IDs or mutation-token order. Previously tested variants absent from the supplied snapshot cannot be identified automatically across manually assembled rounds.

## Implementation milestones

Each verified functional milestone receives a self-contained commit. Coordinate frontend contracts through exact offline OpenAPI exports and deterministic fixtures before frontend integration; parallelize independent UI work only after overall approval. No production deployment, scientific submission, push, or PR is part of this design turn.

### 1. Measurement contracts and input review

Put standalone mutation parsing and combination-domain contracts inside `app/design/mutation_ridge/`; let the workflow reuse them. Keep workflow-only settings and exploration policy beside `workflow/protein_optimization/`, not in global helper/schema modules.

Implement Polars CSV parsing, row-associated diagnostics, canonical chain/substitution identities, parental FASTA matching, replicate aggregation, and candidate-space counts. Use only the standard 20 amino acids in sequences/substitutions for this first implementation; normalize whitespace/case in sequences without trimming biological residues. Chain IDs remain case-sensitive, unambiguous tokens with no whitespace, colon, or comma; never use them directly as storage paths. Reject duplicate chain records, conflicting substitutions at one site, and original-residue mismatches. Keep uploaded display IDs separate from internal candidate identity.

Input review is local and non-scientific: show required chains, mismatches and invalid labels before explicit Job submission. Sequence/table/design edits invalidate stale review and submission intent, without dropping the user's rows. Resource admission also checks bytes, row counts, chain lengths/count, candidate count and estimated output size; do not silently truncate protein sequences, training rows or candidates.

Tests cover quoted mutation lists, arbitrary multichain parents, ordering-equivalent variants, duplicates/replicates, reference mismatches, empty parent observations, invalid labels, safe IDs, and immutable original inputs.

### 2. Standalone mutation-ridge app and Combination path

Expose `app/design/mutation_ridge/app.py` through existing app discovery/CLI and a workflow-compatible operation. Fit an intercept-bearing ridge regressor on unscaled binary columns for measured exact substitutions. Use the approved app-level `alpha=1.0` rather than add automatic hyperparameter search; record the parameter and use the same policy in validation. A later tuning option must remain training-fold-local, never use evaluation labels. This default is not a claim of optimal regularization. The native implementation uses scikit-learn 1.9.1, sparse CSR features and its intercept-aware `lsqr` solver with tolerance `1e-10`.

Count compatible novel combinations before admission, enumerate in bounded batches, exclude measured/parental variants, and score using the fitted intercept and active substitution coefficients rather than constructing a dense candidate-by-feature matrix. Stream complete candidate rows to CSV. Perform support-aware held-out-combination validation when feasible, then refit on the complete aggregated measurements for production scoring.

Tests recover a synthetic additive landscape, expose a synthetic interaction limitation, preserve support across whole holdout folds, flag confounding, report singles-only validation unavailability, verify count/exclusion boundaries, and reconstruct every emitted sequence from its mutation list. Include a CLI example and discovery/help checks.

### 3. Standalone TabPFN app and exact dependencies

Expose `app/misc/tabpfn/app.py` with labeled training and inference tables, a target column, optional identifier column, declared feature columns and app-level model controls. Parse with Polars and convert only at the native estimator boundary. Match inference columns against the recorded feature names/types/order; never train on IDs by accident or interpret an unknown column as a feature silently. Native supported missing features are separate from invalid/missing targets. Return predictions as CSV from the same fit-and-infer invocation.

Resolve the latest stable TabPFN release/model and ESM release compatible with ESMC600M; pin package versions, checkpoint revisions/digests and runtime dependencies. Provision foundation weights explicitly and verify readiness before work; disable unnecessary telemetry and use local inference in the approved compute environment. Keep heavyweight dependencies in their task images, not the API environment. Do not implement fitted-model save/load or accept user model/checkpoint uploads.

Declared categorical features retain their exact strings and missing values. The native categorical-cardinality limit covers the app's 10,000-row training ceiling, preventing numeric-looking category labels from silently becoming quantities. Verify inferred categorical types after every fit and retain all inferred feature modalities alongside widths in internal publication metadata, ordered by validation fold and then final fit. This policy is part of the TabPFN runtime identity.

Offline boundary tests cover schema reorder/mismatch, categorical handling, nonfinite values and output identity. Native package/image checks, checkpoint access, deterministic settings, and batched-versus-whole prediction tolerance remain separately reported gates, not claims made by mocked tests.

The standalone implementation pins TabPFN 9.0.0 with the immutable V3.5 checkpoint recorded in the model API audit, Torch 2.11.0/CUDA 13.0 and scikit-learn 1.9.1. It stages content-bound CSV inputs in its own Volume, verifies them again before native consumption, and returns a content-bound prediction CSV. Workflow callers may use equivalently validated Parquet features. The default native ensemble has eight estimators and inference batches of 256 rows. The pinned native validator requires at least two training observations. Preparation is a tracked CPU stage with writable foundation storage; inference mounts it read-only and disables implicit network downloads. No fitted estimator is persisted. These are implemented offline boundaries, not evidence that the image or GPU inference has been exercised.

### 4. Exploration and protein-optimization workflow

Implement `workflow/protein_optimization/workflow.py`, composing both apps but executing only the selected mode. Keep ESMC600M extraction workflow-owned initially. The workflow owns one existing execution-kernel Run; included app calls do not spawn nested coordinators. Reuse global total/GPU admission controls and pinned containing-deployment operation names.

Construct the allowed position/residue space, enforce at least one new substitution and the configured upper bounds, and allocate the finite evaluation budget across mutation counts. Use combinatorial counting and reproducible sampling without enumerating an enormous rejected pool. Within a run, encode each unique full chain once, exclude special/padding tokens from float32 pooling, and concatenate in recorded chain order. Reuse these frozen features for validation/final fitting without saving a reusable fitted model. Do not add a cross-run embedding-cache service.

Fit any required feature compression only on the relevant training partition. Derive supported feature width from the pinned estimator; do not carry forward older-model limits or bypass native guards. Run same-position alternative-substitution validation, then refit preprocessing and TabPFN on all validated measurements and score novel candidates. The combination mode must not initialize an encoder or schedule GPU fitting.

Tests cover independent multichain reconstruction, C/M mask scope and overrides, unchanged parental C/M, novelty classification, balanced sampling/exhausted strata, exact budget/uniqueness, held-out-substitution leakage, final all-data refitting, pooling masks and batching invariance. Native embedding/inference checks remain distinct from offline tests.

The native feature adapter now pins `esm==3.4.1.post1`, the audited ESMC600M revision and both asset digests. It rejects chains longer than 2,046 residues, pools attended standard residue tokens in float32, and concatenates 1,152-wide chain blocks in sorted case-sensitive chain-ID order. Identical full chains are encoded once within the current invocation, with length-sorted batches of four. Offline tensor-double tests exercise the production pooling expression and padding exclusion; they do not establish numerical invariance of the real GPU encoder.

Exploration holdouts select up to five exact substitutions without labels, remove every affected variant, and require a different measured alternative at that site plus at least two unique remaining training variants. A multi-mutant may appear in more than one substitution test; average its held-out predictions before computing metrics and count it once. Extra unsupported substitutions and insufficient coverage remain explicit warnings. The generic TabPFN fitting boundary accepts these explicit row-index folds and optional numeric-only PCA, fits each transform on that fold's training rows, then creates fresh preprocessing and a new estimator for the final all-data fit. No fold or final fitted object is serialized.

The composed workflow now uses `fit_score_combinations → publish` for Combination and `prepare_models → extract_features → fit_score_exploration → publish` for Exploration. Included app calls remain in that one execution Run. Exploration projects concatenated numeric features to at most 1,024 components, further bounded by training-row count minus one, using seeded randomized PCA separately for each fold and final fitting. This is an explicit first implementation policy, not a benchmark-optimized setting. Resolved fitted widths enter internal provenance. Empty Exploration spaces publish a header-only CSV with a skipped-compute warning, without downloading weights or admitting GPU calls.

Native app outputs are verified through mounted external-Volume checks and consolidated into one terminal workflow-volume CSV with an internal content-bound manifest. That terminal publication contains the full user-facing result, so coordinator reopen does not re-run successful science. Offline graph tests use real parsing, sampling, PCA, ridge, publication and the shared coordinator, but deterministic replacements for ESMC/TabPFN inference. Scientific image installation and live GPU smoke tests remain outstanding.

### 5. Service integration and CSV result delivery

Add `service/protein_optimization/` with typed options/input review/submission, owner-scoped retained inputs, bounded candidate queries and full/selected CSV download. Register one Tool and mode-appropriate semantic stages. Reuse authentication, CSRF, exact-intent replay, pinned deployment, Job limits, cancellation, logs, result-preparation retry and deletion; no campaign tables or second lifecycle.

Keep the authoritative scientific result as CSV. Reuse shared artifact publication/cache/streaming mechanisms; a ZIP wrapper is not required. Verify one-time parsing plus a disposable query projection if needed for large tables, so paging/sorting does not repeatedly parse the full CSV or retain unbounded Python row objects. Such a projection is only local result-delivery state and follows existing cleanup/access rules, not a fitted-model cache or extra user download. Select its simplest bounded implementation against the agreed million-row scale before freezing service limits.

Test foreign/deleted access, lost-response replay, selected-row ownership, stable sort/page/filter behavior, late responses, cleanup, CSV text safety, and exact preservation of numeric predictions. Publish exact offline OpenAPI plus realistic fixture settings/limits to the frontend agent.

The implemented service registers `protein_optimization` and retains the existing shared Job lifecycle. Submission includes the original review fields plus `review_digest` and optional display name; exact-key replay precedes validation or deployment I/O. An invalid design is a known `422 invalid_design`; changed review is `409 review_changed`. Retained original inputs are owner-scoped and require a new explicit review for rerun. Combination and Exploration use only their actual plan stages in the shared timeline.

Result preparation verifies the remote CSV against its manifest and admitted design, parses it once in streaming Polars batches, and creates a disposable SQLite projection in the existing private Result cache. Only integer-generated SQL column identifiers are interpolated; filter values and candidate IDs are bound parameters. `chain_columns` maps chain IDs to CSV column names. Candidate pages contain at most 200 rows, with scalar sorts on ID, mutations, predicted label, total mutations or new mutations; filters are a case-sensitive mutation substring and exact mutation counts. Removing the explicit sort restores original scientific order. Sequence strings never become filesystem or SQL identifiers.

The authoritative workflow CSV remains unchanged. The service's full and selected download presentation prefixes formula-like text (including after leading whitespace/control characters) with an apostrophe, without changing numeric fields or raw query values. Full CSV uses the shared Job download. A CSRF-protected selected-download preparation validates all IDs and returns a five-minute same-origin ticket; authenticated GET checks owner, Job, result digest, expiry and the current ticket before streaming from the projection. One ticket per Job replaces older unused tickets; no candidate CSV is duplicated for selection. Tickets expire on service restart, cache clearing or deletion. Active responses hold the existing Result lease and finish before cleanup. The projection and small selection intent participate in cache byte accounting and deletion, not scientific model persistence.

A local synthetic million-row delivery check produced a 241,036,828-byte CSV and 339,513,344-byte index in 2.57 seconds. Fifty-row initial, deep-offset, score-sorted and substring-filtered queries took approximately 1.4, 30.9, 1.7 and 56.2 milliseconds, returning about 18 KiB each. Process peak RSS was about 1 GiB including synthetic source generation. These are single-machine, single-run observations, not service SLOs or neural-model benchmarks; the temporary harness and data are not committed.

### 6. Frontend and end-to-end verification

Add one Protein sequence optimization catalog entry and a dedicated input/result flow. Upload measurements first, choose one mode, show mode-specific gray guidance, and require explicit submission after review. Combination can review and submit the table alone. Exploration displays discovered chains, requests parental FASTA beneath Design settings, and has an editable per-position residue table initialized to the accepted 18-residue set. Use authoritative API bounds/defaults rather than duplicate independent frontend ceilings. Review contract version 2 advertises these mode-specific requirements; deploy the updated containing workflow and restart the matching API before using the updated frontend.

Display the compact validation summary and bounded novel-candidate table on shared Job detail. Support filtering, numeric sorting, full-sequence inspection appropriate for arbitrary proteins, persistent-across-pages manual selection, and full/selected CSV downloads. Do not force non-antibody proteins through antibody numbering/germline services. Reuse generic UI controls and shared Job actions rather than extending the humanization result component with another scientific mode.

Use deterministic offline fixtures for both modes and arbitrary chains. Browser tests exercise stale input review, mode changes, per-position defaults/overrides, explicit submission/replay, validation warnings, no measured candidates in results, pagination/selection, CSV downloads, auth, cancellation and deletion. Run backend tests/static checks/discovery smoke tests and frontend unit/lint/build/schema/browser checks; run `prek` on changed backend files before each milestone commit.

### 7. Review, benchmarks and rollout handoff

Review the completed implementation against this specification and report any deferred checks. Benchmark combination enumeration/output/querying and, only with an approved compute budget, ESMC600M plus TabPFN cost/memory/time. Do not publish Exploration cost guidance before those measurements. Freeze operational limits from evidence, surfacing any resulting material reduction in agreed scope before release.

The website requires deployment of the containing protein-optimization workflow, an exact service pin, updated API registration and the matching frontend. Separate app deployments are only needed for their standalone deployed CLI use. Provide the user with commands and a bounded test plan; do not deploy or launch paid scientific jobs without authorization.

## Approval boundary

The user approved the consolidated implementation plan, including module names, ridge fitting default, generic input validity details and milestone mechanics. Exact dependency pins, feature compression details and resource ceilings remain verification tasks, not silent permission to change product behavior. Deployments, paid runs and pushes require separate authorization.

The accepted domain terms live in `CONTEXT.md`; implementation details remain here. No ADR is needed yet for the straightforward first-release omissions.

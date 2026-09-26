# Protein optimization: model API evidence

Checked September 24, 2026. Research only; no model installation, weight download, GPU inference, deployment, or product decision was performed. The user-supplied [optimization proposal](protein-sequence-optimization.md) remains unchanged. This note checks the model integration boundary, not the paper's reported predictive performance.

## Version choice must be explicit

The proposal's `tabpfn==8.0.3` is a real package release, separate from the TabPFN-3 model generation. Its package dependencies include PyTorch, scikit-learn, pandas, joblib, and `tabpfn-common-utils[telemetry-interactive]`; Python 3.9 or newer is declared. The official main branch now declares package `9.0.0`, Python 3.10 or newer, and a changed dependency set. These are source observations, not an installation compatibility test. [Version 8.0.3 metadata](https://raw.githubusercontent.com/PriorLabs/TabPFN/v8.0.3/pyproject.toml), [current metadata](https://raw.githubusercontent.com/PriorLabs/TabPFN/main/pyproject.toml).

The current default model is TabPFN-3.5, not TabPFN-3. Prior Labs exposes `TabPFNRegressor.create_default_for_version(ModelVersion.V3)` for the older generation. GPU inference is recommended; current documented CPU safeguards allow up to 5,000 rows for V3 and V3.5. A bare constructor is therefore inappropriate for a frozen scientific baseline. Headless checkpoint access requires accepted terms and a `TABPFN_TOKEN`, or an appropriately pre-provisioned local checkpoint. Local fit/predict is distinct from the hosted client. [Official README](https://github.com/PriorLabs/TabPFN).

**Product decision, September 26:** use the latest stable release rather than preserve the paper's V3 baseline; see the [accepted specification](../specs/protein-sequence-optimization.md). Freeze package, checkpoint revision/hash, ensemble count, preprocessing, precision, and seed together after compatibility checks. Release choice alone does not establish protein-optimization performance. Earlier-version details below are evidence about the supplied paper environment, not selected implementation pins.

## Tabular app boundary

The official model table describes V3 numerical/categorical features and missing feature values, with ceilings of 1,000,000 training rows and 2,000 columns. V3.5 raises the column ceiling to 20,000 and adds local text support. The documentation explicitly cautions that row and column maxima cannot necessarily be combined. These model ceilings are not appropriate service resource promises. The model weights have separate noncommercial terms; access and deployment permission need checking independently of source-code licensing. [Model capabilities and terms](https://docs.priorlabs.ai/models).

The estimator supports one regression target, an sklearn-style `fit`/`predict` interface, explicit categorical-column indices, and fit modes including `fit_preprocessors` and `fit_with_cache`. Its current source recommends leaving raw categorical/missing features to native preprocessing. A supplied integer `n_estimators` is a scientific setting; automatic defaults can change with model/configuration. [Regressor source](https://raw.githubusercontent.com/PriorLabs/TabPFN/main/src/tabpfn/regressor.py).

**Recommended first app contract:** training table, target-column name, optional row-ID column, explicitly selected ordered feature names, inference table, and a small fixed set of model settings. Use Polars for ingestion and output. Convert only at the estimator boundary to its supported NumPy/pandas representation. Preserve feature names, logical types, categorical intent, and order in a manifest; reorder by recorded names rather than trusting uploaded column order. Reject missing/extra selected features or incompatible types instead of silently coercing them. Numeric ESMC features need no pandas parsing.

Choose regression-only initially unless generic classification is genuinely needed. Missing feature values can be supported without accepting missing, infinite, or censored targets. Do not silently infer a target from the last column, or train on identifiers. Bound bytes, rows, columns, and inference work independently after representative measurements. Predict in batches; single-row calls repeatedly incur context work.

Regression can return mean, median, mode, quantiles, or a richer distribution. Quantile output is available without inventing a separate ensemble uncertainty estimate. A predictive interval is not automatically a calibrated confidence interval for affinity improvement or epistemic uncertainty. [Regression API](https://docs.priorlabs.ai/capabilities/regression).

For cached inference, current documentation says test rows are conditionally independent given the KV cache, but floating-point results can differ with chunking. That statement is not a tested equivalence guarantee for the proposal's pinned V8 `fit_preprocessors` path; freeze chunk size and test tolerance. [Chunking documentation](https://github.com/PriorLabs/TabPFN#what-environment-variables-can-i-use-to-configure-tabpfn).

## Fitted model reuse is not ordinary weight training

The native persistence API is `save_fitted_tabpfn_model` / `load_fitted_tabpfn_model`. Current source saves estimator initialization, fitted attributes, and inference-engine state in an archive; the latter two use joblib. It does not package the foundation weights in that archive and initializes those separately at load. Consequently, fitted bundles must be treated as private training-context artifacts, not anonymous generic checkpoints. They also require the matching runtime and foundation model. [Persistence implementation](https://raw.githubusercontent.com/PriorLabs/TabPFN/main/src/tabpfn/model_loading.py).

Version 8.0.3's loader identifies the V3 regressor as `Prior-Labs/tabpfn_3` / `tabpfn-v3-regressor-v3_default.ckpt`. Its default downloader follows repository state rather than a caller-selected immutable revision; its checkpoint loader uses `torch.load`. Provision an approved pinned checkpoint explicitly, record its digest, and do not accept arbitrary user-supplied weights. [Pinned loader](https://raw.githubusercontent.com/PriorLabs/TabPFN/v8.0.3/src/tabpfn/model_loading.py).

**First-release boundary:** the user selected fit and inference in one invocation, with no fitted-model reuse or exported model bundle. The persistence findings explain why this removes an otherwise meaningful trust/compatibility boundary; do not implement native save/load or user model uploads. Foundation-weight provisioning remains necessary.

The common-utils telemetry package documents `TABPFN_DISABLE_TELEMETRY=1`. Set this before imports for the pinned V8 environment and verify network behavior in an isolated smoke test. Current V9 metadata no longer lists that dependency, so do not describe one version's telemetry implementation as universal. Checkpoint authentication/download traffic is distinct from optional usage telemetry. [Official common-utils documentation](https://github.com/PriorLabs/tabpfn_common_utils).

## ESMC feature extraction

The current official ESM README advertises `esm v3.4.post1` and a local interface using `EsmcForMaskedLM.from_pretrained(...)`, `EsmcTokenizer`, evaluation mode, and `torch.inference_mode()`. The SDK `encode`/`logits` example elsewhere in the README is the hosted client; it should not accidentally send proprietary sequences to a remote inference service. The current local implementation differs from older `ESMC.from_pretrained` examples. [Official ESM README](https://github.com/Biohub/esm).

Official configuration defines `biohub/ESMC-300M`, `biohub/ESMC-600M`, and `biohub/ESMC-6B`. Configuration fields include `hidden_size`; compatibility mapping still accepts the older `d_model` spelling. This audit could not retrieve the live 300M checkpoint configuration, so the proposed 960-dimensional width is not independently confirmed here. Read and validate the pinned downloaded configuration before freezing output schema. [Configuration source](https://raw.githubusercontent.com/Biohub/esm/main/esm/models/esmc/config.py).

The implementation returns `last_hidden_state` with batch, token, and representation dimensions after final layer normalization. Requesting all hidden states also retains all layers; that is unnecessary for final-layer pooling. The masked-LM wrapper computes vocabulary logits, while the exposed base `EsmcModel` returns representations without that head. Prefer the smallest officially supported representation path once the selected package/checkpoint is smoke-tested. [Model implementation](https://raw.githubusercontent.com/Biohub/esm/main/esm/models/esmc/model.py).

The tokenizer defines special tokens, including chain breaks. Pool only actual amino-acid tokens, excluding padding and boundary tokens; an attention mask alone need not implement that distinction. [Tokenizer source](https://raw.githubusercontent.com/Biohub/esm/main/esm/models/esmc/tokenizer.py).

**Accepted representation direction:** deduplicate exact chains within a run, encode each complete chain independently, mean-pool the final residue representations in float32, and concatenate in the fixed recorded order of the user-supplied chain IDs. The website now supports general proteins, not only H/L pairs. This is the proposal's own representation, not a verified recreation of every feature in the cited paper. Reuse unchanged chains within the run; do not approximate a double-mutant embedding by adding single-mutant embeddings. Keep feature compression inside training folds. Cross-run embedding caching is not approved.

**Feature-count consequence:** assuming the proposal's 960-wide encoder, paired global means require 1,920 columns, just below V3's documented 2,000-column ceiling. Its mutation-aware challenger has `3 * 960 + 1 = 2,881` columns per chain and 5,762 for pairs; it cannot be passed unchanged to that baseline within the documented ceiling. Training-fold PCA or a narrower feature recipe is required; bypassing the size guard is not a scientific validation strategy.

## Remaining verification

The evolving [specification](../specs/protein-sequence-optimization.md) owns product choices and open questions. Remaining technical evidence includes dependency solve and image installation; authorized pinned checkpoint acquisition; fit/predict and chunked prediction tolerance; pooling/padding/batch-order invariance; actual memory/time for the agreed training and candidate limits. No performance or cost numbers in this note were measured. Native fitted-model persistence tests are unnecessary for the accepted first-release scope.

## Validation split recommendation (September 26)

Validation should imitate the intended prediction task rather than randomly separate replicate rows. Protein-fitness studies distinguish held-out combinations near the training regime from higher mutation-count extrapolation; these are different tests, not interchangeable accuracy estimates. The following support-constrained split is a design recommendation for this product, not a claim that one published benchmark prescribes it. [Protein extrapolation study](https://www.nature.com/articles/s41467-024-50712-3).

**Combination:** withhold entire measured multi-mutant variants only when every exact substitution in each held-out variant remains represented among training variants. For example, retain measured `A:Y52F` and `B:S30A`, and predict the withheld `A:Y52F,B:S30A`. Single-mutant measurements are useful training anchors and need not be held out. If support exists only within other combinations, retain those instead, but flag that co-occurring substitutions may be confounded: occurrence alone does not establish independent effects. Check support after assigning the whole held-out fold, not independently for each row; otherwise two withheld variants can remove one another's training support.

This tests new combinations of measured substitutions. With only single-mutant measurements, no measured combinations exist against which to test that claim: report combination validation unavailable and still allow explicitly warned additive predictions. Do not hold out a substitution from one-hot ridge and present its unknown coefficient as a test of recombination. Report the number and mutation-count range of scored held-out variants; double-mutant validation does not establish accuracy for five-mutant designs.

**Exploration:** the user's proposed same-position/different-amino-acid test is appropriate for previously unseen alternatives at known positions. To withhold `A:Y52W`, remove every training variant containing that exact substitution, including multi-mutants and replicates, while retaining another measured alternative such as `A:Y52F`. For a clean one-new-substitution test, require the other substitutions of the scored held-out variants to retain training support. This does not validate exploration at entirely unmeasured positions; identify that limitation instead of pooling both tasks under one score. If no eligible alternate-amino-acid contrast exists, report this validation unavailable.

Canonicalize variants and aggregate their replicates before splitting, preserving replicate count and dispersion; never split the same sequence across training and validation. Group-aware validation is specifically intended to prevent related samples from appearing in both sides. [Scikit-learn grouped validation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data).

Use a reproducible, label-blind split construction and at most five bounded folds or holdouts; fewer valid splits are acceptable. Fold count is an implementation budget, not evidence of reliability. Report coverage rather than forcing every row into a holdout. If a variant appears in multiple exploration holdouts, do not count those as independent observations in an aggregate metric. Show prediction error in the supplied label units and rank correlation only when its sample size and variation permit a defined statistic. Retain final fit on all usable measurements after validation.

All learned preprocessing, including PCA or feature scaling, must be fitted on training data within each fold. Any ridge-penalty tuning must remain inside that training portion (or use a prespecified penalty), never reuse the final evaluation labels for model selection. Frozen sequence embeddings can be computed once because their checkpoint is not fitted on these measurements. [Scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).

## Implementation verification — September 26

PyPI release metadata reconfirmed scikit-learn 1.9.1, TabPFN 9.0.0 and ESM 3.4.1.post1. The ridge implementation pins scikit-learn 1.9.1 and uses sparse CSR inputs with `Ridge(alpha=1.0, fit_intercept=True, solver="lsqr", tol=1e-10)`. This solver supports intercept-bearing sparse regression; no scaler or fitted-model serialization is involved. [Native Ridge reference](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html).

A single local synthetic Combination measurement used a 70-residue parent, parent plus 70 single-mutant observations, and enumeration through four mutations: 974,050 novel rows, 139,420,399 output bytes, 3.95 seconds wall time and 1,014,624 KiB maximum process RSS under `/usr/bin/time -v`. This includes imports, fitting, streaming candidate CSV generation, Polars streaming sorting, and a row-count verification. Temporary files were removed; no benchmark-only code was added to production. This is a development-machine observation, not a remote SLA or evidence about longer/multichain outputs, repeated API queries or Exploration cost. Native tests separately cover additive recovery, interaction failure, whole-fold support, final all-data fitting, batching and empty output.

## Native release and provisioning audit — September 26

This follow-up inspected the published wheels, public checkpoint metadata and
small configuration files, and performed a metadata-only dependency solve in
`/tmp`. No package installation, foundation-weight download, model execution,
Modal call or deployment occurred. Earlier main-branch observations above are
superseded by the exact versions below where they differ.

### Package and checkpoint identities

| Component | Selected immutable identity |
| --- | --- |
| TabPFN package | `tabpfn==9.0.0`; wheel SHA256 `0adb0e69be839051c7f9e92caadec93324cccc0256321bf967fda91e5c493f98` |
| TabPFN source | Tag `v9.0.0`, commit `c70b6ef0488858d32244c52222abfc0c5be207d6` |
| TabPFN weights | `Prior-Labs/tabpfn_3_5`, revision `06bf2ba35c80a92a3b9abb436b99cf49e7a0365e`, file `tabpfn-v3.5-20260909.safetensors` |
| TabPFN weight digest | SHA256 `ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3`; 876,027,932 bytes |
| ESM package | `esm==3.4.1.post1`; wheel SHA256 `f9e62b363519860d27762989871a531cdbd02d69c824cc8c8a1047b16dce3ef6` |
| ESM source | Tag `v3.4.1.post1`, commit `43b4548b86762edfa747b07d5f440aad3c33acee` |
| ESMC600M weights | `biohub/ESMC-600M`, revision `fcc9d36f5fff97d9bb69a36e3252de61cc78c968`, file `model.safetensors` |
| ESMC weight digest | SHA256 `526573c304b7718f9cfd2a8adf79638feb6752eb38056968bf21d9036b7882ce`; 2,300,201,352 bytes |
| ESMC configuration | Same revision, `config.json`; SHA256 `2494562385804f86db4ba04cba57e4453b1c080ce9eb758443df67ee852bfd31` |

Wheel hashes were computed locally and matched PyPI. Checkpoint hashes and
sizes are publisher LFS metadata, **not locally verified weight downloads**.
The default V3.5 checkpoint serves both regression and classification; there
is no separate selected regressor file. The inspected regressor/model source
files matched the immutable source commits byte-for-byte.
[TabPFN release](https://pypi.org/pypi/tabpfn/9.0.0/json),
[ESM release](https://pypi.org/pypi/esm/3.4.1.post1/json),
[TabPFN checkpoint metadata](https://huggingface.co/api/models/Prior-Labs/tabpfn_3_5/revision/06bf2ba35c80a92a3b9abb436b99cf49e7a0365e?blobs=true),
[ESMC checkpoint metadata](https://huggingface.co/api/models/biohub/ESMC-600M/revision/fcc9d36f5fff97d9bb69a36e3252de61cc78c968?blobs=true).

### Compatible dependencies, without claiming runtime validation

ESM requires Python 3.12+, `torch>=2.11.0,<2.12.0` and
`transformers>=4.57.6,<5.0.0`. The exact compatible releases verified here are
`torch==2.11.0` and `transformers==4.57.6`. TabPFN requires Torch 2.5+ and
accepts that selection. Do not independently upgrade to the newest Torch or
Transformers major release. On Linux x86_64, ESM also unconditionally depends
on `cuequivariance-torch` and `cuequivariance-ops-torch-cu13`; Torch 2.11's
default Linux wheel resolves CUDA 13 dependencies. This needs a compatible
image/driver, even when using the ordinary SDPA attention backend.
[ESM dependency metadata](https://pypi.org/pypi/esm/3.4.1.post1/json),
[Torch metadata](https://pypi.org/pypi/torch/2.11.0/json),
[Transformers metadata](https://pypi.org/pypi/transformers/4.57.6/json).

A successful `uv pip compile --python-version 3.12
--python-platform x86_64-unknown-linux-gnu --no-build` with the five direct
pins `tabpfn==9.0.0`, `esm==3.4.1.post1`, `torch==2.11.0`,
`transformers==4.57.6`, `scikit-learn==1.9.1` selected these relevant
transitives: `huggingface-hub==0.36.2`, `tokenizers==0.22.2`,
`safetensors==0.8.0`, `accelerate==1.15.0`, `numpy==2.5.3`,
`scipy==1.18.1`, `pandas==3.0.6`, `einops==0.8.2`, `skrub==0.10.1`,
`lightgbm==4.7.0`, `biotite==1.7.1`, `rdkit==2026.3.6`,
`cuequivariance-torch==0.12.0`, `cuequivariance-ops-torch-cu13==0.12.0`
and `triton==3.6.0`. This proves dependency satisfiability on the selected
platform, not installation, import, GPU-kernel or model compatibility.
Pandas is an upstream dependency; application table parsing remains Polars.

### Native TabPFN integration

Construct `TabPFNRegressor` with an existing absolute checkpoint path,
explicit integer `n_estimators`, `random_state`, `device`,
`inference_precision`, and `fit_mode="fit_preprocessors"`. Call
`fit(X_train, y_train)` once, then `predict(X_chunk, output_type="mean")`
on bounded groups of rows. The constructor's `fit_mode="batched"` is a
different interface for already-preprocessed data/fine-tuning and is not
needed for ordinary batched prediction. Record chunk size: the chosen fit
mode recomputes transformer context on each prediction call.
[Pinned regressor API](https://github.com/PriorLabs/TabPFN/blob/c70b6ef0488858d32244c52222abfc0c5be207d6/src/tabpfn/regressor.py).

Use a numerical NumPy matrix for embeddings, or a mixed NumPy/pandas boundary
representation preserving numerical values, missing values and declared
categorical strings. Pass zero-based `categorical_features_indices`. A
declared **string** column remains categorical at any cardinality; a declared
numeric column can still be inferred numerical. Numeric-looking category
labels therefore need a representation that preserves category intent.
Disable `TRANSFORM_TEXT` and `TRANSFORM_DATES` for this app's selected
numerical/categorical contract. No external one-hot/scaling step is required
for TabPFN. Record the resolved inferred feature schema alongside requested
types; the upstream categorical-index setting is not a universal hard typing
override.
[Pinned modality detection](https://github.com/PriorLabs/TabPFN/blob/c70b6ef0488858d32244c52222abfc0c5be207d6/src/tabpfn/preprocessing/modality_detection.py),
[pinned inference configuration](https://github.com/PriorLabs/TabPFN/blob/c70b6ef0488858d32244c52222abfc0c5be207d6/src/tabpfn/inference_config.py).

V3.5 documents 1,000,000 training rows and 20,000 input features, with at most
6,000 columns recommended; maxima cannot necessarily be combined. CPU use
has a separate 5,000-row safeguard. These are model limits, not approved
service ceilings. Per-estimator feature subsampling and other resolved
defaults are read from checkpoint metadata. An explicit ensemble size can
leave features uncovered if too small; inspect and record the resolved
preprocessing after approved provisioning. The wheel alone does not expose
all selected checkpoint settings. Hosted text-token limits do not apply to
this numerical/categorical local adapter.
[Model limits](https://docs.priorlabs.ai/models),
[checkpoint configuration loading](https://github.com/PriorLabs/TabPFN/blob/c70b6ef0488858d32244c52222abfc0c5be207d6/src/tabpfn/model_loading.py).

The V9 wheel has no `tabpfn-common-utils` dependency and a source search found
no telemetry, PostHog or Sentry integration. `TABPFN_DISABLE_TELEMETRY=1`
is a legacy precaution, not a verified V9 switch. Set
`HF_HUB_DISABLE_TELEMETRY=1`, `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1` before imports in inference containers. Native
TabPFN loading skips download/authentication when its exact file exists;
preflight that path because a missing file otherwise triggers download and
license-authentication code. `TABPFN_NO_BROWSER=1` prevents interactive
login, but is not a network-denial mechanism.
[Pinned loader](https://github.com/PriorLabs/TabPFN/blob/c70b6ef0488858d32244c52222abfc0c5be207d6/src/tabpfn/model_loading.py),
[Hugging Face environment controls](https://github.com/huggingface/huggingface_hub/blob/v0.36.0/src/huggingface_hub/constants.py).

### Native ESMC600M integration and bounds

The selected configuration confirms **1,152 features per chain**, 36 layers
and 18 attention heads. Two concatenated chain means therefore have 2,304
features; the earlier 960-wide assumption does not describe ESMC600M.
The model card specifies a 2,048-token context. With the native tokenizer's
one leading `<cls>` and one trailing `<eos>`, enforce at most **2,046
residues per independently encoded chain**, without truncation. This is a
documented-context bound: native `EsmcConfig` ignores
`max_position_embeddings`, and the published tokenizer configuration has
an effectively unbounded `model_max_length` sentinel. Neither protects the
application from excessive input.
[Pinned configuration](https://huggingface.co/biohub/ESMC-600M/blob/fcc9d36f5fff97d9bb69a36e3252de61cc78c968/config.json),
[model context](https://huggingface.co/biohub/ESMC-600M/blob/fcc9d36f5fff97d9bb69a36e3252de61cc78c968/README.md),
[tokenizer configuration](https://huggingface.co/biohub/ESMC-600M/blob/fcc9d36f5fff97d9bb69a36e3252de61cc78c968/tokenizer_config.json).

Use `from esm.models.esmc import EsmcModel, EsmcTokenizer` and
`EsmcModel.from_pretrained(local_directory, device="cuda",
dtype=torch.float32, attn_implementation="sdpa", local_files_only=True)`.
This is the package's native loader; it accepts `device` and `dtype`, not
the Transformers `device_map`/`torch_dtype` keywords. The bare encoder
explicitly strips the `esmc.` weight prefix and permits unused `lm_head.*`
keys. It avoids vocabulary-logit computation. There is no need for
`trust_remote_code` or downloading the repository's remote Python adapter.
Loaders return evaluation mode; retain explicit `.eval()` and
`torch.inference_mode()` in the adapter.
[Pinned native model](https://github.com/Biohub/esm/blob/43b4548b86762edfa747b07d5f440aad3c33acee/esm/models/esmc/model.py),
[pinned local loader](https://github.com/Biohub/esm/blob/43b4548b86762edfa747b07d5f440aad3c33acee/esm/models/hub.py).

`EsmcTokenizer()` constructs the fixed character vocabulary locally. Encode
a bounded list with `padding=True`, `truncation=False`,
`return_tensors="pt"`, `return_special_tokens_mask=True`. Remove the special
mask from model kwargs. The encoder accepts `input_ids` and `attention_mask`
and returns `last_hidden_state` shaped `(batch, tokens, 1152)` after final
normalization. Mean-pool in float32 over attended, non-special residue
tokens; for validated standard amino acids, IDs 4 through 23 are exactly
the 20 accepted residues. Assert residue count equals each original chain
length. Padding ID 1, `<cls>` ID 0, `<eos>` ID 2, mask and chain-break
tokens must contribute zero. Do not request all hidden states, logits or
SAE outputs. Preserve stable chain order and deduplicate unchanged chain
strings only within the invocation. Padding/batch-order invariance remains
an authorized runtime smoke-test requirement, not a result of this audit.
[Pinned tokenizer](https://github.com/Biohub/esm/blob/43b4548b86762edfa747b07d5f440aad3c33acee/esm/models/esmc/tokenizer.py),
[encoder output](https://github.com/Biohub/esm/blob/43b4548b86762edfa747b07d5f440aad3c33acee/esm/models/esmc/model.py).

### Separate provisioning and read-only inference runbook

1. Before downloading weights, confirm applicable use rights and obtain
   authorization for acquisition. TabPFN's license explicitly requires a
   separate commercial license for a hosted/API/SaaS service, **including a
   free service**, and for production use of outputs. Public metadata
   currently reports `gated=false`; that is not a usage grant. Its default
   downloader still implements Prior Labs acceptance/token handling. ESM's
   selected release states MIT model licensing and links its acceptable-use
   policy; this is a newer model release, not the old 2024 ESMC license.
   [TabPFN license, sections 2 and 3](https://huggingface.co/Prior-Labs/tabpfn_3_5/blob/06bf2ba35c80a92a3b9abb436b99cf49e7a0365e/LICENSE),
   [ESM release licensing](https://github.com/Biohub/esm/blob/43b4548b86762edfa747b07d5f440aad3c33acee/README.md#licenses),
   [ESM license](https://github.com/Biohub/esm/blob/43b4548b86762edfa747b07d5f440aad3c33acee/LICENSE.md).
2. A separate authorized provisioning command/function gets writable model
   storage and any required credentials. Use explicit `hf_hub_download`
   repository/revision/filename arguments: the single TabPFN safetensors
   file; ESMC `model.safetensors` and `config.json`. The locally constructed
   tokenizer needs no extra assets. Download into staging, verify SHA256
   and byte counts, store repository/revision/package/license provenance,
   then atomically publish the completed immutable revision directory and
   commit the volume. Never publish a completion marker before verification.
   [Hub pinned downloads](https://github.com/huggingface/huggingface_hub/blob/v0.36.0/src/huggingface_hub/file_download.py).
3. Inference mounts that storage read-only using
   `volume.with_mount_options(read_only=True)`. Supply only explicit local
   paths, preflight the provisioned manifest, and fail on missing/mismatched
   artifacts. Keep disposable runtime caches outside this mount, with no
   provisioner credentials in the inference container. No lazy weight
   download or fitted-model write belongs in `fit`/`predict`.
   [Modal read-only mounts](https://modal.com/docs/guide/volumes#read-only-mounts).
4. After separately approved compute is available, verify image installation
   and imports, offline weight loading, resolved TabPFN checkpoint settings,
   numerical/categorical/missing/unseen-category predictions, chunk
   tolerance, ESM residue masking and batch invariance, and resource use.
   Test missing-artifact failure with outbound model-host access disabled.
   Record actual runtime ceilings before making service capacity promises.

Outstanding gates are the TabPFN hosted-use license, authorized weight
provisioning, native GPU/runtime verification and measured service bounds.
Dependency metadata and mock adapters alone do not satisfy those gates.

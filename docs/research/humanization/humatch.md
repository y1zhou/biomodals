# Humatch app research

Research date: 2026-09-02

Likely implementation target: `src/biomodals/app/design/humatch/app.py`

## Recommendation in brief

Humatch is a useful independent candidate generator for the antibody-
humanization stack because it is the only method in the planned set that
explicitly optimizes a **paired** VH/VL and targets named human V-gene
families. It should be exposed as a paired humanization operation, not as a
per-residue scorer: its scores are sequence-level classifier probabilities,
and its mutation policy depends on all three heavy, light, and paired models
([paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11610552/),
[implementation](https://github.com/oxpig/Humatch/blob/06205ad50f64b84468fea14eea573a3fce699550/Humatch/humanise.py)).

The first Biomodals version can use one CPU container per bounded CSV batch.
The upstream paper reports a mean of 36 seconds per pair on a 16-CPU desktop,
and the implementation already batches the thousands of single-point variants
created within an iteration. Benchmark 1, 10, and 100 pairs before considering
pair-level fanout
([paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11610552/),
[data generator](https://github.com/oxpig/Humatch/blob/06205ad50f64b84468fea14eea573a3fce699550/Humatch/dataset.py)).

## Immutable source and asset pins

- The current upstream `master` is
  [`06205ad50f64b84468fea14eea573a3fce699550`](https://github.com/oxpig/Humatch/tree/06205ad50f64b84468fea14eea573a3fce699550),
  committed 2025-11-11. There are no Git tags or GitHub releases. Package
  metadata calls this version `1.0.1`, so install the pinned Git commit rather
  than treating `1.0.1` as an immutable published distribution
  ([setup.py](https://github.com/oxpig/Humatch/blob/06205ad50f64b84468fea14eea573a3fce699550/setup.py)).
- The inference assets are the immutable Zenodo version record
  [`10.5281/zenodo.13764771`](https://zenodo.org/records/13764771). It contains
  three Keras weight files and 24 V-gene frequency arrays. The code downloads
  from that version record but neither checks status codes nor verifies
  content
  ([model loader](https://github.com/oxpig/Humatch/blob/06205ad50f64b84468fea14eea573a3fce699550/Humatch/model.py),
  [array loader](https://github.com/oxpig/Humatch/blob/06205ad50f64b84468fea14eea573a3fce699550/Humatch/germline_likeness.py)).

| Runtime asset | Size | Zenodo MD5 |
| --- | ---: | --- |
| `heavy.weights.h5` | 28.8 MB | `9a881232a25bf8b6f8399df4ec19acd5` |
| `light.weights.h5` | 28.8 MB | `f9ef8c5e75ee157c7d2b0efadb5b2a32` |
| `paired.weights.h5` | 59.0 MB | `2941dd7079ab3437110cdfdb76b3fb01` |
| `hv1`-`hv7`, `lv1`-`lv10`, `kv1`-`kv7` `.npy` files | 32.1 kB each | Per-file manifest on the [Zenodo record](https://zenodo.org/records/13764771) |

The three training-data archives on the same record total about 1 GB and are
not needed for inference. Bake only the three weights and 24 lookup arrays into
the image, verify every recorded MD5 while building, and disable runtime
downloads. This is roughly 117 MB of weights plus less than 1 MB of lookup
tables
([Zenodo record](https://zenodo.org/records/13764771),
[package data declaration](https://github.com/oxpig/Humatch/blob/06205ad50f64b84468fea14eea573a3fce699550/setup.py)).

## Licenses

The source is BSD-3-Clause, copyright Lewis Alexander Chinery (2024)
([license](https://github.com/oxpig/Humatch/blob/06205ad50f64b84468fea14eea573a3fce699550/LICENSE)).
The Zenodo asset record declares CC-BY-4.0 for the weights, lookup arrays, and
training/evaluation data; an image that redistributes inference assets must
retain attribution to the record and its creator
([version DOI](https://doi.org/10.5281/zenodo.13764771)). ANARCI, the required
numbering dependency, is also BSD-3-Clause
([ANARCI repository](https://github.com/oxpig/ANARCI)). This is not legal
advice, but no non-commercial restriction was found in the materials needed
for inference.

## Scientific method and score semantics

Humatch contains three CNN classifiers. CNN-H assigns a complete heavy chain
to a non-human class or human V-gene family HV1-HV7; CNN-L assigns a complete
light chain to non-human or LV1-LV10/KV1-KV7; CNN-P assigns the concatenated
pair to artificial (`fake`) or natural (`true`) pairing. Inputs are aligned to
200 common IMGT positions, encoded with ten-dimensional Kidera factors, and a
pair is represented as 200 heavy positions, ten zero-padding positions, and
200 light positions
([paper methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC11610552/),
[class constants](https://github.com/oxpig/Humatch/blob/06205ad50f64b84468fea14eea573a3fce699550/Humatch/utils.py),
[model architecture](https://github.com/oxpig/Humatch/blob/06205ad50f64b84468fea14eea573a3fce699550/Humatch/model.py)).

CNN-H and CNN-L are multiclass softmax probabilities. `CNN_H` and `CNN_L` in a
humanization result are the final probabilities assigned to the selected
target V-gene families, while `CNN_P` is the final probability assigned to the
`true` paired class. They are model classifier outputs, not residue
probabilities, calibrated immunogenicity risks, binding predictions, or direct
measures of germline identity
([classification implementation](https://github.com/oxpig/Humatch/blob/06205ad50f64b84468fea14eea573a3fce699550/Humatch/classify.py),
[paper discussion](https://pmc.ncbi.nlm.nih.gov/articles/PMC11610552/)).

Humanization has two phases:

1. Select explicit target gene families or choose each chain's highest-scoring
   family. Mutate the least germline-like framework residue, one at a time,
   toward the modal residue for that family until the mean position-frequency
   score reaches 0.40 by default.
2. Enumerate every allowed canonical single substitution in either chain.
   Score variants with the relevant chain CNN and CNN-P, scale score changes by
   target-gene residue frequency and distance from each target threshold, then
   choose the best previously unseen variant. Stop when CNN-H, CNN-L, and CNN-P
   all reach 0.95 by default, the combined edit distance exceeds 60, or no new
   variant remains.

CDRs are excluded from both phases by default; users may allow CDR mutation or
fix additional IMGT positions. Although the default YAML contains
`noise: 0.01`, `humanise(...)` never reads it: prediction scaling always uses
the helper's default `noise_factor=0.01`. The effective constant changes the
frequency weighting and is not random noise
([paper method](https://pmc.ncbi.nlm.nih.gov/articles/PMC11610552/),
[default config](https://github.com/oxpig/Humatch/blob/06205ad50f64b84468fea14eea573a3fce699550/Humatch/configs/default.yaml),
[humanization source](https://github.com/oxpig/Humatch/blob/06205ad50f64b84468fea14eea573a3fce699550/Humatch/humanise.py),
[germline scoring](https://github.com/oxpig/Humatch/blob/06205ad50f64b84468fea14eea573a3fce699550/Humatch/germline_likeness.py)).

The training evidence is broad but not universal. CNN-H/L used 8.26/12.73
million human and 3.77/1.41 million non-human sequences from OAS, with rhesus
included among the harder negative examples. CNN-P used 1.67 million natural
human pairs and 5.01 million artificially mismatched human pairs; no non-human
pairs trained CNN-P. The authors explicitly warn that artificial bad pairs are
noisy and that Humatch designs, like other computational designs, lack direct
experimental safety validation
([paper methods and discussion](https://pmc.ncbi.nlm.nih.gov/articles/PMC11610552/)).

## Supported inputs and outputs

Humanization requires a complete VH **and** complete VL. The paper's training
filter required conserved cysteines at IMGT 23 and 104 and coverage from IMGT
1 through 128 for heavy or 127 for light. The implementation supports kappa
and lambda light chains and has no source-species selector: its intended input
is an animal-derived paired Fv, but the evidence does not establish equal
validity for every species or for VHH/single-domain antibodies
([paper methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC11610552/),
[CLI validation](https://github.com/oxpig/Humatch/blob/06205ad50f64b84468fea14eea573a3fce699550/Humatch/humanise.py)).

Upstream exposes three console commands and ordinary Python functions:

- `Humatch-align` accepts individual VH/VL strings or CSV columns and emits
  the 200-position alignment, optionally one column per IMGT position.
- `Humatch-classify` permits one chain or a pair and emits all V-family
  probabilities, or top families and scores with `--summarise`.
- `Humatch-humanise` requires both chains, accepts a YAML config, and emits
  `Humatch_H`, `Humatch_L`, `Edit`, `HV`, `LV`, `CNN_H`, `CNN_L`, and `CNN_P`.

CSV humanization is sequential over rows. Returned chain strings are still
200-position padded strings, although terminal output removes `-` characters
([README](https://github.com/oxpig/Humatch/blob/06205ad50f64b84468fea14eea573a3fce699550/README.md),
[alignment CLI](https://github.com/oxpig/Humatch/blob/06205ad50f64b84468fea14eea573a3fce699550/Humatch/align.py),
[humanization CLI](https://github.com/oxpig/Humatch/blob/06205ad50f64b84468fea14eea573a3fce699550/Humatch/humanise.py)).

## Runtime, batching, memory, and determinism

Upstream documents Python 3.9. Its metadata has only lower bounds for NumPy,
pandas, TensorFlow, scikit-learn, plotting packages, PyYAML, and Biopython, and
pins `hmmer==3.4.0.0`; ANARCI must be installed separately. The paper records
Python 3.9, TensorFlow 2.16, and Keras 3.0, whereas current metadata requests
TensorFlow 2.17 or newer. A reproducible app must choose and lock one
Python/TensorFlow/ANARCI stack, rather than resolving these unconstrained
dependencies at every image build
([README](https://github.com/oxpig/Humatch/blob/06205ad50f64b84468fea14eea573a3fce699550/README.md),
[setup.py](https://github.com/oxpig/Humatch/blob/06205ad50f64b84468fea14eea573a3fce699550/setup.py),
[paper methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC11610552/)).

The supported path is CPU TensorFlow. Upstream suppresses “no GPU” messages,
never configures a GPU, encodes sequences through a multiprocessing pool, and
passes variants to Keras with a nominal batch size of 16,384. Default config
requests 16 CPU workers. One humanization iteration can create roughly 19
variants per mutable residue across both chains, so a pair may create several
thousand 200- or 410-position float64 arrays. Benchmark memory as well as
latency; 4-8 GiB is a safer initial characterization range than assuming the
117 MB weight footprint is the whole working set
([dataset](https://github.com/oxpig/Humatch/blob/06205ad50f64b84468fea14eea573a3fce699550/Humatch/dataset.py),
[classification](https://github.com/oxpig/Humatch/blob/06205ad50f64b84468fea14eea573a3fce699550/Humatch/classify.py),
[default config](https://github.com/oxpig/Humatch/blob/06205ad50f64b84468fea14eea573a3fce699550/Humatch/configs/default.yaml)).

Inference and greedy selection contain no sampling; Keras `predict` disables
dropout. Results should be repeatable within a pinned CPU software stack.
Floating-point reductions, ties, TensorFlow kernels, and worker counts can
still affect close rankings, so treat exact sequences as deterministic only
after repeat testing on the deployed stack and record all dependency and asset
identities in the result manifest
([model](https://github.com/oxpig/Humatch/blob/06205ad50f64b84468fea14eea573a3fce699550/Humatch/model.py),
[selection logic](https://github.com/oxpig/Humatch/blob/06205ad50f64b84468fea14eea573a3fce699550/Humatch/humanise.py)).

## Validation, security, and upstream hazards

1. **Upstream silently deletes bad residues.** `strip_padding_from_seq` keeps
   only 20 uppercase canonical amino acids, so lowercase, ambiguous residues,
   punctuation, and embedded whitespace can silently change an input. A
   Biomodals app must reject them locally and remotely instead
   ([alignment source](https://github.com/oxpig/Humatch/blob/06205ad50f64b84468fea14eea573a3fce699550/Humatch/align.py)).
2. **Failed rows are silently omitted.** CSV humanization drops any pair whose
   VH or VL fails ANARCI and returns the remaining rows without input IDs. That
   destroys row identity and violates an atomic scientific batch contract.
   Require unique `id,vh,vl`, validate every row before inference, and fail the
   whole batch with indexed errors
   ([humanization CLI](https://github.com/oxpig/Humatch/blob/06205ad50f64b84468fea14eea573a3fce699550/Humatch/humanise.py)).
3. **Runtime downloads are unauthenticated.** `requests.get` has no timeout,
   status check, size limit, or checksum and writes into installed package
   directories. Bake verified assets and make inference offline
   ([model loader](https://github.com/oxpig/Humatch/blob/06205ad50f64b84468fea14eea573a3fce699550/Humatch/model.py),
   [array loader](https://github.com/oxpig/Humatch/blob/06205ad50f64b84468fea14eea573a3fce699550/Humatch/germline_likeness.py)).
4. **Free-form YAML is too broad.** The CLI accepts an arbitrary config file
   and does not validate types/ranges. Expose typed fields for target families,
   thresholds, maximum edits, CDR mutation, and fixed IMGT positions; do not
   accept YAML bytes or file paths across the trust boundary
   ([config loading](https://github.com/oxpig/Humatch/blob/06205ad50f64b84468fea14eea573a3fce699550/Humatch/humanise.py)).
5. **Threshold and edit semantics are surprising.** The loop stops only after
   selecting a variant that can push `Edit` *above* `max_edit`, then falls back
   to the previously visited sequence with the largest unscaled sum of the
   three CNN scores. The output has no explicit `humanisation_failed` field.
   Preserve this behavior for equivalence. The first implementation will use
   upstream code without patches and derive only success or failure from the
   returned scores; an exact stop reason would require later instrumentation
   ([humanization source](https://github.com/oxpig/Humatch/blob/06205ad50f64b84468fea14eea573a3fce699550/Humatch/humanise.py)).
6. **Scores must not be overclaimed.** The paired negative class is synthetic,
   and the authors say computational designs are predicted rather than proven
   safe. Label CNN outputs by their exact target classes and retain the usual
   requirement for experimental binding, stability, expression, and
   immunogenicity evaluation
   ([paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11610552/)).
7. **Package metadata omits imports.** `model.py` and
   `germline_likeness.py` import `requests`, and the aligner imports `anarci`,
   but neither package appears in `install_requires`. Install and pin both
   explicitly rather than relying on an incidental transitive dependency
   ([setup.py](https://github.com/oxpig/Humatch/blob/06205ad50f64b84468fea14eea573a3fce699550/setup.py),
   [model loader](https://github.com/oxpig/Humatch/blob/06205ad50f64b84468fea14eea573a3fce699550/Humatch/model.py),
   [aligner](https://github.com/oxpig/Humatch/blob/06205ad50f64b84468fea14eea573a3fce699550/Humatch/align.py)).

Apply explicit byte and row limits, canonical amino-acid checks, complete-Fv
checks, unique IDs, finite numeric parameter bounds, and target/fixed-position
enum validation at both trust boundaries. Do not expose arbitrary paths,
regular expressions, URLs, or environment settings.

## Equivalence and test strategy

Create a pinned legacy oracle container from the exact Git commit, all 27
verified Zenodo assets, the chosen TensorFlow stack, and a fixed ANARCI commit.
Compare the Biomodals operation against direct `humanise(...)` calls using:

- the repository example pairs plus explicit kappa and lambda fixtures;
- automatic and explicit heavy/light target families;
- already-human, ordinary non-human, and max-edit failure cases;
- default CDR preservation, opt-in CDR mutation, and fixed IMGT positions;
- one pair and multi-row CSVs, with separate validation tests for incomplete,
  noncanonical, duplicate-ID, and unnumberable inputs.

Require exact aligned/final sequences, target families, mutation locations,
edit counts, and derived success status. Compare CNN probabilities and germline-likeness
values at a tight declared tolerance rather than serialized bit identity.
Repeat warm CPU inference to establish whether exact discrete results remain
stable. The upstream repository contains examples but no automated tests or
golden outputs, so Biomodals must retain its own small oracle fixtures
([repository tree](https://github.com/oxpig/Humatch/tree/06205ad50f64b84468fea14eea573a3fce699550),
[example CSV](https://github.com/oxpig/Humatch/blob/06205ad50f64b84468fea14eea573a3fce699550/data/example.csv)).

## Smallest workflow-compatible Biomodals boundary

Expose one public **humanize** operation with the same wide `id,vh,vl` CSV
shape selected for Sapiens. Require complete, uniquely identified pairs and
fail the whole batch for invalid input, an unnumberable pair, or an execution
error. A below-target candidate remains a valid result with
`humanization_success=false`. Apply one set of controls to the complete batch:
automatic or explicit target HV/LV families, one shared GL target, separate
H/L/P classifier targets, a maximum combined edit count, one CDR mutation flag,
and separate fixed VH/VL IMGT positions. Keep the unused YAML `noise` field out
of the public interface and preserve its effective hardcoded value. Keep
alignment and raw classification internal unless a concrete consumer later
needs independent scoring.

The public defaults match upstream's effective defaults:

- `vh_target_family="auto"` and `vl_target_family="auto"`;
- `germline_likeness_target=0.40` for both chains;
- `vh_classifier_target=0.95`, `vl_classifier_target=0.95`, and
  `pair_classifier_target=0.95`;
- `max_edits=60`;
- `mutate_cdrs=false` for both chains and both phases; and
- empty `fixed_vh_positions` and `fixed_vl_positions`.

Validate with a Humatch-owned copy adapted from the small Sapiens validation
routine rather than importing the Sapiens app package. Bound input at 3 MiB,
1,000 pairs, 200 characters per ID, and 200 canonical residues per chain;
then require ANARCI to identify a complete VH plus kappa or lambda VL with the
IMGT coverage needed by Humatch. This deliberate duplication keeps the two
workflow-compatible apps independent.

Use a one-Node `ExecutionDefinition` for CLI/service runs and make the same
remote operation directly usable by a parent workflow. Return an inline
`.tar.zst` with these files:

- `input.csv` and `humanized.csv` use the common `id,vh,vl` schema;
- `humanized.fasta` contains the final pairs;
- `summary.csv` contains selected families, edit count, success, selected-target
  classifier endpoints, and H/L germline-likeness endpoints;
- `classifier_scores.csv` contains wide input/final class distributions;
- `alignment.csv` contains the long input-to-final IMGT comparison;
- `mutations.csv` contains only parental-to-final substitutions; and
- `manifest.json` contains parameters and scientific identities.

Do not patch upstream or emit an accepted-mutation trace in the first
implementation. To report the post-germline scores, call the deterministic
upstream germline helper separately, then pass the original parental pair to
`humanise()` so its edit count remains correct. Accept at most 1,000 pairs and
do not cache results. Initially run rows sequentially in one warm CPU container
with `cpu=(0.125, 16.125)`, `memory=(256, 16384)`, and a 24-hour timeout.
Benchmark one pair cold and warm, then pause before testing larger batches. Use
pair-level fanout when CPU utilization is at least 70% of the limit for at
least 70% of humanization; test in-container batching when utilization is at
most 40%, and compare both paths between those bounds.

## Initial implementation benchmark

Benchmark date: 2026-09-03

The initial implementation was tested on upstream's non-human mouse example
(the `is_human=0` row whose documented result has 24 edits) in the agreed
`cpu=(0.125, 16.125)`, `memory=(256, 16384)` Modal container. The image used the
pinned source, assets, and runtime listed above. The wrapper ran the complete
pair in one worker without pair fanout or in-container pair batching.

| Measurement | First call | Same-container warm call |
| --- | ---: | ---: |
| End-to-end worker time | 41.96 s | 35.90 s |
| Time inside upstream `humanise()` | 20.57 s | 23.46 s |
| Model load time | 0.22 s | 0.00 s |
| Mean CPU cores, complete worker | 2.12 | 2.47 |
| Mean CPU cores, `humanise()` | 2.55 | 2.46 |
| Mean CPU fraction of 16.125-core limit, complete worker | 13.2% | 15.3% |
| Mean CPU fraction of 16.125-core limit, `humanise()` | 15.8% | 15.2% |

A separate instrumented repeat took 46.20 seconds end to end and 25.71 seconds
inside `humanise()`. It observed 3,267.7 MiB of aggregate cgroup memory at
completion and a 2,776.9 MiB high-water RSS for the worker process. This Modal
cgroup exposes current aggregate memory but not an aggregate peak counter, so
the first figure is not claimed as the container's absolute peak.

The Biomodals result exactly matched the unmodified upstream CLI's final VH
and VL, `hv1`/`kv3` targets, and 24-edit count. Its full-precision scores round
to the upstream CLI values (`CNN_H=0.958`, `CNN_L=1.000`, `CNN_P=0.982`). First
and warm Biomodals calls returned identical sequences, mutations, selected
families, and full-precision classifier and germline-likeness scores. The
compressed eight-file result bundle was about 4.3 kB.

CPU use is well below the agreed 40% low-utilization boundary. These results
do not justify pair-per-container fanout: for a later performance iteration,
batching multiple pairs in one warm container is the more plausible first
experiment. The initial implementation deliberately remains sequential until
that optimization is requested.

## Place in the antibody-humanization stack

Sapiens and Humatch should produce alternative candidates from the same input,
not silently feed one method's mutations into the other. Sapiens supplies
per-residue 20-amino-acid probabilities from independent heavy/light masked
language models and changes every allowed position to its argmax. Humatch has
no analogous residue matrix; it contributes explicit V-family targeting,
gene-specific frequency priors, iterative mutation budgeting, and a joint
VH/VL pairing objective
([Sapiens paper/API discussion](https://pmc.ncbi.nlm.nih.gov/articles/PMC8837241/),
[Humatch paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11610552/)).

That complementarity makes Humatch scientifically useful even when Sapiens is
already present. The final workflow should preserve method identity and scores
so users can compare candidates. It should not interpret agreement as proof of
low immunogenicity or disagreement as an error.

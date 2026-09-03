# Sapiens app research

Research date: 2026-09-02

Implementation target: `src/biomodals/app/design/sapiens/app.py`

This note separates the small standalone `sapiens` Python package from the
larger BioPhi humanization workflow. That distinction is the main design
decision for a Biomodals app.

## Primary-source snapshot and reproducible pins

- The upstream repository's current `main` commit is
  [`3d676ecde0b6fc113d3f9c5bcb3721a7d70a85b7`](https://github.com/Merck/Sapiens/tree/3d676ecde0b6fc113d3f9c5bcb3721a7d70a85b7),
  dated 2026-01-22. The repository has no Git tags. Its code declares version
  `1.1.0`; later commits only update documentation. The latest PyPI release is
  likewise
  [`sapiens==1.1.0`](https://pypi.org/project/sapiens/1.1.0/), uploaded on
  2025-05-11; the wheel SHA-256 is
  `6ec1edc20c4126c99aad9f3e77dea24fc18c2383879874e47f91a5424d9fa221`.
- Version 1.1.0 is a scientific-runtime migration from bundled Fairseq
  checkpoints to Hugging Face Transformers and remotely hosted checkpoints;
  the migration commit is
  [`792e20118d0b217590b59292287cc1264acb6fe5`](https://github.com/Merck/Sapiens/commit/792e20118d0b217590b59292287cc1264acb6fe5).
- The exact current Hugging Face revisions are VH
  [`2caf2a813361918e9152347d119537e8a48bceb5`](https://huggingface.co/prihodad/biophi-sapiens1-vh/tree/2caf2a813361918e9152347d119537e8a48bceb5),
  VL
  [`4bb31b6841023645811d869cf0af9d1e0fec805e`](https://huggingface.co/prihodad/biophi-sapiens1-vl/tree/4bb31b6841023645811d869cf0af9d1e0fec805e),
  and tokenizer
  [`ef6ef1489be29a1cdb0a54f89b6dbf7a47c0ad40`](https://huggingface.co/prihodad/biophi-sapiens1-tokenizer/tree/ef6ef1489be29a1cdb0a54f89b6dbf7a47c0ad40).
  Upstream code names repositories without revisions, so an integration must
  add these revision pins itself to be reproducible
  ([loader source](https://github.com/Merck/Sapiens/blob/3d676ecde0b6fc113d3f9c5bcb3721a7d70a85b7/sapiens/predict.py)).
- Recommended starting pin: install `sapiens==1.1.0`, lock its otherwise
  unconstrained runtime dependencies, and materialize all three Hugging Face
  repositories at the revisions above. Pinning the Git head buys no code over
  the wheel and makes the build less conventional.

## Scientific purpose and terminology

Sapiens is a Transformer-encoder masked language model trained only on human
antibody variable-region sequences. Separate heavy- and light-chain models
were trained on subsets of 20 million heavy and 19 million light chains from
the Observed Antibody Space. Training perturbed residues by masking or random
mutation and trained the model to recover the original residue from sequence
context
([BioPhi paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8837241/),
[upstream README](https://github.com/Merck/Sapiens/blob/3d676ecde0b6fc113d3f9c5bcb3721a7d70a85b7/README.md)).

The model produces a position-by-amino-acid score matrix. The paper uses that
matrix to recognize and replace non-human-like residues in a non-human
parental antibody, but it explicitly frames this as an *in silico* primary
filter: binding, affinity, functional activity, and immunogenicity still
require experimental validation, and performance varied substantially among
benchmark antibodies
([BioPhi paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8837241/)). Sapiens is
not OASis: OASis is the separate 9-mer repertoire-search humanness metric used
to evaluate candidates independently
([BioPhi paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8837241/)).

The published BioPhi humanization algorithm performs one model pass, chooses
the highest-scoring canonical amino acid at every position, then restores the
parental CDRs. Repeating this operation produces `Sapiens*1`, `Sapiens*2`, and
so on. The paper benchmark used Kabat CDR preservation by default and discusses
IMGT, Chothia, and North alternatives in BioPhi
([paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8837241/),
[BioPhi implementation](https://github.com/Merck/BioPhi/blob/bc59cd17b690a634553ac50a60840b9a89bd21b0/biophi/humanization/methods/humanization.py)).

## Supported inputs

- The standalone API accepts one unpaired antibody **variable-region amino
  acid sequence** plus an explicit chain type. `H` selects the heavy model;
  `K` and `L` both select the same light model. There is no paired-chain
  conditioning and no species parameter
  ([API source](https://github.com/Merck/Sapiens/blob/3d676ecde0b6fc113d3f9c5bcb3721a7d70a85b7/sapiens/predict.py)).
- Training data are human, but the intended humanization input is a non-human
  parental antibody; the paper evaluates murine and inferred parental
  sequences. This does not establish special support for every source species
  or for camelid VHHs
  ([paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8837241/)).
- The tokenizer vocabulary contains the 20 canonical uppercase amino acids and
  five RoBERTa special tokens, with no lowercase or ambiguous residue tokens
  ([pinned vocabulary](https://huggingface.co/prihodad/biophi-sapiens1-tokenizer/blob/ef6ef1489be29a1cdb0a54f89b6dbf7a47c0ad40/vocab.json)).
  Upstream nevertheless documents `X` and `*` as infill markers. They tokenize
  as unknown tokens, and `predict_masked` replaces only positions whose
  original character is exactly `X` or `*`
  ([README example](https://github.com/Merck/Sapiens/blob/3d676ecde0b6fc113d3f9c5bcb3721a7d70a85b7/README.md),
  [implementation](https://github.com/Merck/Sapiens/blob/3d676ecde0b6fc113d3f9c5bcb3721a7d70a85b7/sapiens/predict.py)).
- The VH configuration has 146 positional embeddings and the VL configuration
  has 130; each uses four encoder layers, eight attention heads, hidden size
  128, intermediate size 256, vocabulary size 25, and FP32 weights
  ([VH config](https://huggingface.co/prihodad/biophi-sapiens1-vh/blob/2caf2a813361918e9152347d119537e8a48bceb5/config.json),
  [VL config](https://huggingface.co/prihodad/biophi-sapiens1-vl/blob/4bb31b6841023645811d869cf0af9d1e0fec805e/config.json)).
  Because RoBERTa adds boundary tokens and offset position IDs while the
  tokenizer advertises no useful maximum, the effective raw limits are
  inferred to be 142 VH residues and 126 VL residues. The app should enforce
  and oracle-test these boundaries instead of allowing an opaque embedding
  index error.
- The Sapiens package has no file parser or CLI. Its public interface is Python
  functions over one sequence. BioPhi adds FASTA input and infers heavy versus
  light chain with AbNumber; paired chains conventionally share an ID with
  optional `_VH`/`_VL` or `_HC`/`_LC` suffixes
  ([Sapiens package source](https://github.com/Merck/Sapiens/blob/3d676ecde0b6fc113d3f9c5bcb3721a7d70a85b7/sapiens/predict.py),
  [BioPhi CLI documentation](https://github.com/Merck/BioPhi/blob/bc59cd17b690a634553ac50a60840b9a89bd21b0/README.md#biophi-command-line-interface)).

For a Biomodals file interface, accept bounded UTF-8 FASTA, require unique
nonempty IDs, require uppercase canonical residues except `X`/`*` in infill
mode, require a chain type per record (or make chain inference an explicit
feature), and reject sequences above the chain-specific model limit before the
remote call. Do not silently pass whitespace, lowercase, gaps, stop symbols,
or arbitrary tokenizer unknowns.

## Standalone Python API and exact output semantics

The exported API is
[`predict_scores`, `predict_best_score`, `predict_masked`,
`predict_residue_embedding`, and `predict_sequence_embedding`](https://github.com/Merck/Sapiens/blob/3d676ecde0b6fc113d3f9c5bcb3721a7d70a85b7/sapiens/__init__.py).

`predict_scores(seq, chain_type, checkpoint_path=None, probs=True,
return_embeddings=False, layer=None, tokenizer_path=...)` returns a pandas
DataFrame with one row per input character and canonical amino-acid columns in
`ACDEFGHIKLMNPQRSTVWY` order. With `probs=True`, softmax is computed across all
25 vocabulary tokens and then only the 20 amino-acid columns are retained, so
the returned row is not explicitly renormalized over those 20 columns. With
`probs=False`, the same columns contain logits
([implementation](https://github.com/Merck/Sapiens/blob/3d676ecde0b6fc113d3f9c5bcb3721a7d70a85b7/sapiens/predict.py)).

`predict_best_score` chooses the per-position argmax among those 20 columns.
`predict_masked` uses those argmax residues only at original `X` or `*`
positions and preserves every other original character
([implementation](https://github.com/Merck/Sapiens/blob/3d676ecde0b6fc113d3f9c5bcb3721a7d70a85b7/sapiens/predict.py)).
This masked infilling function is not the paper's full humanization procedure:
unmasked non-human residues are never changed.

Residue embeddings have shape `(5, sequence_length, 128)` when `layer=None`:
the embedding output plus four encoder-layer hidden states. Selecting one layer
returns `(sequence_length, 128)`. Sequence embeddings are the arithmetic mean
over residue positions and therefore have shape `(5, 128)` or `(128,)`
([implementation](https://github.com/Merck/Sapiens/blob/3d676ecde0b6fc113d3f9c5bcb3721a7d70a85b7/sapiens/predict.py),
[model configuration](https://huggingface.co/prihodad/biophi-sapiens1-vh/blob/2caf2a813361918e9152347d119537e8a48bceb5/config.json)).

BioPhi exposes three useful artifact precedents: humanized FASTA, a long CSV
containing ID/chain/input residue plus the 20 position scores, and a per-chain
CSV containing the mean probability assigned to the observed residues
([BioPhi CLI implementation](https://github.com/Merck/BioPhi/blob/bc59cd17b690a634553ac50a60840b9a89bd21b0/biophi/humanization/cli/sapiens.py)).
The mean observed-residue probability is a sequence-level summary, not the
independent OASis humanness score.

The resolved Biomodals artifacts are a normalized paired CSV, paired humanized
CSV, humanized FASTA, mutation-history CSV, endpoint residue-score Parquet, and
a scientific-identity manifest. The app uses Polars for app-owned tabular
serialization even though the upstream API internally returns pandas.

## Installation, execution, and model caching

The package metadata claims Python `>=3.7`, recommends Python 3.10, and leaves
`pandas`, `transformers`, and `torch` entirely unbounded
([setup.py](https://github.com/Merck/Sapiens/blob/3d676ecde0b6fc113d3f9c5bcb3721a7d70a85b7/setup.py),
[README](https://github.com/Merck/Sapiens/blob/3d676ecde0b6fc113d3f9c5bcb3721a7d70a85b7/README.md#usage)).
Upstream CI still selects Python 3.7 and uses old checkout/setup-python actions,
so metadata and CI do not establish Python 3.12 compatibility
([workflow](https://github.com/Merck/Sapiens/blob/3d676ecde0b6fc113d3f9c5bcb3721a7d70a85b7/.github/workflows/python-package-conda.yml)).
The checkpoint configs record Transformers 4.44.2. A Biomodals image should
pin a tested Python 3.12, Torch, Transformers, pandas, and Sapiens combination
rather than inherit current transitive releases.

`from_pretrained` downloads the selected model and tokenizer into Hugging
Face's normal cache on first use; process-global dictionaries then cache loaded
objects by the caller-supplied key. The API provides no revision argument and
does not use a Biomodals/Modal Volume
([loader implementation](https://github.com/Merck/Sapiens/blob/3d676ecde0b6fc113d3f9c5bcb3721a7d70a85b7/sapiens/predict.py)).
The VH and VL repositories contain approximately 569k and 567k FP32 parameters
respectively, about 2.3 MB of stored model data each
([VH model](https://huggingface.co/prihodad/biophi-sapiens1-vh),
[VL model](https://huggingface.co/prihodad/biophi-sapiens1-vl)).

Because the immutable assets are tiny, the simplest reproducible deployment is
to download snapshots at fixed revisions during image construction and run
with local paths and offline Hugging Face settings. A model Volume plus an
explicit setup function is also compatible with repository convention, but is
more lifecycle machinery for under 5 MB. This is a design choice for the grill.

## CPU/GPU, batching, memory, and determinism

Upstream `predict_scores` handles exactly one sequence, does not pad or batch,
does not move the model or inputs to a device, executes under `torch.no_grad`,
and copies logits and selected hidden states to CPU
([source](https://github.com/Merck/Sapiens/blob/3d676ecde0b6fc113d3f9c5bcb3721a7d70a85b7/sapiens/predict.py)).
Therefore the supported upstream path is CPU inference. The models are very
small; a GPU is unlikely to justify container scheduling overhead for
single-sequence calls. BioPhi parallelizes whole antibodies with a
multiprocessing pool rather than batching tensors
([BioPhi CLI](https://github.com/Merck/BioPhi/blob/bc59cd17b690a634553ac50a60840b9a89bd21b0/biophi/humanization/cli/sapiens.py)).

Recommended v1 execution is one CPU Modal function per bounded FASTA file,
loading both models once per warm container and iterating records. Start with a
small CPU/memory request and benchmark locally/Modal before adding fanout or
GPU. A native padded batch implementation would be an intentional execution
fork and must be checked against pinned upstream per-sequence outputs before an
equivalence claim.

Inference contains no sampling and pretrained Transformers models load in eval
mode, so the algorithm is expected to be deterministic for a pinned software,
checkpoint, and device stack. Cross-version/device floating-point differences
can still change close argmax ties. Record the package, dependency, tokenizer,
and checkpoint identities in outputs and pin CPU inference for the reference
oracle.

## Licenses and commercial caveats

The Sapiens code is MIT licensed by Merck Sharp & Dohme Corp.
([license](https://github.com/Merck/Sapiens/blob/3d676ecde0b6fc113d3f9c5bcb3721a7d70a85b7/LICENSE)).
All three Hugging Face repositories declare MIT, although their model cards are
otherwise empty
([VH](https://huggingface.co/prihodad/biophi-sapiens1-vh),
[VL](https://huggingface.co/prihodad/biophi-sapiens1-vl),
[tokenizer](https://huggingface.co/prihodad/biophi-sapiens1-tokenizer)).
The paper is published under CC BY-NC, but that license applies to the article,
not the separately MIT-licensed code and model repositories
([paper copyright notice](https://pmc.ncbi.nlm.nih.gov/articles/PMC8837241/)).
The MIT materials permit commercial use but provide no warranty; deployment
should retain required notices. This note is not legal advice.

## Upstream hazards and integration consequences

1. **“Sapiens app” is underspecified.** The standalone package offers raw
   scores, masked infilling, and embeddings. The user-facing paper/BioPhi
   humanization adds AbNumber, chain inference, multiple iterations, CDR
   definitions, and parental-CDR restoration
   ([package API](https://github.com/Merck/Sapiens/blob/3d676ecde0b6fc113d3f9c5bcb3721a7d70a85b7/sapiens/predict.py),
   [BioPhi algorithm](https://github.com/Merck/BioPhi/blob/bc59cd17b690a634553ac50a60840b9a89bd21b0/biophi/humanization/methods/humanization.py)).
   This must be settled before implementation.
2. **Remote assets float.** Default model IDs and tokenizer ID omit immutable
   revisions. Download pinned snapshots, validate expected files, and use
   local-only inference.
3. **Dependency surface floats.** All three dependencies are unbounded. Pin the
   full inference stack and test it on the repository's Python 3.12 runtime.
4. **Upstream tests are stale.** The test suite still passes the removed
   `model_version=` keyword to `predict_scores`, so it does not test 1.1.0 as
   written
   ([test](https://github.com/Merck/Sapiens/blob/3d676ecde0b6fc113d3f9c5bcb3721a7d70a85b7/tests/test_predict.py),
   [new signature](https://github.com/Merck/Sapiens/blob/3d676ecde0b6fc113d3f9c5bcb3721a7d70a85b7/sapiens/predict.py)).
   Biomodals needs its own pinned oracle fixtures.
5. **No upstream validation.** Arbitrary strings can become unknown tokenizer
   tokens; model-length overflow is left to Transformers; lowercase is not a
   documented equivalent. Validate twice at the local and remote trust
   boundaries.
6. **No native batch API.** A Python loop preserves upstream behavior. Vectorized
   padding is reasonable only after a direct oracle comparison including both
   chains, `X`/`*`, boundary lengths, scores, argmax sequences, and embeddings.
7. **Scores need precise labels.** Per-residue outputs are model probabilities
   over a vocabulary subset, not calibrated immunogenicity risk and not OASis
   humanness. The app must avoid calling the mean score “immunogenicity.”
8. **Full-humanization CDR behavior changes biological risk.** Mutating CDRs can
   affect antigen binding; BioPhi disables it by default
   ([BioPhi option and warning](https://github.com/Merck/BioPhi/blob/bc59cd17b690a634553ac50a60840b9a89bd21b0/biophi/humanization/cli/sapiens.py)).
   Preserve parental CDRs by default if the full humanization workflow is in
   scope.

## Resolved app design

The design grill resolved the first app as a goal-focused BioPhi-compatible
humanization operation:

- Accept a UTF-8 CSV with exactly `id`, `vh`, and `vl` columns. Require complete
  VH-VL pairs, unique nonempty IDs, uppercase canonical amino acids, the
  chain-specific model length limits, and at most 1,000 pairs. Reject the whole
  batch before submission when any row is invalid.
- Perform one through five explicit Humanization Iterations, defaulting to one.
  Each pass applies BioPhi's per-position canonical-amino-acid argmax policy.
- Expose numbering scheme and CDR definition independently. Support `kabat`,
  `chothia`, and `imgt` numbering and `kabat`, `chothia`, `imgt`, and `north`
  CDR definitions, with both defaults set to `kabat`. Preserve parental CDRs by
  default and permit their mutation only through an explicit opt-in flag.
- Always return the parental-input and final-humanized 20-amino-acid residue
  score matrices. Intermediate iteration scores are transient. Include the
  paired humanized sequences, FASTA, mutation history, and scientific identity
  manifest in one inline `.tar.zst` bundle written with Polars and orjson where
  applicable.
- Bake the exact VH, VL, and tokenizer revisions into the Modal image and force
  offline loading. Run sequentially in one CPU-only container first; do not add
  app-owned result caching, GPU execution, tensor batching, or Task fanout
  without benchmark evidence.
- Provide both a production standalone Local Entrypoint backed by the shared
  execution kernel and a workflow-compatible app function. The later
  `feat/antibody-humanization` branch will compose that operation rather than
  start a nested app coordinator.
- Implement the narrow algorithm using pinned Sapiens and antibody-numbering
  dependencies, not the full BioPhi or OASis runtime. Require exact agreement
  with the pinned BioPhi oracle for sequences, mutations, numbering, and CDR
  classification, and tight numerical agreement for complete score matrices.
- Characterize one published VH-VL pair under cold and warm execution, then
  warm batches of 10 and 100 pairs, starting with two CPUs and 2 GiB of memory.
  Treat timings as evidence for later tuning rather than a performance gate.

The app does not expose masked infilling, embeddings, scores without
humanization, OASis evaluation, partial-batch results, app-owned result caching,
or a separate large-batch execution path.

## Implementation benchmark and oracle check

The implemented image uses Python 3.12, the BioPhi-pinned `abnumber==0.3.2`
and `anarci==2020.04.23`, CPU `torch==2.6.0`, and
`transformers==4.53.3`. The immutable model snapshots are baked into the image
and loaded through local paths with Hugging Face offline mode enabled.

On 2026-09-02, development runs on one Modal container with 2 CPUs and 2 GiB
of memory completed the full operation, including numbering, humanization,
both endpoint matrices, and archive creation, in:

| Complete VH-VL pairs | Worker time | Amortized time per pair |
| ---: | ---: | ---: |
| 1 | 4.861 s | 4.861 s |
| 10 | 6.749 s | 0.675 s |
| 100 | 18.420 s | 0.184 s |

The 10- and 100-pair inputs intentionally repeated one published pair under
unique IDs to isolate warm-model throughput. They are suitable for timing but
not representative of compression ratios or biological sequence diversity.
Both models load during the first pair; later pairs reuse the process-global
Sapiens caches. This supports a sequential CPU v1 for the expected workload of
fewer than 100 pairs.

For the heavy-chain sequence in BioPhi's pinned `test_sapiens_humanize`
fixture, the app produced the exact expected one-iteration Kabat/CDR-preserved
sequence:

```text
QVQLVQSGAEVKKPGASVKVSCKASGYTFTNYYMYWVRQAPGQGLEWMGGINPSNGGTNFNEKFKNRVTLTTDTSTTTAYMELRSLRSDDTAVYYCARRDYRFDMGFDYWGQGTLVTVSS
```

The bundle's score table was also checked to contain exactly the input and
final endpoints for both chains, with all 20 canonical amino-acid columns.
This establishes the discrete published fixture and output contract. A future
dependency change must additionally compare the floating-point matrices to a
pinned oracle within an explicit tolerance before updating scientific
identity.

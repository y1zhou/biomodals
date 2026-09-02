# p-AbNatiV2 app research

Research date: 2026-09-02

Prospective implementation target: `src/biomodals/app/design/pabnativ2_app.py`

## Executive recommendation

p-AbNatiV2 is a strong candidate for a **paired** humanization app because its
heavy-chain reconstruction is conditioned on the light chain and vice versa,
and it exposes a separate learned pairing score. That is meaningfully different
from applying two independent chain models. The first app should stay narrow:
accept complete VH/VL pairs, run the upstream paired humanizer, and return the
input and final sequences with paired, chain, region, residue, and pairing
scores. Do not include unpaired AbNatiV, VHH humanization, training, plotting,
or custom checkpoints in the same app
([paired model](https://gitlab.doc.ic.ac.uk/sormanni-lab/abnativ/-/blob/eb517f1f0b947084cb7e44a54ef34103e9692f5e/abnativ/model/abnativ2_paired.py),
[paired humanizer](https://gitlab.doc.ic.ac.uk/sormanni-lab/abnativ/-/blob/eb517f1f0b947084cb7e44a54ef34103e9692f5e/abnativ/humanisation/vh_vl_humanisation_functions.py)).

There are two blockers to resolve before treating this as a generally available
Biomodals app:

1. The code is CC BY-NC-SA 4.0 and the README expressly prohibits commercial
   use. Commercial or mixed-use deployment needs permission from the authors or
   a legal determination; this note is not legal advice
   ([license](https://gitlab.doc.ic.ac.uk/sormanni-lab/abnativ/-/blob/eb517f1f0b947084cb7e44a54ef34103e9692f5e/LICENSE),
   [README](https://gitlab.doc.ic.ac.uk/sormanni-lab/abnativ/-/blob/eb517f1f0b947084cb7e44a54ef34103e9692f5e/README.md)).
2. The peer-reviewed method averages solvent accessibility over ten predicted
   structures, while the current source hard-codes four predictions. The app
   must deliberately choose and version one behavior after checking with
   upstream; it must not claim both current-code and paper equivalence
   ([paper](https://doi.org/10.1080/19420862.2026.2646361),
   [current RASA call](https://gitlab.doc.ic.ac.uk/sormanni-lab/abnativ/-/blob/eb517f1f0b947084cb7e44a54ef34103e9692f5e/abnativ/humanisation/humanisation_utils.py)).

## Reproducible upstream snapshot

- The current `main` commit is
  [`eb517f1f0b947084cb7e44a54ef34103e9692f5e`](https://gitlab.doc.ic.ac.uk/sormanni-lab/abnativ/-/tree/eb517f1f0b947084cb7e44a54ef34103e9692f5e),
  dated 2026-05-31. It declares package version `2.0.8`; the most recent Git tag
  is only `1.2.0`, so the tag is not an adequate p-AbNatiV2 pin
  ([package metadata](https://gitlab.doc.ic.ac.uk/sormanni-lab/abnativ/-/blob/eb517f1f0b947084cb7e44a54ef34103e9692f5e/pyproject.toml)).
- The matching distribution is
  [`abnativ==2.0.8`](https://pypi.org/project/abnativ/2.0.8/), uploaded on
  2026-05-31. PyPI provides a 30.1 MB wheel with SHA-256
  `1345b2d27d5f5edd13d80e1944f11bee8c53aef4030f190d9fc4ee5de3b743b1`.
- Recommended code pin: the 2.0.8 wheel plus its hash. Also record the Git
  commit above because the scientific oracle should be inspectable at a fixed
  source revision.
- The peer-reviewed AbNatiV2 paper was published in 2026, although the upstream
  README still labels its link as a preprint
  ([paper](https://doi.org/10.1080/19420862.2026.2646361),
  [README references](https://gitlab.doc.ic.ac.uk/sormanni-lab/abnativ/-/blob/eb517f1f0b947084cb7e44a54ef34103e9692f5e/README.md)).

## Scientific model and training data

AbNatiV2 is a vector-quantized variational autoencoder. p-AbNatiV2 starts from
the updated heavy- and light-chain networks and adds bidirectional cross-chain
attention, so each chain's reconstruction depends on the other chain. It also
adds a logistic pairing head over learned heavy/light representations
([paper](https://doi.org/10.1080/19420862.2026.2646361),
[cross-attention implementation](https://gitlab.doc.ic.ac.uk/sormanni-lab/abnativ/-/blob/eb517f1f0b947084cb7e44a54ef34103e9692f5e/abnativ/model/abnativ2_paired.py)).

The paper reports fine-tuning the paired model on 3,745,614 unique natively
paired human antibodies assembled from p-IgGen and PairedAbNGS/OAS sources.
The pairing head uses noise-contrastive training: native pairs are positives;
shuffled partners and mismatched class, species, or study pairs provide
synthetic negatives. Those negatives are useful supervision, but they are not
experimental evidence that a particular alternative pair cannot fold or bind
([paper](https://doi.org/10.1080/19420862.2026.2646361)). Training and test
archives are separately published on
[Zenodo](https://zenodo.org/records/17466150); they are not needed for
inference.

The scientific identity of an app run therefore includes at least the code
version, paired checkpoint checksum, dependency lock, AHo alignment behavior,
humanization parameters, structure-prediction checkpoint, and device class.

## Input contract and alignment

The upstream paired scorer accepts either a pandas DataFrame with ID, VH, and
VL columns or one VH/VL pair through the CLI. Unaligned variable-region amino
acid sequences are converted by ANARCI into two fixed 149-position AHo
alignments; disabling alignment requires each supplied chain to be exactly 149
characters including gaps
([scoring API](https://gitlab.doc.ic.ac.uk/sormanni-lab/abnativ/-/blob/eb517f1f0b947084cb7e44a54ef34103e9692f5e/abnativ/model/scoring_functions.py),
[CLI documentation](https://gitlab.doc.ic.ac.uk/sormanni-lab/abnativ/-/blob/eb517f1f0b947084cb7e44a54ef34103e9692f5e/README.md)).

Upstream's one-hot alphabet is the 20 canonical amino acids plus gap. `X` is
represented as a uniform vector in profile output rather than rejected
([encoder](https://gitlab.doc.ic.ac.uk/sormanni-lab/abnativ/-/blob/eb517f1f0b947084cb7e44a54ef34103e9692f5e/abnativ/model/onehotencoder.py),
[profile generation](https://gitlab.doc.ic.ac.uk/sormanni-lab/abnativ/-/blob/eb517f1f0b947084cb7e44a54ef34103e9692f5e/abnativ/model/scoring_functions.py)).
That permissiveness is undesirable at the Biomodals boundary because an
ambiguous residue has no single observed-residue score.

Recommended Biomodals input is a UTF-8 wide CSV with exactly one complete pair
per row and canonical columns `id,vh,vl`. Require:

- 1 to 1,000 rows, a unique nonempty ID in every row, and both chains present;
- uppercase canonical amino acids only in unaligned user input;
- successful antibody numbering and explicit confirmation that `vh` is heavy
  and `vl` is kappa or lambda light;
- bounded file bytes and bounded sequence lengths in addition to the row cap;
  and
- all-or-nothing validation before scheduling expensive inference.

These checks close concrete upstream hazards. During alignment, the scorer
collects every recognized H/K/L chain from each nominal input column, joins by
ID through dictionaries, silently omits IDs missing on either side, and lets a
duplicate ID overwrite an earlier sequence. It does not itself enforce that
the nominal VH column classified as heavy or the nominal VL column classified
as light
([alignment block](https://gitlab.doc.ic.ac.uk/sormanni-lab/abnativ/-/blob/eb517f1f0b947084cb7e44a54ef34103e9692f5e/abnativ/model/scoring_functions.py)).

## Score semantics

### Humanness and residue reconstruction

For AbNatiV2, the implementation computes mean per-position reconstruction MSE
over nongap positions, transforms it as `exp(-mean_mse)`, then linearly rescales
the empirical human/non-human decision threshold to 0.8. The paired thresholds
are 0.991817 for the joint pair, 0.992019 for heavy, and 0.993152 for light
before rescaling
([score implementation](https://gitlab.doc.ic.ac.uk/sormanni-lab/abnativ/-/blob/eb517f1f0b947084cb7e44a54ef34103e9692f5e/abnativ/model/scoring_functions.py)).
Consequences for labels and schemas:

- This is a model-derived **nativeness/humanness score**, not a calibrated
  probability of being human and not an immunogenicity probability.
- Because the final operation is affine rescaling, individual regional values
  can be below zero; the checked-in example output contains such values. Do not
  enforce a `[0,1]` result range
  ([example output](https://gitlab.doc.ic.ac.uk/sormanni-lab/abnativ/-/blob/eb517f1f0b947084cb7e44a54ef34103e9692f5e/test/test_paired_scoring/test_paired_abnativ_seq_scores.csv)).
- A residue score is `exp(-position_mse)`. The accompanying 21 reconstruction
  values are the model distribution over 20 residues plus gap, not alternative
  experimentally measured fitness effects
  ([profile implementation](https://gitlab.doc.ic.ac.uk/sormanni-lab/abnativ/-/blob/eb517f1f0b947084cb7e44a54ef34103e9692f5e/abnativ/model/scoring_functions.py)).
- Percentiles are integer empirical percentiles computed against bundled test
  score tables. They are reference-distribution ranks, not confidence
  intervals
  ([percentile implementation](https://gitlab.doc.ic.ac.uk/sormanni-lab/abnativ/-/blob/eb517f1f0b947084cb7e44a54ef34103e9692f5e/abnativ/model/scoring_functions.py)).

The sequence-level paired scorer returns joint heavy-light, heavy-only,
light-only, CDR1/2/3, and framework scores. It adds bundled-test percentiles for
the joint score and each chain's CDR/framework regions, but not separate
whole-heavy or whole-light percentiles. Position-level output concatenates 149
heavy and 149 light AHo positions and includes observed residue, residue score,
and all 21 reconstruction values
([scoring function](https://gitlab.doc.ic.ac.uk/sormanni-lab/abnativ/-/blob/eb517f1f0b947084cb7e44a54ef34103e9692f5e/abnativ/model/scoring_functions.py)).

### Pairing head

The pairing head output is a sigmoid in `[0,1]` trained to discriminate native
pairs from the synthetic negative construction above
([model implementation](https://gitlab.doc.ic.ac.uk/sormanni-lab/abnativ/-/blob/eb517f1f0b947084cb7e44a54ef34103e9692f5e/abnativ/model/abnativ2_paired.py),
[paper](https://doi.org/10.1080/19420862.2026.2646361)). Upstream stores the raw
value in a column named `AbNatiV Pairing Score (%)` without multiplying by 100
([scorer](https://gitlab.doc.ic.ac.uk/sormanni-lab/abnativ/-/blob/eb517f1f0b947084cb7e44a54ef34103e9692f5e/abnativ/model/scoring_functions.py)).
The Biomodals schema should correct the unit ambiguity by naming it
`pairing_score` and documenting raw fraction units. Do not call it a calibrated
probability of physical assembly, expression, affinity, or efficacy.

## Paired humanization procedure

The public `abnativ_vh_vl_humanisation_paired` API accepts exactly one VH/VL
pair. Its default search:

1. aligns both chains to 149-position AHo coordinates;
2. estimates mutational dependence with an in-silico deep-mutational scan over
   the joint pair;
3. limits candidates to permitted AHo positions, by default framework
   positions, intersected with residues whose predicted relative solvent
   accessibility is at least 0.15;
4. proposes residues from human heavy and kappa/lambda PSSMs above a 1%
   frequency cutoff while excluding C, M, and gap;
5. greedily tries liabilities ordered by the dependence estimate, maximizing a
   weighted change in joint humanness and pairing score while rejecting excess
   pairing-score decreases; and
6. re-scores the input and final pair and predicts both structures for
   scaffold/CDR displacement reporting
   ([paired humanizer](https://gitlab.doc.ic.ac.uk/sormanni-lab/abnativ/-/blob/eb517f1f0b947084cb7e44a54ef34103e9692f5e/abnativ/humanisation/vh_vl_humanisation_functions.py),
   [search utilities](https://gitlab.doc.ic.ac.uk/sormanni-lab/abnativ/-/blob/eb517f1f0b947084cb7e44a54ef34103e9692f5e/abnativ/humanisation/humanisation_utils.py)).

The source defaults are residue threshold 0.98, RASA threshold 0.15, maximum
pairing decrease 0.1, objective weights `a=10,b=1`, and forbidden substitutions
`C,M`. Only the enhanced greedy paired strategy is exposed; there is no
exhaustive paired mode
([API and CLI](https://gitlab.doc.ic.ac.uk/sormanni-lab/abnativ/-/blob/eb517f1f0b947084cb7e44a54ef34103e9692f5e/abnativ/humanisation/vh_vl_humanisation_functions.py),
[README](https://gitlab.doc.ic.ac.uk/sormanni-lab/abnativ/-/blob/eb517f1f0b947084cb7e44a54ef34103e9692f5e/README.md)).

If no experimental PDB is supplied, current code uses ABodyBuilder3 four times
to average solvent accessibility. Even setting the RASA threshold to zero only
skips the candidate-selection structures: the outer pipeline still predicts
the input and final structures and computes CDR displacement
([RASA implementation](https://gitlab.doc.ic.ac.uk/sormanni-lab/abnativ/-/blob/eb517f1f0b947084cb7e44a54ef34103e9692f5e/abnativ/humanisation/humanisation_utils.py),
[finalization](https://gitlab.doc.ic.ac.uk/sormanni-lab/abnativ/-/blob/eb517f1f0b947084cb7e44a54ef34103e9692f5e/abnativ/humanisation/vh_vl_humanisation_functions.py)).

Upstream writes PDBs, PAP alignments, DMS tables, plots, CSV files, and a
ChimeraX session, while the Python function returns only a summary DataFrame
for input and final pairs. A Biomodals wrapper should treat those writes as
scratch intermediates and deliberately package only stable scientific outputs
instead of exposing the incidental upstream directory tree.

## Models, installation, and resource footprint

`abnativ init` downloads nine AbNatiV checkpoints plus an ABodyBuilder3
checkpoint. The nine AbNatiV files total about 5.84 GB, even though paired
scoring requires only `vpaired2_model.ckpt`
([download implementation](https://gitlab.doc.ic.ac.uk/sormanni-lab/abnativ/-/blob/eb517f1f0b947084cb7e44a54ef34103e9692f5e/abnativ/init.py),
[AbNatiV model record](https://zenodo.org/records/17295347)). For the narrow
paired app:

- `vpaired2_model.ckpt` is 1,223,267,160 bytes with MD5
  `d9cd90dc5b1a71873720c77c3bfe0906`. The model record is CC BY-SA 4.0
  ([Zenodo metadata](https://zenodo.org/records/17295347)).
- Paired humanization also extracts
  `plddt-loss/best_second_stage.ckpt` from ABodyBuilder3's 440,793,236-byte
  `output.tar.gz`, MD5 `2c1d734ed74013865bd95fc2f95cff1e`.
  That record is CC BY 4.0
  ([ABodyBuilder3 record](https://zenodo.org/records/11354577),
  [extraction code](https://gitlab.doc.ic.ac.uk/sormanni-lab/abnativ/-/blob/eb517f1f0b947084cb7e44a54ef34103e9692f5e/abnativ/init.py)).
- The wheel embeds small PSSMs and much larger percentile reference tables;
  the paired percentile CSV is about 57.7 MB uncompressed. These are already
  compressed into the 30.1 MB wheel
  ([package data declaration](https://gitlab.doc.ic.ac.uk/sormanni-lab/abnativ/-/blob/eb517f1f0b947084cb7e44a54ef34103e9692f5e/pyproject.toml)).
- Upstream download code uses `wget` and checks only whether a file exists; it
  does not verify published checksums. The app build/setup path must perform
  verification and fail closed
  ([initializer](https://gitlab.doc.ic.ac.uk/sormanni-lab/abnativ/-/blob/eb517f1f0b947084cb7e44a54ef34103e9692f5e/abnativ/init.py)).

The repository claims Python 3.12 compatibility and recommends a GPU, but its
environment leaves nearly every Python dependency unbounded. It requires
ANARCI/HMMER, OpenMM/PDBFixer, ABodyBuilder3, FreeSASA, Torch,
PyTorch-Lightning, ProteinTopModel, and a broad scientific/plotting stack; the
README even applies source edits to ABodyBuilder3 before installation
([environment](https://gitlab.doc.ic.ac.uk/sormanni-lab/abnativ/-/blob/eb517f1f0b947084cb7e44a54ef34103e9692f5e/environment.yml),
[installation instructions](https://gitlab.doc.ic.ac.uk/sormanni-lab/abnativ/-/blob/eb517f1f0b947084cb7e44a54ef34103e9692f5e/README.md)).
Pin the complete tested runtime and make any upstream patch explicit and
auditable.

The paired checkpoint plus structure checkpoint transfer is at least 1.66 GB,
before dependencies and extracted assets. Prefer an immutable, checksum-verified
model layer or a staged read-only Modal Volume; do not run the broad upstream
`abnativ init`. Which cache mechanism gives better cold-start behavior should be
decided from a one-pair benchmark.

## Runtime, batching, and determinism

Paired scoring is natively batched (`batch_size=256` in the Python API; the CLI
example uses 128), and alignment can use multiple CPU processes. The scorer
automatically selects CUDA, then Apple MPS, then CPU
([scorer](https://gitlab.doc.ic.ac.uk/sormanni-lab/abnativ/-/blob/eb517f1f0b947084cb7e44a54ef34103e9692f5e/abnativ/model/scoring_functions.py)).
Humanization, however, accepts one pair and repeatedly invokes pair scoring,
DMS, and structure prediction. There is no published one-pair runtime or memory
requirement in the paper, so resource sizing must be measured rather than
inferred.

Start with a single-pair benchmark on a CPU container as requested, capturing
image build time, cold start, asset load, alignment, DMS, each structure pass,
total wall time, peak RSS, and output bytes. Also run the same oracle on the
smallest practical GPU. If single-pair work is long, fan out one pair per
execution-kernel task for a batch while preserving the overall 1,000-pair hard
ceiling. Do not introduce an unbounded `.map()` path.

Pure scoring uses `eval()` plus inference mode and contains no sampling, so it
should be deterministic for a pinned checkpoint, runtime, and device. The full
humanizer has weaker guarantees:

- structure prediction is repeated but the paired humanizer does not expose a
  seed parameter;
- candidate-position intersection uses Python sets before ordering;
- equal-valued mutation ties can depend on candidate iteration order; and
- `forbidden_mut` is a mutable default and is modified in place by appending
  gap, so repeated calls in one warm process can observe changed state
  ([search implementation](https://gitlab.doc.ic.ac.uk/sormanni-lab/abnativ/-/blob/eb517f1f0b947084cb7e44a54ef34103e9692f5e/abnativ/humanisation/humanisation_utils.py)).

The wrapper should pass a fresh forbidden list, sort every set-derived
collection, seed all seedable libraries, record device/software identity, and
test repeated runs in the same warm process. Cross-device floating-point
differences may still change a threshold or greedy tie; CPU and GPU should be
treated as distinct scientific execution identities until shown equivalent.

## Published validation and limits

The paper reports 96.9% reconstruction accuracy for p-AbNatiV2 and strong
separation of human paired sequences from mouse, rat, and artificial antibody
sets. For the learned pairing head, a native partner outranked one matched
random partner in 74.3% of 10,000 comparisons, and the native partner ranked in
the top five among 51 candidates for more than 35% of tested heavy chains; the
paper reports corresponding HuMatch baseline figures of 60.1% and 18%
([paper](https://doi.org/10.1080/19420862.2026.2646361)). These are ranking
benchmarks against constructed negatives, not prospective laboratory pairing
validation.

On a heterogeneous clinical anti-drug-antibody dataset, the paper reports a
correlation around `r=-0.51` between paired AbNatiV2 humanness and ADA rate. The
authors caution that the clinical studies vary in assay, patient population,
dosing, indication, and other factors. Nativeness is therefore a useful design
and triage signal, not a clinical immunogenicity prediction
([paper](https://doi.org/10.1080/19420862.2026.2646361)).

Most importantly for this app, the paper presents the paired humanization
strategy as a demonstration and leaves comprehensive paired-humanization
validation to future work. Every output should carry a research-use caveat,
and the app should not claim preservation of binding or developability without
experimental evidence
([paper](https://doi.org/10.1080/19420862.2026.2646361)).

## Recommended Biomodals boundary

Expose one operation, **paired humanization**, with scoring always returned.
Keep a separate internal scoring function for tests and future workflow use,
but do not make v1 a menu of every AbNatiV mode.

Inputs:

- wide CSV `id,vh,vl`, complete and strictly validated;
- upstream scientific knobs with upstream defaults: residue threshold, RASA
  threshold, maximum pairing-score decrease, objective weights, forbidden
  substitutions, and allowed mutation positions;
- framework-only mutation by default, with CDR mutation requiring an explicit
  opt-in warning; and
- no custom model, user PDB, unpaired chain, VHH, training, or plotting in v1.

Stable outputs:

- normalized input and final humanized VH/VL FASTA;
- one sequence-level table with input/final joint, chain, region, percentile,
  and raw pairing scores;
- input and final position-by-21 reconstruction matrices and observed-residue
  scores, keyed by ID, chain, and AHo position;
- a mutation table with original/new residue and region; and
- a compact manifest containing every pin, checksum, parameter, device, seed,
  validation result, runtime, and upstream/paper behavior choice.

Structures and CDR displacement are scientifically useful audit artifacts and
are already computed by the upstream pipeline. Include the input/final PDBs and
displacement summary in a compressed result bundle if the benchmark shows the
size is modest. Omit PNG, PAP, ChimeraX, and full DMS intermediates by default.

## Required oracle and acceptance tests

1. Freeze one known VH/VL pair and compare aligned sequences, final humanized
   pair, mutation list, all sequence/region scores, pairing score, percentiles,
   and input/final residue matrices against an isolated upstream 2.0.8 oracle.
2. Test heavy/light type swaps, missing chains, duplicate/empty IDs, ambiguous
   residues, lowercase, overlong input, ANARCI failure, and a mixed batch where
   one row is invalid. No row may disappear silently.
3. Run the same pair twice in one warm process to catch the mutable-default and
   ordering problems; run fresh processes to assess structure variability.
4. Test CPU/GPU numerical drift before allowing both backends under one
   scientific version.
5. Verify both published asset checksums and prove runtime network access is not
   required.
6. Pin an expected one-pair resource envelope and fail clearly when structure
   prediction, alignment, or scoring exceeds it.

Until the licensing and 10-versus-4 RASA questions are resolved, an
implementation can be prototyped and benchmarked, but should not be presented
as a production-equivalent p-AbNatiV2 service.

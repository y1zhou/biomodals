# p-AbNatiV2 app research

Research date: 2026-09-02

Prospective implementation target: `src/biomodals/app/design/pabnativ2/app.py`

## Initial app decisions

Decisions settled on 2026-09-03 for the first Biomodals implementation:

- expose only paired VH-VL humanization and always return parental and final
  scores;
- use the pinned AbNatiV 2.0.8 implementation as the scientific oracle,
  including its four-prediction RASA ensemble rather than the ten predictions
  described in the paper;
- stage checksum-verified paired-model and ABodyBuilder3 checkpoints on the
  `biomodals-store` Modal Volume instead of embedding them in the image;
- benchmark one pair on CPU, A10, and L40S before selecting resources or a
  multi-pair dispatch topology;
- retain one deterministic execution-kernel Task per pair, but after the
  shared-A10G benchmark, keep the direct worker for a one-pair input and group
  multi-pair inputs into fixed provider calls of at most four spawned processes
  on one A10G; admit at most eight A10G calls by default and aggregate ordered
  results on the coordinator into a Volume-backed archive;
- retain parental and final predicted PDBs plus scaffold/CDR displacement
  metrics, while omitting incidental upstream scratch outputs;
- seed supported public APIs and record the seed, with hardware retained only
  as operational telemetry; do not patch possible ordering or tie behavior
  unless repeatability tests demonstrate a problem; and
- return parental and final sequence-, region-, pairing-, percentile-, and
  residue-level scores. Position matrices contain the upstream 21-value
  reconstruction distribution and use `pairing_score` for the raw fraction;
- expose `mutate_cdrs` plus separate AHo `fixed_vh_positions` and
  `fixed_vl_positions`, with framework-only mutation by default;
- expose the residue-score, RASA, and maximum relative pairing-score-decrease
  thresholds with upstream defaults, while fixing the objective weights to
  `a=10` and `b=1` for the initial scientific version;
- expose a canonical-amino-acid `forbidden_residues` control defaulting to
  `C,M`, with gap always forbidden internally;
- omit experimental parental PDB input from v1 and use predicted structures;
- provide an explicit artifact-staging entrypoint that verifies and commits
  the two required checkpoints to `biomodals-store`, while inference mounts
  the app subdirectory read-only and fails closed on missing or invalid files;
- report only endpoint mutations rather than transient accepted search steps;
  and
- treat a completed unchanged pair as successful rather than imposing an
  additional scientific score target;
- write sequence-level metrics to a wide `sequence_scores.csv` with one row
  per pair and endpoint, and write endpoint residue reconstruction matrices to
  `residue_scores.parquet` with one row per pair, endpoint, chain, and AHo
  position;
- store predicted structures under a row-number-prefixed sanitized-ID path,
  retaining original IDs in tables and manifests;
- use AHo as the fixed p-AbNatiV2 coordinate system rather than exposing
  alternative numbering schemes or CDR definitions;
- validate all thresholds in `[0,1]`, chain-local fixed AHo positions as
  unique integers from 1 through 149, and forbidden residues as distinct
  uppercase canonical amino acids before remote inference;
- benchmark one cold run on CPU, A10, and L40S, followed by one warm run on the
  leading device;
- begin with `cpu=(0.125,16.125), memory=(1024,32768)` for the CPU benchmark
  and `cpu=8, memory=32768` for each GPU benchmark; and
- install and call pinned `abnativ==2.0.8` directly without importing the
  existing Biomodals AbNatiV scoring app.

Benchmark harnesses and resource profilers are not part of the production app.
Only the resulting resource, topology, and reproducibility decisions are
retained here and in the benchmark record.

Implementation inspection found no duplicate light-CDR percentile append in
the pinned `eb517f1f` scorer, so the app does not patch scoring logic. It only
applies the guarded strict-Matplotlib colormap fix already needed by the
existing AbNatiV image and ABodyBuilder3's documented NumPy tuple-indexing
compatibility fix. See
[the app-specific deviation note](../../agents/pabnativ2-app-deviations.md).

Licensing is outside the implementation decision for this initial branch. The
license and research-use caveat remain documented rather than enforced by a
runtime gate.

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

There are two important constraints on how the initial app is described:

1. The code is CC BY-NC-SA 4.0 and the README expressly prohibits commercial
   use. Commercial or mixed-use deployment needs permission from the authors or
   a legal determination; this note is not legal advice
   ([license](https://gitlab.doc.ic.ac.uk/sormanni-lab/abnativ/-/blob/eb517f1f0b947084cb7e44a54ef34103e9692f5e/LICENSE),
   [README](https://gitlab.doc.ic.ac.uk/sormanni-lab/abnativ/-/blob/eb517f1f0b947084cb7e44a54ef34103e9692f5e/README.md)).
2. The peer-reviewed method averages solvent accessibility over ten predicted
   structures, while the current source hard-codes four predictions. The
   initial app deliberately follows and versions the current four-prediction
   implementation; it must not claim paper-protocol equivalence
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

The scientific identity of an app run therefore includes the code version,
paired checkpoint checksum, pinned top-level runtime dependencies, AHo alignment
behavior, humanization parameters, and structure-prediction checkpoint.
Hardware is operational metadata rather than part of the scientific
fingerprint. During active development, the environment is not yet a fully
resolved transitive dependency lock: compatible Python patch releases, Conda
builds, and transitive packages may change when an image is rebuilt. A complete
resolved lock and its digest are deferred to release hardening, so the current
runtime fingerprint must not be described as release-grade reproducibility.

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
4. treats a position as liable when its residue score is at or below the
   configured threshold or, for framework positions independently, when the
   parental residue is absent from the permitted human PSSM set;
5. proposes residues from human heavy and kappa/lambda PSSMs above a 1%
   frequency cutoff while excluding C, M, and gap;
6. greedily tries liabilities ordered by the dependence estimate, maximizing a
   weighted change in joint humanness and pairing score while rejecting excess
   pairing-score decreases; and
7. re-scores the input and final pair and predicts both structures for
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
- The wheel embeds the much larger percentile reference tables; the paired
  percentile CSV is about 57.7 MB uncompressed. The six humanization PSSMs
  (about 151 KB total) are present in the pinned source tree but absent from
  the 30.1 MB wheel because its package-data declaration targets the wrong
  package directory. The app must restore and hash-verify those six files from
  the pinned commit
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
before dependencies and extracted assets. Stage only these required artifacts
on `biomodals-store`, verify their checksums before use, and mount them read-only
for inference; do not run the broad upstream `abnativ init` or bake them into
the image.

## Runtime, batching, and determinism

Paired scoring is natively batched (`batch_size=256` in the Python API; the CLI
example uses 128), and alignment can use multiple CPU processes. The scorer
automatically selects CUDA, then Apple MPS, then CPU
([scorer](https://gitlab.doc.ic.ac.uk/sormanni-lab/abnativ/-/blob/eb517f1f0b947084cb7e44a54ef34103e9692f5e/abnativ/model/scoring_functions.py)).
Humanization, however, accepts one pair and repeatedly invokes pair scoring,
DMS, and structure prediction. There is no published one-pair runtime or memory
requirement in the paper, so resource sizing must be measured rather than
inferred.

Start with a single-pair benchmark on CPU, A10, and L40S containers, capturing
image build time, cold start, asset load, alignment, DMS, each structure pass,
total wall time, peak RSS, GPU memory and utilization where applicable, and
output bytes. Choose resources and dispatch only after comparing these runs.
Use `cpu=(0.125,16.125), memory=(1024,32768)` for the CPU probe and
`cpu=8, memory=32768` for the A10 and L40S probes. Run one cold probe on each,
then one warm repetition on the leading device. Install pinned
`abnativ==2.0.8` inside this app and do not depend on the implementation of the
separate Biomodals AbNatiV scoring app.
If single-pair work is long, fan out bounded execution-kernel Tasks while
preserving the overall 1,000-pair hard ceiling. Do not introduce an unbounded
`.map()` path.

Pure scoring uses `eval()` plus inference mode and contains no sampling, so it
should be stable for a pinned checkpoint and runtime within ordinary
floating-point tolerances. The full humanizer has weaker guarantees:

- structure prediction is repeated but the paired humanizer does not expose a
  seed parameter;
- candidate-position intersection uses Python sets before ordering;
- equal-valued mutation ties can depend on candidate iteration order; and
- `forbidden_mut` is a mutable default and is modified in place by appending
  gap, so repeated calls in one warm process can observe changed state
  ([search implementation](https://gitlab.doc.ic.ac.uk/sormanni-lab/abnativ/-/blob/eb517f1f0b947084cb7e44a54ef34103e9692f5e/abnativ/humanisation/humanisation_utils.py)).

The wrapper should pass a fresh forbidden list, seed all seedable libraries,
record hardware as operational telemetry, and test repeated runs in the same
warm process. Do not patch set ordering or tie behavior unless those tests show
instability. Cross-device validation requires the same aligned and final
sequences, eligibility decisions, and endpoint mutations, with numeric outputs
agreeing within documented floating-point tolerances. Hardware does not enter
the Workload Plan Fingerprint.

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
- `residue_score_threshold=0.98`, `rasa_threshold=0.15`, and
  `max_relative_pairing_score_decrease=0.10`, all strictly validated;
- `mutate_cdrs=False` with separate chain-local AHo
  `fixed_vh_positions` and `fixed_vl_positions`; fixed positions remain
  protected when CDR mutation is enabled;
- `forbidden_residues=C,M`, validated as canonical amino acids with gap always
  forbidden internally;
- fixed upstream objective weights `a=10,b=1`, the fixed 1% PSSM cutoff, and
  the fixed four-prediction RASA ensemble; and
- no custom model, user PDB, unpaired chain, VHH, training, or plotting in v1.

Stable outputs:

- normalized input and final humanized VH/VL FASTA;
- `sequence_scores.csv`, with one wide row per ID and `input` or `final`
  endpoint containing joint, chain, region, percentile, and raw pairing
  scores;
- `residue_scores.parquet`, with one row per ID, endpoint, chain, and AHo
  position containing the observed residue, its residue score, and the 21
  reconstruction values;
- `mutations.parquet`, an endpoint mutation table with ID, chain, AHo position,
  region, parental/final residue, and parental/final observed-residue score; and
- a compact manifest containing every pin, checksum, parameter, seed,
  validation result, runtime, upstream/paper behavior choice, and device as
  operational telemetry.

Structures and CDR displacement are scientifically useful audit artifacts and
are already computed by the upstream pipeline. Include the input/final PDBs and
displacement summary in a compressed result bundle if the benchmark shows the
size is modest. Store each pair under
`structures/{row_number}_{sanitized_id}/input.pdb` and `final.pdb`, while
retaining the original ID in tables and the manifest. Omit PNG, PAP, ChimeraX,
and full DMS intermediates by default.

The public coordinate system is fixed to chain-local AHo positions spanning
1 through 149. Thresholds must be finite values in `[0,1]`; fixed positions must be
unique integers in that range; forbidden residues must be distinct uppercase
canonical amino acids. Validate all controls before scheduling inference.

A pair is scientifically successful whenever the pinned humanization procedure
completes, including when it returns an unchanged pair. The initial app defines
no additional target score. Invalid input or an execution error remains a batch
failure under the shared Humanization Batch contract.

## Validation and acceptance tests

1. Perform a one-time manual comparison of a known VH/VL pair against a direct
   upstream 2.0.8 run, covering the aligned and final sequences, mutations,
   sequence/region and pairing scores, percentiles, and endpoint residue
   matrices. Record the setup and result in the research documentation; do not
   commit the run harness or a frozen oracle fixture that future CI cannot
   regenerate through Modal.
2. Test heavy/light type swaps, missing chains, duplicate/empty IDs, ambiguous
   residues, lowercase, overlong input, ANARCI failure, and a mixed batch where
   one row is invalid. No row may disappear silently.
3. Run the same pair twice in one warm process to catch the mutable-default and
   ordering problems; run fresh processes to assess structure variability.
4. Require identical discrete outputs across repeated A10G runs and compare
   numeric outputs with documented tolerances; do not fingerprint individual
   hardware. The initial benchmark showed CPU/GPU greedy decisions diverge, so
   CPU must not serve as an equivalent fallback.
5. Verify both published asset checksums and prove runtime network access is not
   required.
6. Record the observed one-pair resource envelope in the benchmark document.
   Do not make a roughly 30-minute Modal inference run part of routine CI.

The initial app targets equivalence to the pinned 2.0.8 source rather than the
paper's ten-structure protocol. Its non-commercial license and research-use
limitations must remain explicit; the app must not be presented as clinical
validation or as evidence that binding and developability are preserved.

See [the initial single-pair benchmark](pabnativ2-benchmarks.md) for the A10G,
L40S, CPU, and repeatability measurements that selected the final worker.

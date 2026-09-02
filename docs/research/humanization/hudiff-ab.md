# HuDiff-Ab app research

Research date: 2026-09-02

Candidate implementation target: `src/biomodals/app/design/hudiff_ab_app.py`

This note covers **HuDiff-Ab**, the paired conventional-antibody model. The
same repository also contains HuDiff-Nb, but it has a different checkpoint,
input shape, fine-tuning objective, and inference script. It should not be
silently exposed through the HuDiff-Ab app.

## Executive recommendation

HuDiff-Ab is scientifically distinct enough to be useful beside Sapiens: it
jointly generates paired VH/VL frameworks from preserved CDRs and can produce
multiple stochastic candidates, whereas Sapiens scores and greedily revises
each chain independently. The smallest honest app is therefore a **paired
candidate generator**, not another humanness scorer.

Do not begin production integration until the upstream authors clarify the
license. The immutable code release is under PolyForm Noncommercial 1.0.0,
while the authors' 2026 protocol calls the same code MIT; the model repository
declares AFL-3.0, while that protocol calls its release dataset CC BY 4.0. The
checkpoint and training LMDBs are also combined in one archive with no
per-file license manifest. These are material, unresolved conflicts, not a
documentation nicety
([code license](https://github.com/TencentAI4S/HuDiff/blob/bb7636f182699f98c37855dad05a5c6c61b576bd/LICENSE),
[model-repository metadata](https://huggingface.co/cloud77/HuDiff/blob/3455856e5d97aa72dea98653b44a1aec800fc998/README.md),
[authors' protocol table](https://bio-protocol.org/en/bpdetail?id=5816&type=0#software-and-datasets)).

If licensing is resolved, start from the immutable upstream release and make a
narrow, versioned inference patch rather than invoking the command-line script
unchanged. The upstream script has broken direct-sequence dispatch, incomplete
input checks, an ignored seed, always-active functional dropout, and no score
artifact. A Biomodals wrapper should preserve the published sampling algorithm
while making its stochastic identity and validation explicit.

## Primary-source snapshot and reproducible pins

- Upstream has one Git tag, [`v1.0.0` at commit
  `bb7636f182699f98c37855dad05a5c6c61b576bd`](https://github.com/TencentAI4S/HuDiff/tree/bb7636f182699f98c37855dad05a5c6c61b576bd),
  dated 2024-11-29. The current `main` commit is
  [`9a7d1d5a458a2fdd6b9dc7e04e891e4b776f6241`](https://github.com/TencentAI4S/HuDiff/tree/9a7d1d5a458a2fdd6b9dc7e04e891e4b776f6241),
  dated 2025-08-28; the only post-tag change is to `README.md`. Use `v1.0.0`
  rather than floating `main` for executable code.
- The peer-reviewed version of record is *Nature Machine Intelligence* 7,
  1698–1712 (2025), DOI
  [`10.1038/s42256-025-01120-9`](https://www.nature.com/articles/s42256-025-01120-9).
  It supersedes the 2024 bioRxiv citation still shown in the repository README.
  The repository is also archived as
  [Zenodo record `10.5281/zenodo.16974296`](https://doi.org/10.5281/zenodo.16974296).
- The sole released asset is `release_data_dir.tar.gz` in Hugging Face model
  revision
  [`3455856e5d97aa72dea98653b44a1aec800fc998`](https://huggingface.co/cloud77/HuDiff/tree/3455856e5d97aa72dea98653b44a1aec800fc998).
  Its Git-LFS identity is SHA-256
  `95d4e9091463939e3032996ae10bee8c8df72d11326d4498344eb1abe0bcb949`
  and its declared size is 2,070,382,005 bytes. The archive combines training
  LMDBs and both HuDiff-Ab/HuDiff-Nb checkpoints
  ([pinned model card](https://huggingface.co/cloud77/HuDiff/blob/3455856e5d97aa72dea98653b44a1aec800fc998/README.md)).
- Upstream documentation disagrees on the HuDiff-Ab checkpoint path/name:
  `checkpoints/antibody/hudiffab.pt` in the repository README versus
  `release_data_dir/checkpoints/antibody/antibody.pt` in the newer protocol.
  A build must inspect the pinned archive, select the fine-tuned HuDiff-Ab
  checkpoint, record its own inner SHA-256 and size, and fail if its structure
  or embedded configuration differs from the expected identity
  ([repository README](https://github.com/TencentAI4S/HuDiff/blob/9a7d1d5a458a2fdd6b9dc7e04e891e4b776f6241/README.md#hudiff-ab),
  [protocol inference procedure](https://bio-protocol.org/en/bpdetail?id=5816&type=0#B2-Humanization-with-HuDiff-Ab)).
- Recommended runtime pin: code commit `bb7636f...`, model revision
  `3455856e...`, outer archive SHA-256 above, and a generated lock of the first
  dependency set that passes the oracle suite. Upstream pins only Python 3.9,
  PyTorch 1.13.0, torchvision 0.14.0, torchaudio 0.13.0, and CUDA 11.6; most
  Python packages are unconstrained
  ([environment](https://github.com/TencentAI4S/HuDiff/blob/bb7636f182699f98c37855dad05a5c6c61b576bd/environment.yaml)).

## Licensing and redistribution

The following conflict must be resolved before a Biomodals app is shipped:

| Material | License asserted by the owning artifact | Conflicting statement |
| --- | --- | --- |
| GitHub source at `v1.0.0` | PolyForm Noncommercial 1.0.0; use and derivative work are limited to permitted noncommercial purposes | The authors' later protocol labels HuDiff source “MIT” |
| Hugging Face repository and combined release archive | `license: afl-3.0` in repository metadata | The protocol labels the “HuDiff release dataset” CC BY 4.0 |
| Evaluation CSV/FASTA files committed with the source | No separate per-file license or provenance manifest | None supplied |
| Training LMDBs and checkpoints inside the 2.07 GB archive | No per-file license manifest; only the repository-level AFL-3.0 declaration is present | The protocol applies CC BY 4.0 to the release dataset but does not separately identify checkpoints |

Sources: [immutable source license](https://github.com/TencentAI4S/HuDiff/blob/bb7636f182699f98c37855dad05a5c6c61b576bd/LICENSE),
[pinned Hugging Face metadata](https://huggingface.co/cloud77/HuDiff/blob/3455856e5d97aa72dea98653b44a1aec800fc998/README.md),
[authors' protocol](https://bio-protocol.org/en/bpdetail?id=5816&type=0#software-and-datasets),
and the paper's [data-availability statement](https://www.nature.com/articles/s42256-025-01120-9#data-availability).

PolyForm Noncommercial is incompatible with a general assumption that the app
may be used commercially. The discrepancy cannot be cured by copying the
protocol's table into Biomodals. Ask upstream to publish an unambiguous license
for (1) code, (2) each checkpoint, (3) bundled training data, and (4) the
committed evaluation sets. Retain all required notices after clarification.
This is a technical inventory, not legal advice.

## Scientific purpose and claims

HuDiff is an adaptive autoregressive diffusion approach with separate models
for conventional antibodies (HuDiff-Ab) and nanobodies (HuDiff-Nb). HuDiff-Ab
pretrains on paired human heavy/light sequences with framework corruption,
then fine-tunes on paired mouse sequences while using frozen AbNatiV VH,
VKappa, and VLambda models as humanness guidance. Inference preserves the
parental CDRs and sequentially reconstructs framework positions
([paper](https://www.nature.com/articles/s42256-025-01120-9),
[fine-tuning code](https://github.com/TencentAI4S/HuDiff/blob/bb7636f182699f98c37855dad05a5c6c61b576bd/antibody_scripts/antibody_finetune.py),
[fine-tuning configuration](https://github.com/TencentAI4S/HuDiff/blob/bb7636f182699f98c37855dad05a5c6c61b576bd/configs/antibody_finetune.yml)).
AbNatiV is therefore part of how the released model was trained, but its models
are not called by the HuDiff-Ab generation script at inference time.

The model represents an IMGT-aligned heavy chain in 152 slots and light chain
in 139 slots, totaling 291. It encodes residue, chain type (H, lambda, or
kappa), region, and position; separate heavy/light convolutional paths are
followed by joint self-attention and a residue-token decoder. Its 23 tokens are
20 canonical amino acids plus `X`, gap `-`, and mask `<msk>`
([model configuration](https://github.com/TencentAI4S/HuDiff/blob/bb7636f182699f98c37855dad05a5c6c61b576bd/configs/antibody_train.yml),
[model implementation](https://github.com/TencentAI4S/HuDiff/blob/bb7636f182699f98c37855dad05a5c6c61b576bd/model/encoder/model.py),
[tokenizer](https://github.com/TencentAI4S/HuDiff/blob/bb7636f182699f98c37855dad05a5c6c61b576bd/utils/tokenizer.py)).

The paper describes generation “from scratch” from CDRs and without a human
template. In the released inference code, that means all occupied framework
positions are initially masked while CDR residues remain visible. The original
framework residues do not condition residue choice, but their IMGT occupancy
pattern, the chain classes, and fixed terminal positions still influence the
input. “CDR-only” should not be interpreted as accepting six isolated CDR
strings or generating arbitrary variable-region lengths
([input construction](https://github.com/TencentAI4S/HuDiff/blob/bb7636f182699f98c37855dad05a5c6c61b576bd/antibody_scripts/sample.py),
[fixed position grids and masks](https://github.com/TencentAI4S/HuDiff/blob/bb7636f182699f98c37855dad05a5c6c61b576bd/dataset/preprocess.py)).

The released fine-tuned HuDiff-Ab checkpoint is specifically presented for
humanizing **mouse** conventional antibodies. It accepts a paired heavy chain
and either kappa or lambda light chain; there is no source-species parameter.
The repository does not establish scientific support for macaque, rat, rabbit,
or other sources. Nanobody/VHH humanization uses HuDiff-Nb, a different model
and operation
([README](https://github.com/TencentAI4S/HuDiff/blob/9a7d1d5a458a2fdd6b9dc7e04e891e4b776f6241/README.md),
[chain encoding](https://github.com/TencentAI4S/HuDiff/blob/bb7636f182699f98c37855dad05a5c6c61b576bd/utils/tokenizer.py)).

The version-of-record paper reports public benchmark comparisons and three
wet-lab examples: one murine antibody and two alpaca-derived nanobodies. Its
best antibody candidate had reported affinity 0.15 nM versus 0.12 nM for the
parent, while the best nanobody candidate had 2.52 nM versus 5.47 nM. These are
selected examples, not evidence that arbitrary candidates retain affinity;
the authors' supplementary discussion reports some generated variants with
roughly tenfold or hundredfold affinity losses and attributes changes to
framework/CDR interactions
([paper abstract](https://www.nature.com/articles/s42256-025-01120-9#Abs1),
[supplementary information](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs42256-025-01120-9/MediaObjects/42256_2025_1120_MOESM1_ESM.pdf)).
Results therefore require downstream humanness, developability, structure,
binding, and experimental review; the generator is not an immunogenicity or
affinity predictor.

The authors evaluate with OASis, T20, germline identity, preservation, and
mutation precision. Those are separate evaluation procedures, not objectives
or outputs of the released inference call. The supplementary results also note
that HuDiff-Ab preserves less of the parental framework and can yield more
diverse sequences than Sapiens because the original framework is not used for
residue generation
([supplementary information](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs42256-025-01120-9/MediaObjects/42256_2025_1120_MOESM1_ESM.pdf),
[evaluation scripts](https://github.com/TencentAI4S/HuDiff/tree/bb7636f182699f98c37855dad05a5c6c61b576bd/evaluation)).

## Exact upstream input contract

The interactive entry point is
[`antibody_scripts/sample_for_anti_cdr.py`](https://github.com/TencentAI4S/HuDiff/blob/bb7636f182699f98c37855dad05a5c6c61b576bd/antibody_scripts/sample_for_anti_cdr.py).
It is a script, not an installable package or stable Python API.

- Documented inputs are a FASTA path or `--heavy_seq` plus `--light_seq`.
  Actual FASTA parsing searches descriptions for the case-sensitive substrings
  `heavy chain` and `light chain`; later matching records overwrite earlier
  ones, unrelated records are ignored, and both matches are asserted.
- The newer protocol claims the first two FASTA records are used as a fallback,
  but the released code has no such fallback. The direct-string mode is also
  unreachable from the CLI as written because `--anti_complex_fasta` has a
  non-`None` default and argparse cannot turn a supplied string into `None`.
- Each sequence is parsed as an AbNumber `Chain(..., scheme="imgt")`, then
  numbered again through ANARCI and mapped into the fixed 152/139 position
  grids. Chain identity is inferred; H, K, and L are the only model classes.
- The generation mask is not user-configurable. The fine-tuned path preserves
  fixed Kabat CDR masks (represented on the IMGT grid), plus fixed tail slots,
  and masks existing framework residues. There is no option to mutate CDRs or
  choose a different CDR definition.
- The tokenizer accepts canonical uppercase residues, `X`, gap, and mask, but
  the public interface performs no explicit alphabet, byte, record-count,
  duplicate-ID, chain-role, or length validation. Unknown grid insertions can
  be printed and ignored, including an error message for missing CDR positions,
  rather than failing the request.

Sources: [interactive script](https://github.com/TencentAI4S/HuDiff/blob/bb7636f182699f98c37855dad05a5c6c61b576bd/antibody_scripts/sample_for_anti_cdr.py),
[input construction](https://github.com/TencentAI4S/HuDiff/blob/bb7636f182699f98c37855dad05a5c6c61b576bd/antibody_scripts/sample.py),
[position grids](https://github.com/TencentAI4S/HuDiff/blob/bb7636f182699f98c37855dad05a5c6c61b576bd/dataset/preprocess.py),
and the conflicting [authors' protocol](https://bio-protocol.org/en/bpdetail?id=5816&type=0#B2-Humanization-with-HuDiff-Ab).

## Sampling, mutations, scores, and output semantics

At the start of a candidate batch, every occupied framework site is `<msk>`.
The site order is either the natural aligned order or a NumPy shuffle. For each
site, the model performs a full forward pass, softmaxes its logits, and samples
one token with `torch.multinomial`; the chosen token is written back before the
next site. Heavy and light chains are passed together on every step, so later
choices can condition on earlier choices across the pair. `batch_size` is the
number of candidate replicas for **one input pair**, not the number of input
pairs
([sampling loop](https://github.com/TencentAI4S/HuDiff/blob/bb7636f182699f98c37855dad05a5c6c61b576bd/antibody_scripts/sample_for_anti_cdr.py),
[batch construction](https://github.com/TencentAI4S/HuDiff/blob/bb7636f182699f98c37855dad05a5c6c61b576bd/antibody_scripts/sample.py)).

This is wholesale stochastic framework reconstruction, not a ranked list of
proposed point mutations. A “mutation” is simply an aligned parental/generated
residue difference after sampling; a generated residue may equal the parental
residue by chance. CDR residues are intended to be copied exactly. The sampling
softmax excludes `<msk>` but still includes `X` and `-`, and decoding drops
sampled gaps, so malformed or length-changing candidates are technically
possible and must be detected
([token slice and decoder](https://github.com/TencentAI4S/HuDiff/blob/bb7636f182699f98c37855dad05a5c6c61b576bd/antibody_scripts/sample_for_anti_cdr.py),
[tokenizer](https://github.com/TencentAI4S/HuDiff/blob/bb7636f182699f98c37855dad05a5c6c61b576bd/utils/tokenizer.py)).

The script defaults to ten parallel candidates and requests ten samples. It
deduplicates exact VH/VL pairs, but decrements `sample_number` even for a
duplicate, so it can return fewer unique candidates than requested. Every
candidate receives the same name. The only scientific output is a CSV with a
trailing empty column and fields `Specific,name,hseq,lseq`; it includes the
parental pair followed by generated pairs. A timestamped directory and text log
are side effects
([output loop](https://github.com/TencentAI4S/HuDiff/blob/bb7636f182699f98c37855dad05a5c6c61b576bd/antibody_scripts/sample_for_anti_cdr.py)).

HuDiff-Ab inference exposes **no returned score**. Per-site categorical
probabilities exist transiently during sampling but are discarded. Because
each distribution is conditional on sampling order and all prior sampled
tokens, it is not a position-independent score matrix like Sapiens. If a future
app records it, label it `conditional_sampling_probability`, record the exact
step/order/context, and do not call it humanness, immunogenicity, confidence,
or OASis. OASis and T20 are separate evaluation commands
([sampling code](https://github.com/TencentAI4S/HuDiff/blob/bb7636f182699f98c37855dad05a5c6c61b576bd/antibody_scripts/sample_for_anti_cdr.py),
[evaluation workflow](https://github.com/TencentAI4S/HuDiff/blob/9a7d1d5a458a2fdd6b9dc7e04e891e4b776f6241/README.md#evaluation)).

## Runtime, batching, and determinism

Upstream's environment targets Python 3.9, PyTorch 1.13.0, and CUDA 11.6. It
also needs sequence-models, ESM, AbNumber/ANARCI, Biopython, EasyDict, NumPy,
and PyYAML along the inference import path; several are not version-pinned and
Biopython/PyMOL are used by imported modules despite not appearing in the
environment file. There is no `setup.py`, `pyproject.toml`, lockfile, CI, or
test suite
([environment](https://github.com/TencentAI4S/HuDiff/blob/bb7636f182699f98c37855dad05a5c6c61b576bd/environment.yaml),
[imports](https://github.com/TencentAI4S/HuDiff/blob/bb7636f182699f98c37855dad05a5c6c61b576bd/model/encoder/model.py),
[repository tree](https://github.com/TencentAI4S/HuDiff/tree/bb7636f182699f98c37855dad05a5c6c61b576bd)).

The script automatically uses CUDA when visible and otherwise CPU; it has no
device argument. The authors' newer protocol recommends an NVIDIA GPU with at
least 8 GB VRAM and at least 16 GB host RAM for inference. On an RTX 3060 Ti it
reports approximately 2 minutes for ten variants of one antibody versus about
20 minutes on CPU. Treat these as upstream characterization, not a Biomodals
service-level promise
([protocol hardware and timing](https://bio-protocol.org/en/bpdetail?id=5816&type=0#general-notes)).

Inference is intentionally and accidentally stochastic:

1. NumPy shuffles the framework-site order.
2. `torch.multinomial` samples every generated residue.
3. The model uses `torch.nn.functional.dropout` without passing
   `training=self.training`, so dropout remains active even after `model.eval()`.
4. The interactive script accepts and logs `--seed` but comments out the only
   `seed_all(args.seed)` call.

Sources: [interactive sampler](https://github.com/TencentAI4S/HuDiff/blob/bb7636f182699f98c37855dad05a5c6c61b576bd/antibody_scripts/sample_for_anti_cdr.py),
[functional dropout](https://github.com/TencentAI4S/HuDiff/blob/bb7636f182699f98c37855dad05a5c6c61b576bd/model/encoder/model.py),
and [seed helper](https://github.com/TencentAI4S/HuDiff/blob/bb7636f182699f98c37855dad05a5c6c61b576bd/utils/misc.py).

Consequently upstream's `--seed` does not reproduce candidates. Restoring the
seed may reproduce one pinned software/device execution, but exact CPU/GPU or
cross-version identity is not established. Disabling functional dropout would
be a plausible bug fix but would change the inference distribution; it must be
an explicit, fingerprinted scientific mode with separate validation, not a
silent cleanup.

There is no multi-antibody inference batch. The dataset-oriented `sample.py`
loops antibody pairs serially, while tensor batch size produces candidates for
the current pair. At the protocol's reported timing, 100 pairs × 10 candidates
would be hours on one GPU. A Biomodals batch app therefore needs durable
per-pair Tasks or a measured bounded coordinator rather than presenting a huge
CSV as one opaque call.

## Validation and security hazards

1. **Pickled checkpoints.** `torch.load` can execute pickle payloads. Never
   accept a user-supplied checkpoint. Bake or stage only the pinned,
   checksum-verified checkpoint; if a newer Torch requires
   `weights_only=False` for its embedded EasyDict configuration, permit that
   only for this trusted internal artifact
   ([checkpoint loader](https://github.com/TencentAI4S/HuDiff/blob/bb7636f182699f98c37855dad05a5c6c61b576bd/antibody_scripts/sample_for_anti_cdr.py)).
2. **No trust-boundary validation.** Upstream relies on `assert`, AbNumber, and
   broad exception handling. It can ignore unsupported numbered positions and
   then continue. Validate bytes, rows, IDs, alphabet, chain roles, complete
   numbering, every grid position, and limits locally and remotely; never use
   optimized Python assertions as request validation.
3. **Broken error recovery.** A bare `except` around input construction logs a
   message and then continues with variables that may not exist. Fail the pair
   with a precise diagnostic instead
   ([interactive script](https://github.com/TencentAI4S/HuDiff/blob/bb7636f182699f98c37855dad05a5c6c61b576bd/antibody_scripts/sample_for_anti_cdr.py)).
4. **Unsafe broad CLI surface.** The dataset sampler defines a `type=eval`
   argument and multiple misleading `type=bool` arguments. An internal
   numbering helper interpolates a sequence into a `shell=True` command. These
   paths must not receive user input; do not expose the upstream CLI or vendor
   the whole training/evaluation surface into the operation
   ([dataset sampler](https://github.com/TencentAI4S/HuDiff/blob/bb7636f182699f98c37855dad05a5c6c61b576bd/antibody_scripts/sample.py),
   [shell helper](https://github.com/TencentAI4S/HuDiff/blob/bb7636f182699f98c37855dad05a5c6c61b576bd/utils/anti_numbering.py)).
5. **Fasta ambiguity.** Case-sensitive description matching, repeated-chain
   overwrite, unrelated records, and broken string dispatch make the
   interactive parser unsuitable as a public contract. Parse a normalized
   table in Biomodals instead.
6. **Generated-output validity.** Reject or explicitly resample candidates with
   `X`, lost positions, wrong chain classification, changed CDRs, or a failed
   numbering round-trip. Resampling needs a bounded-attempt policy and must be
   captured in result identity.
7. **Dependency confusion and unnecessary imports.** Loading `model_selected`
   eagerly imports training datasets, AbNatiV helpers, and PyMOL even though
   HuDiff-Ab inference only needs `AntiTFNet`. Isolate the exact inference
   architecture and eliminate unused imports under oracle coverage.
8. **Output paths and logs.** Do not allow user-controlled output or checkpoint
   paths. Use per-call temporary directories, produce a validated archive, and
   avoid global logger-handler accumulation in warm containers.

## Required upstream patches and scientific identity

The minimum production patch set should be small and separately documented:

- expose a pure function over one validated VH/VL pair rather than spawning
  `sample_for_anti_cdr.py`;
- replace FASTA/name/path handling with app-owned data structures;
- load only the pinned HuDiff-Ab checkpoint and narrow inference model;
- explicitly seed Python, NumPy, and Torch for each pair/candidate set and
  record device/software identity;
- use an explicit enum for shuffled versus left-to-right sampling;
- return candidates in memory and preserve the actual number of unique results;
- reject missing/ignored IMGT positions and invalid generated tokens;
- remove unused training, evaluation, subprocess, PyMOL, and AbNatiV runtime
  imports from the inference path; and
- decide explicitly whether to preserve upstream always-active dropout or add
  a deterministic-dropout-fix mode. The latter changes scientific behavior and
  must alter the execution fingerprint/cache identity.

The app should not claim upstream identity merely because it loads the same
weights. Record the code commit, outer and inner checkpoint digests, complete
dependency lock, device class, sampling order, root seed, per-pair derived seed,
dropout policy, candidate count, and invalid-sample policy in the result
manifest.

## Oracle and equivalence plan

Because upstream ships no tests and stochastic outputs, equivalence must be
established at several levels:

1. Build the exact Python 3.9/PyTorch 1.13/CUDA 11.6 environment from
   `v1.0.0`, extract the pinned checkpoint, and inject seeding immediately
   before upstream input construction and sampling. Preserve always-active
   dropout for this reference oracle.
2. Use the repository's 7K9I pair, several Humab25 kappa/lambda pairs, CDR
   insertion cases, maximum supported grid occupancy, and deliberately invalid
   sequences. Do not treat the protocol's displayed stochastic candidates as
   fixed goldens.
3. Compare exact chain classification, 152/139 padded tokens, preserved-mask
   indices, region/chain tensors, sampling order, and the set of mutable sites.
4. At every autoregressive step, compare logits and conditional probabilities
   within a tight tolerance on the same device/runtime. With identical RNG
   state, compare sampled token IDs and final candidates exactly.
5. Compare aligned mutation tables exactly and assert parental CDR and terminal
   preservation, canonical output, numbering round-trip, and heavy/light chain
   roles.
6. Test retry/resumption by deriving each pair's seed independently of task
   scheduling. The same pair, configuration, checkpoint, and device must yield
   the same candidates after a retry.
7. If upgrading Python/Torch/CUDA, removing functional dropout, vectorizing
   across input pairs, changing invalid-sample resampling, or changing the
   tokenizer softmax support, treat it as a new scientific implementation.
   Re-run per-step oracles and characterize sequence/diversity changes across
   many seeds; do not expect bitwise cross-device equality.
8. OASis, T20, Sapiens, or another independent humanness evaluator may be run
   as downstream characterization, but none belongs in the HuDiff-Ab
   equivalence oracle.

## Smallest Biomodals app boundary

Recommended operation: `hudiff_ab.humanize`.

Input should be a bounded UTF-8 wide CSV with `id,vh,vl`, unique nonempty IDs,
complete canonical uppercase variable regions, and exactly one H plus one K/L
chain per row. Add explicit `candidate_count`, root `seed`, and
`sampling_order=shuffle|left_to_right`. Keep the CDR definition fixed to the
released model's Kabat mask on an IMGT grid; adding alternate definitions or
CDR mutation would not be HuDiff-Ab-equivalent. Establish both row and byte
limits from GPU benchmarks, and separately cap total candidates.

Use one GPU worker per pair or a measured small group of pairs, with candidate
replicas batched inside that worker. Represent multi-pair CLI work as an
ExecutionDefinition with durable per-pair Tasks and bounded scheduling so
workflow callers and standalone callers share the same scientific operation.
Start by benchmarking T4 and A10G with 1, 10, and 20 candidates; the authors'
RTX 3060 Ti result suggests CPU should be a fallback/oracle path, not the
default production resource.

Return a compact inline `.tar.zst` containing:

- normalized input CSV;
- one row per generated candidate with stable `id`, `candidate_index`, VH, VL,
  seed, sampling order, and validity status;
- an aligned mutation table with chain, IMGT position, parental residue, and
  generated residue;
- paired FASTA for valid candidates; and
- an identity/diagnostics manifest containing checkpoint, runtime, patch,
  stochastic, retry, and rejection metadata.

Do not fabricate a humanness score. A conditional sampling trace can be added
later as a clearly named optional artifact, but it is large, order-dependent,
and not required for the smallest useful generator.

Model setup should download the pinned 2.07 GB archive during image
construction, verify its outer digest, extract only the fine-tuned HuDiff-Ab
checkpoint, verify and record the new inner digest, and discard all LMDBs and
the HuDiff-Nb checkpoint. A persistent model Volume is unnecessary unless
measurements show image construction or checkpoint size makes it advantageous.
Because the faithful environment is Python 3.9 while Biomodals is Python 3.12+,
use the repository's documented cross-runtime execution boundary or prove a
modernized runtime equivalent before importing project source into the image.

## Relationship to Sapiens and the humanization stack

HuDiff-Ab and Sapiens overlap in producing CDR-preserving humanized VH/VL
sequences from a parental pair, but their scientific contracts differ:

| Property | HuDiff-Ab | Sapiens/BioPhi-style humanization |
| --- | --- | --- |
| Chain context | Joint paired H + K/L model | Separate heavy and light models |
| Framework use | Masks the parental framework, retaining its aligned occupancy | Scores and revises the parental sequence directly |
| Candidate policy | Stochastic autoregressive reconstruction; many candidates | Per-position argmax, optionally repeated |
| Native score output | None | Full per-residue amino-acid probability matrix |
| Released target | Mouse paired conventional antibodies | No source-species parameter; published benchmarks used murine/inferred parental sequences |
| Primary value in a stack | Diverse paired candidate generation | Auditable conservative humanization and scoring |

Sources: [HuDiff-Ab sampler](https://github.com/TencentAI4S/HuDiff/blob/bb7636f182699f98c37855dad05a5c6c61b576bd/antibody_scripts/sample_for_anti_cdr.py),
[HuDiff supplementary comparison](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs42256-025-01120-9/MediaObjects/42256_2025_1120_MOESM1_ESM.pdf),
[Sapiens API](https://github.com/Merck/Sapiens/blob/3d676ecde0b6fc113d3f9c5bcb3721a7d70a85b7/sapiens/predict.py),
and [BioPhi humanization](https://github.com/Merck/BioPhi/blob/bc59cd17b690a634553ac50a60840b9a89bd21b0/biophi/humanization/methods/humanization.py).

They are complementary as **independent candidate-generation strategies** and
as generator/evaluator: Sapiens can score HuDiff candidates, provided the
result is labeled a Sapiens model score rather than an independent
immunogenicity measurement. Running both does not validate binding or make
either method's scientific claims transfer to the other.

Humatch, CUMAb, p-AbNatiV2, and OASis should be compared only after their own
contracts and licenses are pinned. The safe stack architecture is to preserve
each method's native semantics behind a common paired-input/result-envelope
shape, then compare or rank candidates in a separate workflow layer. OASis is
not required to generate HuDiff-Ab candidates: upstream installs BioPhi/OASis
only for evaluation
([README evaluation section](https://github.com/TencentAI4S/HuDiff/blob/9a7d1d5a458a2fdd6b9dc7e04e891e4b776f6241/README.md#evaluation)).

## Go/no-go checklist

Before implementation:

- obtain authoritative commercial-use and redistribution terms for code,
  checkpoints, training data, and evaluation data;
- inspect the pinned archive and record the HuDiff-Ab checkpoint's inner path,
  SHA-256, size, and embedded configuration;
- prove a locked inference environment can load and run the checkpoint;
- choose and version the upstream-dropout versus fixed-dropout behavior;
- benchmark T4/A10G memory and latency for 1/10/20 candidates and paired-task
  concurrency; and
- capture same-device seeded upstream oracle fixtures before refactoring.

Until the first item is resolved, HuDiff-Ab belongs in the research/design
stack but not in a distributable production PR.

# Preserve HuDiff-Ab stochastic inference semantics

Status: accepted.

The initial HuDiff-Ab app targets the immutable `v1.0.0` source and released
checkpoint while repairing its ignored seed and preserving its accidentally
active inference dropout. `candidate_count` counts sampling attempts, shuffled
position order remains shared across a pair's attempts, and
`upstream_inference_dropout=true` is the default scientific mode; disabling it
is a separately fingerprinted HuDiff-derived mode. Because current Modal no
longer supports upstream Python 3.9, the production worker uses the nearest
compatible Python 3.10, Torch 1.13, and CUDA 11.6 runtime, while exact Python
3.9 oracle comparisons remain external. These choices preserve the released
sampling distribution as closely as the deployment platform permits and make
retry behavior reproducible for a fixed seed, runtime, and device. The worker
enables Torch's strict deterministic-algorithm mode, disables cuDNN benchmarking,
forces deterministic cuDNN behavior, and sets
`CUBLAS_WORKSPACE_CONFIG=:4096:8` before importing Torch. This policy is both
fingerprinted and reported in each pair result; it does not promise bitwise
identity across different device classes or runtime versions.

The image clones the pinned upstream repository and invokes its antibody
inference script. Narrow, exact-preimage guarded patches activate the ignored
seed, expose the fingerprinted dropout choice, remove unused inference import
edges, and preserve complete candidate-attempt accounting. The source commit
and patch digest are part of scientific identity.

The released archive is audited and staged in `biomodals-store` before
inference under the stable app-specific `hudiff/` directory. Training data and
other non-runtime bulk data are removed, while the complete checkpoint subtree
is retained and manifested for later HuDiff apps. A version-addressed directory
is unnecessary for this pinned, unlikely-to-change publication. The initial
performance measurement is one cold A10G call for one parental pair and the
default ten attempts; no separate container call warms inference.

Sampling retains upstream's full token support. Outputs containing `X`, gaps,
changed fixed positions, wrong chain roles, or incomplete numbering are
recorded as invalid attempts with their raw generated chains and rejection
reason. They consume attempts and are neither filtered at the logits nor
resampled. Validation precedes deduplication: the first valid unique pair is a
candidate and later exact valid matches are duplicates that reference it. A
candidate attempt is identified by its pair seed and attempt index rather than
an invented independent per-attempt seed.

Independent output validation uses ANARCI's lower-level domain metadata rather
than its `number()` convenience function. In the pinned ANARCI release,
`number()` deliberately maps both raw kappa (`K`) and lambda (`L`) assignments
to the public light-chain class `L`; the domain metadata retains the distinction
needed to reject a generated light chain whose subtype differs from its parent.
The parental VH and VL are numbered once per pair, before candidate validation,
so this check does not repeat HMMER for every sampling attempt.

The first implementation accepts 1–10 attempts per pair and at most 10,000
attempts over the existing 1,000-pair input ceiling. It validates the complete
batch before dispatch and each provider batch again before model loading. A
single-pair request maps directly to one A10G Provider Call. Multi-pair requests
use scheduler-owned fixed batches of two; the final call may contain one pair.
Each batch runs its pairs as concurrent upstream subprocesses with independently
derived seeds and retains one independently publishable Task result per pair.

The operation returns one inline `.tar.zst` containing normalized input, all
attempts, valid unique candidates, paired candidate FASTA,
`mutations.parquet`, and a scientific manifest. It uses no cross-run result
cache; `biomodals-store` holds staged model assets rather than result bundles.
The execution kernel may durably materialize invocation-scoped results.

The initial 7K9I cold benchmark used root seed zero and completed ten attempts
on one A10G in 16.00 seconds. Two-second samples showed 0.97 mean CPU core,
7.27 GiB maximum sampled host memory, 39.9% mean and 90% peak GPU utilization,
and 1.96 GiB peak GPU memory. Nine attempts were valid unique candidates and
one was a duplicate. These measurements predate strict deterministic-algorithm
enforcement and remain hardware-sizing evidence rather than a runtime promise.

Two separate source-backed production executions then validated the strict
policy on A10G with one 7K9I attempt and root seed `20260905`. Both reported
deterministic Torch algorithms enabled, deterministic cuDNN enabled, cuDNN
benchmarking disabled, and the pinned cuBLAS workspace configuration. Their
input, attempt, candidate, FASTA, and Parquet artifacts were byte-identical;
only the intentionally distinct run name differed between manifests. These
runs validate fixed-runtime, fixed-device retry identity without creating a
frozen CI fixture that future tests could not regenerate.

A later two-process experiment completed distinct 7K9I and 3F8 pairs together
on one A10G in 24.67 and 24.89 seconds across two cold containers. The
instrumented run averaged 1.89 CPU cores, reached 4,002 MiB GPU memory, and held
100% GPU utilization during sampling. Sampled aggregate process RSS reached
17.50 GiB, although that `/proc` sum double-counts shared pages. Against the
16.00-second single-pair baseline, this improves throughput by 1.29 times and
reduces GPU time per pair by about 22%, but it is slower than running two A10G
containers in parallel. The production implementation therefore keeps the
single-pair direct path and uses fixed two-pair batches for multi-pair requests.
The batch worker permits up to 64 GiB host memory, matching the measured shape.

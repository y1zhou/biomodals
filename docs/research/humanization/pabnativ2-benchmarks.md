# p-AbNatiV2 single-pair benchmark

Date: 2026-09-03

This benchmark used the upstream 3F8 VH/VL example, AbNatiV 2.0.8, the
published paired checkpoint, ABodyBuilder3, and otherwise-default paired
humanization controls. The input and raw result archives are outside the
repository under `/tmp/pabnativ2-benchmark/`. Modal app
`ap-Lu8yhQxi1gm5jBrDBXDI4R` ran the three cold probes concurrently.

## Baseline results

| Device | Total | DMS | RASA | Peak process RAM | Mean GPU | Peak GPU RAM | Mutations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A10G | 1,648.5 s | 195.6 s | 18.6 s | 8,568.8 MiB | 5.4% | 5,726 MiB | 28 |
| L40S | 1,716.8 s | 221.4 s | 18.6 s | 8,307.2 MiB | 7.8% | 8,332 MiB | 28 |
| CPU | 2,340.4 s | 668.8 s | 28.7 s | 7,841.6 MiB | n/a | n/a | 33 |

The L40S was 4.1% slower than the A10G for this pair. Most elapsed time is in
the serial greedy candidate search rather than the initial DMS, structure
prediction, endpoint scoring, asset validation, or packaging. Low average GPU
utilization is therefore expected and the higher-priced L40S does not improve
this workload.

The intended warm A10G repetition took 1,709.5 seconds, but its invocation
index was zero. The first A10G container had scaled down while the harness
waited for L40S, so this was a second cold repeat rather than a valid warm
measurement. It is useful as a reproducibility replicate but must not be used
to estimate warm-start savings.

## Reproducibility

The two A10G repetitions produced identical final sequences, identical 28-row
mutation tables, and byte-equivalent score values. Their input and final PDB
files had identical atom topology and maximum numeric differences of 0.001 Å
and 0.002 Å, respectively.

A10G and L40S also produced identical final sequences and mutation decisions.
The maximum absolute difference across reconstruction probabilities was
`1.73e-6`; across aggregate endpoint scores it was `5.99e-7`. Their structures
had identical atom topology with maximum numeric differences of 0.191 Å for
the input prediction and 0.023 Å for the final prediction.

CPU is not scientifically interchangeable with GPU for the greedy humanizer.
It selected 33 mutations and produced different final VH and VL sequences.
Small device-level score or structure differences can cross downstream
candidate eligibility and greedy selection boundaries. A CPU implementation
must not silently serve as fallback for the A10G app identity.

## Runtime decision

Use A10G for the workflow worker with `cpu=(0.125, 8.125)` and
`memory=(1024, 32768)`. Keep device selection fixed in the provider binding and
record it as operational telemetry, while excluding it from the scientific
fingerprint. Do not use L40S unless later evidence shows a material improvement,
and do not use CPU as an equivalent execution path.

The measured one-pair runtime is too long for serial batch execution. The
production app therefore creates one deterministic execution-kernel Task per
pair, uses a configurable run-wide GPU-call ceiling with a default of eight,
and performs ordered aggregation on the coordinator. Benchmark-only phase,
cgroup, and GPU utilization instrumentation is excluded from production code.

The baseline image pinned Lightning 2.5.0 while the checkpoint reported that
it was produced by Lightning 2.5.5, generating a warning for every model load.
The final image pins both `pytorch-lightning` and `lightning` to 2.5.5.

A true back-to-back A10G validation on the final image used Modal app
`ap-8ickueusrROs7MKxHR9xCj`. The cold call took 1,641.2 seconds and the warm
call took 1,684.6 seconds with invocation indexes zero and one, respectively.
Warnings were suppressed. Cold and warm runs had identical final sequences,
mutation tables, aggregate scores, and residue matrices; PDB topology was
identical and maximum numeric differences were 0.001 Å. The final cold output
was also identical to the Lightning 2.5.0 A10G baseline except for expected
runtime identity metadata and 0.001–0.002 Å PDB differences. Warm reuse does
not materially reduce wall time because model initialization and asset
validation are negligible beside the greedy search.

## Three-pair concurrency in one A10G container

Date: 2026-09-04

Modal app `ap-2iDgJN70qV2E2wDHQqN9Wi` ran the upstream test pairs 3F8,
Abagovomab, and Abciximab simultaneously as three spawned processes in one
A10G container. The container used `cpu=(0.125, 8.125)` and
`memory=(512, 65536)`. Each child used its own derived seed and temporary
directory and initialized its own CUDA models. The benchmark imposed a
40-minute cutoff but completed normally before it.

| Pair | Completion | Mutations | Peak child RSS |
| --- | ---: | ---: | ---: |
| Abciximab | 553.3 s | 2 | 8,150.5 MiB |
| Abagovomab | 733.1 s | 9 | 8,598.4 MiB |
| 3F8 | 1,681.7 s | 28 | 8,506.6 MiB |

The complete three-pair container call took 1,686.1 seconds (28.1 minutes).
The long-running 3F8 pair was only 33.2 seconds, or 2.0%, slower than its
1,648.5-second one-pair cold baseline. It produced the same final VH and VL and
the same 28 mutations as the baseline. The two additional pairs completed
while 3F8 was still running, so this workload compressed 49.5 minutes of
observed child elapsed time into 28.1 billed A10G minutes without materially
delaying the longest pair. Pair runtime remained strongly search-dependent:
the two-edit Abciximab result completed much earlier than 3F8.

Ten-second `nvidia-smi` samples measured 9.0% mean and 100% peak GPU
utilization, compared with 5.4% mean for the one-pair baseline. Peak GPU memory
was 15,345 MiB. During the three-way phase, direct cgroup inspection measured
about three CPU cores and at most 11.2 GiB current host-memory use. Child peak
RSS values include shared mappings and must not be added together. The
benchmark harness expected cgroup v2 counters while the Modal container used
cgroup v1, so no continuous CPU or host-memory series was retained; the
reported values are live cgroup observations taken during the run.

Naive three-process concurrency is therefore viable for this tested mix: it
fit comfortably on one A10G, preserved the known 3F8 discrete result, and did
not push the longest pair near the cutoff. This measurement does not guarantee
the same memory and latency for four worst-case pairs.

The selected production topology preserves the existing direct worker when
the complete input contains one pair. Inputs containing two through four pairs
run those pairs as separate spawned processes in one A10G container. Larger
inputs retain durable per-pair Tasks but group provider calls into stable
batches of four, with a final smaller batch when needed. The multiprocessing
worker uses `cpu=(0.125, 8.125)` and `memory=(512, 65536)`. All benchmark
instrumentation was temporary and is excluded from production commits.

## Direct-upstream output comparison

Date: 2026-09-04

Modal app `ap-JAQ0VlKqjU9OzRSfXFKHMV` called the installed upstream AbNatiV
2.0.8 paired humanizer and paired scorer directly, bypassing the Biomodals
execution, normalization, and packaging code. It used the pinned compatibility
image and model assets, an A10G, the 3F8 input, pair seed `2049599637`, and the
same source-default controls as the accepted final-image cold result. The run
started device calculation at 14:53:18 SGT and completed at 15:19:57 SGT,
approximately 1,599 seconds later. The temporary harness and raw output remain
outside the repository.

The comparison target was the existing
`pabnativ2_a10_lightning255_cold_pabnativ2.tar.zst` result. The direct upstream
run produced identical final VH and VL sequences, the same 28 mutations, and
identical input/final AHo alignments and ordered endpoint residue keys. Maximum
absolute differences were `3.16e-10` across sequence, region, pairing, and
percentile scores; `2.96e-8` across observed residue scores and
reconstruction-probability matrices; and zero across the reported scaffold/CDR
displacement values. These differences are floating-point noise, so the
production wrapper preserves the pinned upstream scientific output for this
oracle pair.

This is a one-time manual source comparison, not a committed fixture or a
routine CI test. Future CI cannot regenerate the oracle without a long Modal
GPU run, so local tests instead cover input validation, field normalization,
batch scheduling, per-pair failures, and aggregation deterministically.

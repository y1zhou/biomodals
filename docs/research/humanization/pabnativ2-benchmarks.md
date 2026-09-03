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

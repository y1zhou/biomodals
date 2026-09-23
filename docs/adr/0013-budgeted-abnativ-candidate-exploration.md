# Bound AbNatiV exploration without pruning passing designs

Status: accepted and implemented; native deployment verification remains pending.

Native exhaustive VHH search can grow combinatorially and applies a separate
two-objective output frontier. Optional Biomodals exploration instead evaluates
all allowed combinations when they fit a budget, otherwise samples unique
combinations with balanced coverage across mutation counts, and retains every
native-score-passing design for joint ranking with HuDiff. This deliberately
trades exhaustive coverage and its optimum guarantees for bounded candidate
evaluation; it is not an upstream-equivalent exhaustive algorithm or evidence
of improved experimental quality.

Enhanced remains the default. Preserve native substitution restrictions,
protected residues and mode-specific score rules, version the sampling policy
and seed, and distinguish complete from sampled coverage in compact provenance.
The [specification](../specs/nanobody-humanization.md#exhaustive-diversity-follow-up-23-september-2026)
owns the controls, resource limits, structural-work scope and verification plan.

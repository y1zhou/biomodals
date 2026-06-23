# Adopt the canonical App Run Layout without legacy path compatibility

Biomodals apps moving to `AppRunLayout` will read and write only the canonical `inputs/`, `outputs/`, `logs/`, `failures/`, `metrics/`, and `.markers/` locations. RFdiffusion, Rosetta, FlowPacker workflow outputs, PPIFlow logs, and IgGM logs may require one-time migration or recomputation; the branch will not retain legacy cache probes or dual write formats because it has not yet merged and maintaining two durable layouts would complicate artifact recovery.

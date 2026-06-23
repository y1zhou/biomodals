# Defer PPIFlow stage resource tuning

PPIFlow stage-specific remote wrappers start with the current workflow resource defaults. Separate wrappers preserve a clean path for later CPU, memory, timeout, GPU, and mount tuning, but this refactor should not guess resource settings before real stage telemetry exists.

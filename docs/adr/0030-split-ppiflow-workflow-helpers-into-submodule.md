# Split PPIFlow workflow helpers into a submodule

PPIFlow-local manifest, table, staging, and coordinator helpers live under a `biomodals.workflow.ppiflow` submodule, while `ppiflow_workflow.py` remains the public workflow entrypoint discovered by the CLI and catalog. This keeps the top-level workflow module focused on DAG assembly and node contracts instead of absorbing all candidate-manifest and stage-coordinator mechanics.

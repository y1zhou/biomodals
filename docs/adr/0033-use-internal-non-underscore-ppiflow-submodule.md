# Use an internal non-underscore PPIFlow submodule

PPIFlow helpers live under `biomodals.workflow.ppiflow` rather than `_ppiflow`. The name is clearer and matches the workflow domain, but the submodule remains workflow-internal with a minimal or empty `__all__`; user-facing workflow access stays through `ppiflow_workflow.py` and the CLI/catalog.

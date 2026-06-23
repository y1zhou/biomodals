# Keep PPIFlow node classes in the workflow entrypoint

PPIFlow node classes stay in `ppiflow_workflow.py` because they define the visible workflow DAG contract. Helper internals for manifests, tables, staging, and candidate-wide coordination move into `biomodals.workflow.ppiflow`, but the public entrypoint remains the place to read node contracts and DAG assembly.

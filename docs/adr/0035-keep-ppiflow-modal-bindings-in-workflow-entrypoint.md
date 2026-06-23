# Keep PPIFlow Modal bindings in the workflow entrypoint

PPIFlow Modal decorators, app registration, and app-bound remote helper functions stay in `ppiflow_workflow.py`. The `biomodals.workflow.ppiflow` submodule provides pure or near-pure helper logic for manifests, tables, staging, and coordinator mechanics so importing helper modules does not create hidden Modal app registration side effects and helper tests can run without Modal bindings.

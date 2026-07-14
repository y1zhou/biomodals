# GROMACS API Service App Deviation

`src/biomodals/app/service/gromacs_api_app.py` is a discoverable Modal app but
does not define the usual `submit_<tool>_task` local entrypoint. It is a
long-lived ASGI control plane: humans and clients submit work through its HTTP
routes, while the included `run_gromacs_job` remote function is the reusable
compute boundary.

Adding a local task entrypoint to this wrapper would duplicate the existing
GROMACS CLI and imply that invoking it starts the web service. Develop and
deploy the service with `modal serve` and `modal deploy`; use the existing
`gromacs` app entrypoint for one-off CLI simulations.

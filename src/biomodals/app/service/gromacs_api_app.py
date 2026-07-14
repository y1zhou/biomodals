"""Serve the GROMACS job API with FastAPI on Modal.

FastAPI upstream: <https://github.com/fastapi/fastapi>.

The API accepts one PDB upload, returns a detached job id, and exposes polling,
result-manifest, and cancellation endpoints. Requests require Modal proxy-token
headers. Large simulation files remain in the ``Gromacs-outputs`` Volume; the
result endpoint returns their provider-neutral file manifest.

Develop with
``uv run modal serve -m biomodals.app.service.gromacs_api_app`` and deploy with
``uv run modal deploy -m biomodals.app.service.gromacs_api_app``. The included
GROMACS app supplies the compute functions and may build its pinned container
images on first deployment. Create request credentials with
``uv run modal workspace proxy-tokens create``.

Modal retains detached call results for seven days. The response also returns
the stable GROMACS ``run_name`` so persistent Volume artifacts remain locatable.
The coordinator is still subject to Modal's 24-hour function limit; jobs that
outgrow it can be resumed with the existing GROMACS CLI and returned run name.
"""

import os

import modal

from biomodals.app.bioinfo import gromacs_app
from biomodals.helper import patch_image_for_helper
from biomodals.schema import AppConfig

FASTAPI_VERSION = "0.139.0"
CONF = AppConfig(
    name="GromacsAPI",
    package_name="fastapi",
    repo_url="https://github.com/fastapi/fastapi",
    version=FASTAPI_VERSION,
    python_version="3.13",
    timeout=150,
    tags={"group": "service"},
)

api_image = (
    modal.Image
    .debian_slim(python_version=CONF.python_version)
    .uv_pip_install(f"fastapi[standard]=={FASTAPI_VERSION}")
    .env(CONF.default_env)
    .pipe(patch_image_for_helper)
    .add_local_python_source("biomodals.service")
)

app = modal.App(CONF.name, image=api_image, tags=CONF.tags).include(
    gromacs_app.app,
    inherit_tags=False,
)
job_registry = modal.Dict.from_name(
    os.environ.get("BIOMODALS_GROMACS_JOB_DICT", "biomodals-gromacs-api-jobs"),
    create_if_missing=True,
)


@app.function(
    image=api_image,
    cpu=0.25,
    memory=(512, 2048),
    timeout=CONF.timeout,
    max_containers=5,
)
@modal.concurrent(max_inputs=10)
@modal.asgi_app(requires_proxy_auth=True)
def api():
    """Return the proxy-authenticated FastAPI application."""
    from biomodals.service.gromacs_api import create_app
    from biomodals.service.modal_gromacs import ModalGromacsBackend

    return create_app(
        ModalGromacsBackend(gromacs_app.run_gromacs_job, job_registry),
        trusted_proxy_auth=True,
    )

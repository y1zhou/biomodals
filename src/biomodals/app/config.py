"""Common configurations for Biomodals apps."""

import os
from functools import cached_property
from pathlib import Path

from modal import Volume
from pydantic import BaseModel, computed_field, model_validator

from biomodals.app.constant import MAX_TIMEOUT


class AppConfig(BaseModel):
    """Base configuration model for Biomodals apps."""

    # Metadata
    name: str
    repo_url: str | None = None
    repo_commit_hash: str | None = None
    package_name: str | None = None
    version: str | None = None
    python_version: str | None = None
    tags: dict[str, str] | None = None

    # Runtime configs
    # Model GPU (https://modal.com/docs/guide/gpu)
    # 16GB: T4
    # 24GB: L4, A10G
    # 40GB: A100-40G, A100 (using A100 may cause Modal to auto-upgrade to A100-80G)
    # 48GB: L40S
    # 80GB: A100-80G, H100 (may auto-upgrade to H200, use H100! to avoid)
    # 96GB: RTX-PRO-6000
    # 141GB: H200
    # 180GB: B200 (B200+ may auto-upgrade to B300, which requires CUDA13.0+)
    gpu: str = "A10G"
    # https://modal.com/docs/guide/cuda
    cuda_version: str = "cu128"
    # Default execution timeout in seconds (https://modal.com/docs/guide/timeouts)
    timeout: int = int(os.environ.get("TIMEOUT", "1800"))
    # Location to cache model weights and other large artifacts
    model_volume_mountpoint: str = "/biomodals-store"
    # Location to mount output volume (if in use)
    output_volume_mountpoint: str = "/biomodals-outputs"

    def get_out_volume(self) -> Volume:
        """Volume for storing outputs."""
        vol_name = f"{self.name}-outputs"
        return Volume.from_name(vol_name, create_if_missing=True, version=2)

    @computed_field
    @cached_property
    def default_env(self) -> dict[str, str]:
        """Environment variables to set in the runtime image."""
        model_cache_dir = Path(self.model_volume_mountpoint).resolve()
        return {
            "UV_COMPILE_BYTECODE": "1",  # slower image build, faster runtime
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HOME": str(model_cache_dir / "huggingface"),
            "TORCH_HOME": str(model_cache_dir / "torch"),
            "UV_TORCH_BACKEND": self.cuda_version,
        }

    @computed_field
    @cached_property
    def model_dir(self) -> Path:
        """Directory to store model weights."""
        return Path(self.model_volume_mountpoint).resolve() / self.name

    @computed_field
    @cached_property
    def git_clone_dir(self) -> Path:
        """Directory to store cloned Git repositories."""
        return Path(f"/opt/{self.name}")

    @computed_field
    @cached_property
    def cuda_version_numeric(self) -> str:
        """Numeric CUDA version, e.g., '128' for 'cu128'.

        https://github.com/astral-sh/uv/blob/main/crates/uv-torch/src/backend.rs
        """
        if not self.cuda_version.startswith("cu"):
            return ""

        available_uv_backends = {
            "130",
            "129",
            "128",
            "126",
            "125",
            "124",
            "123",
            "122",
            "121",
            "120",
            "118",
            "117",
            "116",
            "115",
            "114",
            "113",
            "112",
            "111",
            "110",
            "102",
            "101",
            "100",
            "92",
            "91",
            "90",
        }

        if (cuda_ver := self.cuda_version[2:]) not in available_uv_backends:
            raise ValueError(
                f"CUDA version {self.cuda_version} is not supported by UV. "
                f"Available versions: {available_uv_backends}"
            )
        return f"{cuda_ver[:-1]}.{cuda_ver[-1]}.0"

    @model_validator(mode="after")
    def ensure_package_info(self):
        """Ensure that the package information is complete."""
        if self.repo_url is None and self.package_name is None:
            raise ValueError(
                "At least one of 'repo_url' or 'package_name' must be provided."
            )
        if self.repo_commit_hash is None and self.version is None:
            raise ValueError(
                "Provide 'repo_commit_hash' or 'version' for reproducibility."
            )
        return self

    @model_validator(mode="after")
    def ensure_cuda_gpu_compatibility(self):
        """Ensure that the specified CUDA version is compatible with the GPU."""
        if not self.cuda_version.startswith("cu"):
            raise ValueError("CUDA version must start with 'cu', e.g., 'cu128'.")

        is_cu12 = self.cuda_version.startswith("cu12")
        if is_cu12 and self.gpu.startswith("B200+"):
            raise ValueError("CUDA 12.x is not compatible with 'B200+ / B300' GPU.")

        return self

    @model_validator(mode="after")
    def ensure_timeout_within_range(self):
        """Ensure that the specified timeout is within a reasonable range."""
        # between 1 second and 24 hours
        self.timeout = max(1, min(self.timeout, MAX_TIMEOUT))
        return self

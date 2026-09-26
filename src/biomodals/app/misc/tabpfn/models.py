"""Immutable native TabPFN identities and explicit, separate weight provisioning."""

from pathlib import Path

from biomodals.helper.artifacts import file_matches_sha256

TABPFN_VERSION = "9.0.0"
MODEL_REPOSITORY = "Prior-Labs/tabpfn_3_5"
MODEL_REVISION = "06bf2ba35c80a92a3b9abb436b99cf49e7a0365e"
MODEL_FILENAME = "tabpfn-v3.5-20260909.safetensors"
MODEL_SHA256 = "ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3"
MODEL_BYTES = 876_027_932
TORCH_VERSION = "2.11.0"
RUNTIME_IDENTITY = f"tabpfn={TABPFN_VERSION}|model={MODEL_REVISION}|sha256={MODEL_SHA256}|torch={TORCH_VERSION}|sklearn=1.9.1|schema=1"


def checkpoint(root: Path) -> Path:
    """Reject missing/corrupt weights rather than trigger implicit native download."""
    path = root / MODEL_REVISION / MODEL_FILENAME
    if not file_matches_sha256(path, MODEL_BYTES, MODEL_SHA256):
        raise ValueError(
            "Pinned TabPFN checkpoint is missing or invalid; prepare the environment first"
        )
    return path


def provision_checkpoint(root: Path) -> Path:
    """Download an immutable snapshot only in an explicitly writable setup stage."""
    from huggingface_hub import hf_hub_download  # type: ignore[ty:unresolved-import]

    target = root / MODEL_REVISION
    target.mkdir(parents=True, exist_ok=True)
    existing = target / MODEL_FILENAME
    if not file_matches_sha256(existing, MODEL_BYTES, MODEL_SHA256):
        hf_hub_download(
            repo_id=MODEL_REPOSITORY,
            revision=MODEL_REVISION,
            filename=MODEL_FILENAME,
            local_dir=target,
            force_download=True,
        )
    return checkpoint(root)


def native_regressor(
    root: Path,
    *,
    categorical_indices: list[int] | None = None,
    seed: int = 0,
    n_estimators: int = 8,
    device: str = "cuda",
):
    """Load local frozen weights; no user-fitted state survives this invocation."""
    path = checkpoint(root)
    import torch  # type: ignore[ty:unresolved-import]

    from tabpfn import TabPFNRegressor  # type: ignore[ty:unresolved-import]

    return TabPFNRegressor(
        model_path=path,
        n_estimators=n_estimators,
        random_state=seed,
        device=device,
        inference_precision=torch.float32,
        categorical_features_indices=categorical_indices,
        fit_mode="fit_preprocessors",
        inference_config={"TRANSFORM_TEXT": False, "TRANSFORM_DATES": False},
    )

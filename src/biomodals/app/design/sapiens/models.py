"""Immutable model identities and image-build helper for the Sapiens app."""

from dataclasses import dataclass
from pathlib import Path

MODEL_ROOT = Path("/opt/sapiens-models")


@dataclass(frozen=True, slots=True)
class SapiensIdentity:
    """Pinned packages and model snapshots that define scientific behavior."""

    sapiens_version: str = "1.1.0"
    abnumber_version: str = "0.3.2"
    anarci_version: str = "2020.04.23"
    torch_version: str = "2.6.0"
    transformers_version: str = "4.53.3"
    huggingface_hub_version: str = "0.33.4"
    tokenizers_version: str = "0.21.4"
    safetensors_version: str = "0.8.0"
    numpy_version: str = "1.26.4"
    pandas_version: str = "2.1.4"
    biopython_version: str = "1.88"
    hmmer_version: str = "3.4"
    vh_repo: str = "prihodad/biophi-sapiens1-vh"
    vh_revision: str = "2caf2a813361918e9152347d119537e8a48bceb5"
    vl_repo: str = "prihodad/biophi-sapiens1-vl"
    vl_revision: str = "4bb31b6841023645811d869cf0af9d1e0fec805e"
    tokenizer_repo: str = "prihodad/biophi-sapiens1-tokenizer"
    tokenizer_revision: str = "ef6ef1489be29a1cdb0a54f89b6dbf7a47c0ad40"


IDENTITY = SapiensIdentity()
RUNTIME_IDENTITY = "|".join((
    "output-protocol=4",
    f"sapiens={IDENTITY.sapiens_version}",
    f"abnumber={IDENTITY.abnumber_version}",
    f"anarci={IDENTITY.anarci_version}",
    f"torch={IDENTITY.torch_version}",
    f"transformers={IDENTITY.transformers_version}",
    f"huggingface-hub={IDENTITY.huggingface_hub_version}",
    f"tokenizers={IDENTITY.tokenizers_version}",
    f"safetensors={IDENTITY.safetensors_version}",
    f"numpy={IDENTITY.numpy_version}",
    f"pandas={IDENTITY.pandas_version}",
    f"biopython={IDENTITY.biopython_version}",
    f"hmmer={IDENTITY.hmmer_version}",
))


def download_sapiens_models() -> None:
    """Bake the three immutable Hugging Face snapshots into the image."""
    from huggingface_hub import snapshot_download  # type: ignore[ty:unresolved-import]

    snapshots = (
        (IDENTITY.vh_repo, IDENTITY.vh_revision, MODEL_ROOT / "vh"),
        (IDENTITY.vl_repo, IDENTITY.vl_revision, MODEL_ROOT / "vl"),
        (
            IDENTITY.tokenizer_repo,
            IDENTITY.tokenizer_revision,
            MODEL_ROOT / "tokenizer",
        ),
    )
    for repo_id, revision, local_dir in snapshots:
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=local_dir,
        )

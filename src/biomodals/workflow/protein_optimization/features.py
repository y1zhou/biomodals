"""Frozen ESMC600M features, independent full chains and run-local reuse only."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from biomodals.app.design.mutation_ridge.inputs import (
    AMINO_ACIDS,
    Variant,
    variant_sequences,
)
from biomodals.helper.artifacts import file_matches_sha256, sha256_file

ESM_VERSION = "3.4.1.post1"
MODEL_REPOSITORY = "biohub/ESMC-600M"
MODEL_REVISION = "fcc9d36f5fff97d9bb69a36e3252de61cc78c968"
MODEL_SHA256 = "526573c304b7718f9cfd2a8adf79638feb6752eb38056968bf21d9036b7882ce"
MODEL_BYTES = 2_300_201_352
CONFIG_SHA256 = "2494562385804f86db4ba04cba57e4453b1c080ce9eb758443df67ee852bfd31"
FEATURE_WIDTH = 1152
MAX_CHAIN_LENGTH = 2046
EMBEDDING_IDENTITY = f"esmc600m={ESM_VERSION}|revision={MODEL_REVISION}|sha256={MODEL_SHA256}|pool=residue_mean_float32|precision=float32|attention=sdpa|torch=2.11.0|transformers=4.57.6"


def checkpoint_directory(root: Path) -> Path:
    """Verify native local assets without automatic weight or remote-code loading."""
    path = root / MODEL_REVISION
    config = path / "config.json"
    if (
        not file_matches_sha256(path / "model.safetensors", MODEL_BYTES, MODEL_SHA256)
        or not config.is_file()
        or sha256_file(config) != CONFIG_SHA256
    ):
        raise ValueError(
            "Pinned ESMC600M assets are missing or invalid; prepare the environment first"
        )
    return path


def provision_encoder(root: Path) -> Path:
    """Download pinned public assets only in a writable CPU preparation stage."""
    from huggingface_hub import hf_hub_download  # type: ignore[ty:unresolved-import]

    directory = root / MODEL_REVISION
    directory.mkdir(parents=True, exist_ok=True)
    for name, digest in (
        ("model.safetensors", MODEL_SHA256),
        ("config.json", CONFIG_SHA256),
    ):
        path = directory / name
        if not path.is_file() or sha256_file(path) != digest:
            hf_hub_download(
                repo_id=MODEL_REPOSITORY,
                revision=MODEL_REVISION,
                filename=name,
                local_dir=directory,
                force_download=True,
            )
    return checkpoint_directory(root)


def native_encoder(root: Path) -> Callable:
    """Return a bounded local batch encoder; its model never receives labels."""
    path = checkpoint_directory(root)
    import torch  # type: ignore[ty:unresolved-import]
    from esm.models.esmc import (  # type: ignore[ty:unresolved-import]
        EsmcModel,
        EsmcTokenizer,
    )

    tokenizer = EsmcTokenizer()
    model = EsmcModel.from_pretrained(
        path,
        device="cuda",
        dtype=torch.float32,
        attn_implementation="sdpa",
        local_files_only=True,
    ).eval()

    def encode(sequences: Sequence[str]):
        if not 1 <= len(sequences) <= 8 or any(
            not 1 <= len(seq) <= MAX_CHAIN_LENGTH or set(seq) - set(AMINO_ACIDS)
            for seq in sequences
        ):
            raise ValueError(
                "Encoder requires 1–8 complete standard-amino-acid chains of at most 2046 residues"
            )
        inputs = tokenizer(
            list(sequences),
            padding=True,
            truncation=False,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        special = inputs.pop("special_tokens_mask").to(model.device)
        inputs = {name: value.to(model.device) for name, value in inputs.items()}
        mask = inputs["attention_mask"].bool() & ~special.bool()
        lengths = torch.tensor([len(seq) for seq in sequences], device=model.device)
        residue_ids = (inputs["input_ids"] >= 4) & (inputs["input_ids"] <= 23)
        if not torch.equal(
            mask, residue_ids & inputs["attention_mask"].bool()
        ) or not torch.equal(mask.sum(1), lengths):
            raise ValueError(
                "Native tokenization does not match the complete input residues"
            )
        with torch.inference_mode():
            hidden = model(**inputs).last_hidden_state
            pooled = (hidden.float() * mask.unsqueeze(-1)).sum(1) / lengths.unsqueeze(
                -1
            )
        return pooled.cpu().numpy()

    return encode


def variant_features(
    parents: dict[str, str],
    variants: Sequence[Variant],
    encode: Callable,
    *,
    batch_size: int = 4,
):
    """Deduplicate full chains within this call, then concatenate in sorted ID order.

    Sorting sequence work by length bounds padded computation; inverse indices
    restore the exact scientific row and chain identities. Neither embeddings
    nor fitted state are shared with other Jobs.
    """
    import numpy as np

    if not 1 <= batch_size <= 8:
        raise ValueError("Embedding batch size must be between 1 and 8")
    chain_order = tuple(sorted(parents))
    if not chain_order:
        raise ValueError("Parental chains are required")
    indices = np.empty((len(variants), len(chain_order)), dtype=np.int32)
    unique: dict[str, int] = {}
    for row, variant in enumerate(variants):
        chains = variant_sequences(parents, variant)
        for column, chain in enumerate(chain_order):
            sequence = chains[chain]
            indices[row, column] = unique.setdefault(sequence, len(unique))
    sequences = list(unique)
    order = sorted(range(len(sequences)), key=lambda i: (len(sequences[i]), i))
    embeddings = np.empty((len(sequences), FEATURE_WIDTH), dtype=np.float32)
    for offset in range(0, len(order), batch_size):
        batch = order[offset : offset + batch_size]
        values = np.asarray(encode([sequences[i] for i in batch]), dtype=np.float32)
        if values.shape != (len(batch), FEATURE_WIDTH) or not np.isfinite(values).all():
            raise ValueError("Encoder returned invalid full-chain features")
        embeddings[batch] = values
    return embeddings[indices].reshape(
        len(variants), len(chain_order) * FEATURE_WIDTH
    ), chain_order

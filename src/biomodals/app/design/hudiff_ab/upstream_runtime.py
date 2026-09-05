"""Narrow runtime copied beside the pinned upstream antibody sampler."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
import random
import subprocess
from pathlib import Path
from typing import Any

CUBLAS_WORKSPACE_CONFIG = ":4096:8"
CUDA_DETERMINISM_POLICY = "torch-strict+cudnn-deterministic+cublas-4096x8"

HEAVY_REGION_INDEX = [
    region
    for region, count in enumerate((26, 12, 17, 10, 39, 37, 11))
    for _ in range(count)
]
LIGHT_REGION_INDEX = [
    region
    for region, count in enumerate((26, 12, 17, 10, 39, 25, 10))
    for _ in range(count)
]


def _distribution_version(name: str, pinned_fallback: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return pinned_fallback


def _hmmer_version() -> str:
    completed = subprocess.run(  # noqa: S603 - fixed environment executable.
        ["/opt/conda/bin/hmmsearch", "-h"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=30,
    )
    for line in completed.stdout.splitlines():
        if line.startswith("# HMMER "):
            return line.split()[2]
    raise RuntimeError("Could not identify the installed HMMER version")


def _configure_torch_determinism(torch: Any) -> None:
    """Enable the deterministic CUDA behavior promised by the runtime identity."""
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError("HuDiff-Ab requires its pinned cuBLAS workspace policy")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main() -> None:
    """Run one validated pair through the released HuDiff-Ab sampler."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--biomodals-input-json", required=True)
    parser.add_argument("--biomodals-output-json", required=True)
    parser.add_argument("--ckpt", required=True)
    args = parser.parse_args()

    import orjson

    request = orjson.loads(Path(args.biomodals_input_json).read_bytes())
    pair = request["pair"]
    candidate_count = request["candidate_count"]
    seed = request["seed"]
    sampling_order = request["sampling_order"]
    upstream_dropout = request["upstream_inference_dropout"]
    if (
        not isinstance(pair, dict)
        or not isinstance(pair.get("id"), str)
        or not isinstance(pair.get("vh"), str)
        or not isinstance(pair.get("vl"), str)
        or type(candidate_count) is not int
        or not 1 <= candidate_count <= 10
        or type(seed) is not int
        or not 0 <= seed <= 2**32 - 1
        or sampling_order not in {"shuffle", "left_to_right"}
        or type(upstream_dropout) is not bool
    ):
        raise ValueError("invalid Biomodals HuDiff-Ab runtime request")

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
    os.environ["HUDIFF_UPSTREAM_INFERENCE_DROPOUT"] = "1" if upstream_dropout else "0"
    import numpy as np
    import torch  # type: ignore[ty:unresolved-import]
    from abnumber import Chain  # type: ignore[ty:unresolved-import]
    from dataset.preprocess import (  # type: ignore[ty:unresolved-import]
        HEAVY_POSITIONS_dict,
        LIGHT_POSITIONS_dict,
    )
    from model.encoder.model import AntiTFNet  # type: ignore[ty:unresolved-import]
    from sample import (  # type: ignore[ty:unresolved-import]
        batch_input_element,
        get_input_element,
    )

    _configure_torch_determinism(torch)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    heavy = Chain(pair["vh"], scheme="imgt")
    light = Chain(pair["vl"], scheme="imgt")
    if heavy.chain_type != "H" or light.chain_type not in {"K", "L"}:
        raise ValueError("HuDiff-Ab requires one heavy and one kappa/lambda chain")

    _, input_tokens, _, _ = get_input_element(heavy.seq, light.seq, pad_region=0)
    if input_tokens.count("-") != len(input_tokens) - len(pair["vh"]) - len(pair["vl"]):
        raise ValueError(
            "HuDiff-Ab input contains positions outside its fixed IMGT grid"
        )
    (
        sampled_tokens,
        regions,
        chain_types,
        _batch_indices,
        mutable_indices,
        tokenizer,
    ) = batch_input_element(
        heavy.seq,
        light.seq,
        batch_size=candidate_count,
        pad_region=0,
        finetune=True,
    )
    if sampling_order == "shuffle":
        np.random.shuffle(mutable_indices)

    checkpoint = torch.load(args.ckpt, map_location="cpu")
    config = checkpoint["pretrain_config"]
    model = AntiTFNet(**config.model)
    model.load_state_dict(checkpoint["model"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for position in mutable_indices:
            prediction = model(
                sampled_tokens.to(device),
                regions.to(device),
                chain_types.to(device),
            )
            probabilities = torch.nn.functional.softmax(
                prediction[:, position, : len(tokenizer.toks) - 1], dim=1
            )
            sampled = torch.multinomial(probabilities, num_samples=1)
            sampled_tokens[:, position] = sampled.squeeze(1).cpu()

    heavy_tokens = sampled_tokens[:, :152]
    light_tokens = sampled_tokens[:, 152:]
    attempts = []
    for index in range(candidate_count):
        attempts.append({
            "attempt_index": index + 1,
            "vh": tokenizer.idx2seq(heavy_tokens[index]),
            "vl": tokenizer.idx2seq(light_tokens[index]),
            "vh_aligned": tokenizer.idx2seq_pad(heavy_tokens[index]),
            "vl_aligned": tokenizer.idx2seq_pad(light_tokens[index]),
        })

    output = {
        "schema_version": 1,
        "id": pair["id"],
        "pair_seed": seed,
        "sampling_order": sampling_order,
        "upstream_inference_dropout": upstream_dropout,
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "runtime_versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "torch_cudnn": torch.backends.cudnn.version(),
            "numpy": np.__version__,
            "sequence_models": importlib.metadata.version("sequence-models"),
            "abnumber": importlib.metadata.version("abnumber"),
            "anarci": _distribution_version("anarci", "2020.04.23"),
            "hmmer": _hmmer_version(),
            "biopython": importlib.metadata.version("biopython"),
            "pandas": importlib.metadata.version("pandas"),
            "scipy": importlib.metadata.version("scipy"),
            "easydict": importlib.metadata.version("easydict"),
            "einops": importlib.metadata.version("einops"),
            "pyyaml": importlib.metadata.version("pyyaml"),
            "tqdm": importlib.metadata.version("tqdm"),
            "cuda_determinism_policy": CUDA_DETERMINISM_POLICY,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "torch_deterministic_algorithms": str(
                torch.are_deterministic_algorithms_enabled()
            ).lower(),
            "torch_cudnn_deterministic": str(
                torch.backends.cudnn.deterministic
            ).lower(),
            "torch_cudnn_benchmark": str(torch.backends.cudnn.benchmark).lower(),
        },
        "input_vh_aligned": "".join(input_tokens[:152]),
        "input_vl_aligned": "".join(input_tokens[152:]),
        "mutable_indices": [int(position) for position in mutable_indices],
        "region_indices": [int(value) for value in regions[0].tolist()],
        "vh_positions": _position_grid(HEAVY_POSITIONS_dict, 152),
        "vl_positions": _position_grid(LIGHT_POSITIONS_dict, 139),
        "attempts": attempts,
    }
    output_path = Path(args.biomodals_output_json)
    temporary = output_path.with_suffix(".part")
    temporary.write_bytes(orjson.dumps(output, option=orjson.OPT_SORT_KEYS))
    os.replace(temporary, output_path)


def _position_grid(position_dict: dict[str, int], length: int) -> list[str | None]:
    result: list[str | None] = [None] * length
    for label, index in position_dict.items():
        result[index] = label
    return result


if __name__ == "__main__":
    main()

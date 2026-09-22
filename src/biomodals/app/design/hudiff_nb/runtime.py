"""Pinned native initialization and fixed-ten, carry-over nanobody sampling."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import orjson

from biomodals.app.design.vhh import VHHInput

BATCH_SIZE = 10


def protect_tokens(
    parent: VHHInput, input_grid: str, tokens: Any, mutable: Any, tokenizer: Any
) -> list[int]:
    """Restore protected native tokens before removing their positions from sampling."""
    occupied = [index for index, residue in enumerate(input_grid) if residue != "-"]
    protected_grid = {occupied[index] for index in parent.protected_indices}
    original_tokens = tokenizer.seq2idx(list(input_grid))
    for index in protected_grid:
        tokens[:, index] = original_tokens[index]
    return [index for index in mutable if index not in protected_grid]


def native_grid(sequence: str, positions: dict[str, int]) -> str:
    """Reject silent native truncation, unsupported positions or non-heavy input."""
    from anarci import anarci  # type: ignore[ty:unresolved-import]

    numbered, details, _ = anarci(
        [("vhh", sequence)], scheme="imgt", output=False, allow={"H"}
    )
    if not numbered[0] or len(numbered[0]) != 1 or details[0][0]["chain_type"] != "H":
        raise ValueError("HuDiff-Nb requires exactly one native-numbered VH domain")
    grid = ["-"] * len(positions)
    for (number, insertion), residue in numbered[0][0][0]:
        if residue == "-":
            continue
        label = f"{number}{insertion.strip()}"
        if label not in positions or grid[positions[label]] != "-":
            raise ValueError("VH has unsupported or duplicate native IMGT positions")
        grid[positions[label]] = residue
    aligned = "".join(grid)
    if aligned.replace("-", "") != sequence:
        raise ValueError("Native numbering changed the prepared VH")
    return aligned


def sample_attempts(
    *,
    model: Any,
    tokens: Any,
    regions: Any,
    mutable: Any,
    tokenizer: Any,
    count: int,
    device: str,
) -> list[dict[str, Any]]:
    """Preserve native token carry-over and consume every draw without refill."""
    import torch  # type: ignore[ty:unresolved-import]

    attempts = []
    while len(attempts) < count:
        with torch.no_grad():
            for position in mutable:
                prediction = model(
                    tokens.to(device), regions.to(device), H_chn_type=None
                )
                probabilities = torch.nn.functional.softmax(
                    prediction[:, position, : len(tokenizer.toks) - 1], dim=1
                )
                sampled = torch.multinomial(probabilities, num_samples=1)
                tokens[:, position] = sampled.squeeze().cpu()
        for row in tokens:
            if len(attempts) == count:
                break
            attempts.append({
                "attempt_index": len(attempts) + 1,
                "sequence": tokenizer.idx2seq(row),
                "aligned_sequence": tokenizer.idx2seq_pad(row),
            })
    return attempts


def run(request: dict[str, Any], checkpoint_path: str) -> dict[str, Any]:
    """Load the released fine-tuned models with explicit seed and extra protection."""
    parent = VHHInput.model_validate(request["parent"])
    count, seed = request["candidate_count"], request["seed"]
    if type(count) is not int or not 1 <= count <= 25:
        raise ValueError("candidate_count must be between 1 and 25")
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an unsigned 32-bit integer")

    import numpy as np
    import torch  # type: ignore[ty:unresolved-import]
    from dataset.preprocess import (  # type: ignore[ty:unresolved-import]
        HEAVY_POSITIONS_dict,
    )
    from model.nanoencoder.abnativ_model import (  # type: ignore[ty:unresolved-import]
        AbNatiV_Model,
    )
    from model.nanoencoder.model import (  # type: ignore[ty:unresolved-import]
        NanoAntiTFNet,
        NanoInfillingFramework,
    )
    from nanobody_scripts.nanosample import (  # type: ignore[ty:unresolved-import]
        batch_input_element,
        get_multi_model_state,
    )
    from utils.misc import seed_all  # type: ignore[ty:unresolved-import]
    from utils.tokenizer import Tokenizer  # type: ignore[ty:unresolved-import]

    from biomodals.app.design.hudiff_ab.upstream_runtime import (
        _configure_torch_determinism,
    )

    input_grid = native_grid(parent.sequence, HEAVY_POSITIONS_dict)
    _configure_torch_determinism(torch)
    seed_all(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    if config.name != "infilling":
        raise ValueError("Expected the pinned HuDiff-Nb infilling checkpoint")
    abnativ_state, _, infilling_state = get_multi_model_state(checkpoint)
    # Keep BOTH native initializations in their released order. Skipping the
    # training scorer here would change RNG state before stochastic sampling.
    abnativ = AbNatiV_Model(checkpoint["abnativ_params"])
    abnativ.load_state_dict(abnativ_state)
    abnativ.to(device)
    infilling = NanoAntiTFNet(**checkpoint["infilling_params"])
    infilling.load_state_dict(infilling_state)
    infilling.to(device)
    config.model.update({
        "equal_weight": True,
        "vhh_nativeness": False,
        "human_threshold": None,
        "human_all_seq": False,
        "temperature": False,
    })
    framework = NanoInfillingFramework(
        config.model,
        {"abnativ": abnativ, "infilling": infilling, "target_infilling": infilling},
        Tokenizer(),
    )
    model = framework.infilling_pretrain.eval()
    tokens, regions, mutable, tokenizer = batch_input_element(
        parent.sequence, inpaint_sample=True, batch_size=BATCH_SIZE
    )
    mutable = np.array(protect_tokens(parent, input_grid, tokens, mutable, tokenizer))
    np.random.shuffle(mutable)
    attempts = sample_attempts(
        model=model,
        tokens=tokens,
        regions=regions,
        mutable=mutable,
        tokenizer=tokenizer,
        count=count,
        device=device,
    )
    for attempt in attempts:
        try:
            parent.validate_candidate(attempt["sequence"])
            aligned = attempt["aligned_sequence"]
            if len(aligned) != len(input_grid) or any(
                (old == "-") != (new == "-") or (index not in mutable and old != new)
                for index, (old, new) in enumerate(
                    zip(input_grid, aligned, strict=True)
                )
            ):
                raise ValueError(
                    "Generated sequence changed native protected positions"
                )
            if native_grid(attempt["sequence"], HEAVY_POSITIONS_dict) != aligned:
                raise ValueError("Generated tokens disagree with native numbering")
            attempt["error"] = None
        except ValueError as error:
            attempt["error"] = str(error)
    return {
        "input_sequence": parent.sequence,
        "input_aligned": input_grid,
        "seed": seed,
        "batch_size": BATCH_SIZE,
        "mutable_indices": [int(index) for index in mutable],
        "attempts": attempts,
    }


def main() -> None:
    """Run in an isolated process so per-parent RNG state never leaks."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    sys.path.insert(0, "/opt/HuDiff")
    result = run(orjson.loads(Path(args.input).read_bytes()), args.checkpoint)
    Path(args.output).write_bytes(orjson.dumps(result))


if __name__ == "__main__":
    main()

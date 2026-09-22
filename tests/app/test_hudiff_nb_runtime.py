"""Native sampling-loop oracle without checkpoints, Torch installation or Modal."""

from __future__ import annotations

import ast
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import orjson
import pytest

from biomodals.app.design.hudiff_nb import patches, runtime, worker
from biomodals.app.design.vhh import VHHInput


class Tensor(np.ndarray):
    """Minimal CPU tensor boundary; the two loops share identical numerical work."""

    def to(self, _device):
        """Keep the oracle CPU-only."""
        return self

    def cpu(self):
        """Match the wrapper's explicit CUDA-to-CPU assignment boundary."""
        return self


def tensor(value):
    """Wrap a deterministic small array, not a neural model implementation."""
    return np.asarray(value).view(Tensor)


class Tokenizer:
    """Observe decoding calls while preserving duplicates."""

    toks = "ACDE?"

    def __init__(self):
        """Retain each decoded row in encounter order."""
        self.decoded = []

    def idx2seq(self, values):
        """Decode every attempt before any native duplicate filtering."""
        sequence = self.idx2seq_pad(values)
        self.decoded.append(sequence)
        return sequence

    def idx2seq_pad(self, values):
        """Use the same tokens for this gap-free loop fixture."""
        return "".join(self.toks[int(value)] for value in values)


@pytest.mark.parametrize("count", [1, 10, 11, 25])
def test_fixed_batch_carry_over_matches_pinned_native_loop(
    count, monkeypatch, tmp_path
):
    """Compare every draw and model input across complete and partial batches."""

    def softmax(values, *, dim):
        weights = np.exp(values - values.max(axis=dim, keepdims=True))
        return weights / weights.sum(axis=dim, keepdims=True)

    def multinomial(values, *, num_samples):
        assert num_samples == 1
        return tensor([[np.random.choice(len(row), p=row)] for row in values])

    torch = SimpleNamespace(
        no_grad=nullcontext,
        nn=SimpleNamespace(functional=SimpleNamespace(softmax=softmax)),
        multinomial=multinomial,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    histories = []
    decodings = []
    for native in (True, False):
        np.random.seed(123)
        history = []

        def model(tokens, _regions, *, H_chn_type, history=history):
            assert tokens.shape == (10, 3) and H_chn_type is None
            history.append(tokens.copy())
            # Depend on previous token values: a remasked second batch diverges.
            logits = np.arange(4)[None, None, :] - tokens[:, :, None]
            return tensor(-(logits**2) / 4)

        tokens = tensor(np.full((10, 3), 3))
        regions, mutable, tokenizer = tensor(np.zeros((10, 3))), [0, 1], Tokenizer()
        if native:
            scope = {
                "sample_number": count,
                "ms_tokenizer": tokenizer,
                "torch": torch,
                "tqdm": lambda values, **_: values,
                "nano_loc": mutable,
                "model": model,
                "nano_pad_token": tokens,
                "nano_pad_region": regions,
                "device": "cpu",
                "save_fpath": tmp_path / "native.csv",
                "pdb_name": "public",
                "duplicated_set": set(),
                "Chain": lambda *_args, **_kwargs: None,
                "logger": SimpleNamespace(info=lambda *_: None),
                "args": SimpleNamespace(sample_number=count),
            }
            fixture = Path(__file__).parents[1] / "fixtures/hudiff_nb_sampling.py"
            exec(compile(fixture.read_text(), str(fixture), "exec"), scope)  # noqa: S102
        else:
            attempts = runtime.sample_attempts(
                model=model,
                tokens=tokens,
                regions=regions,
                mutable=mutable,
                tokenizer=tokenizer,
                count=count,
                device="cpu",
            )
            assert [row["attempt_index"] for row in attempts] == list(
                range(1, count + 1)
            )
            assert len({row["sequence"] for row in attempts}) <= count
        histories.append(history)
        decodings.append(tokenizer.decoded[:count])
    assert decodings[0] == decodings[1]
    assert len(histories[0]) == len(histories[1]) == 2 * ((count + 9) // 10)
    for expected, observed in zip(*histories, strict=True):
        np.testing.assert_array_equal(expected, observed)


def test_native_grid_rejects_truncation_unknown_positions_and_light_chains(monkeypatch):
    """Reject before a neural forward call instead of trusting native truncation."""
    numbering = [((1, ""), "A"), ((2, ""), "C")]
    detail = {"chain_type": "H"}
    fake = SimpleNamespace(anarci=lambda *_, **__: ([[[numbering]]], [[detail]], None))
    monkeypatch.setitem(sys.modules, "anarci", fake)
    assert runtime.native_grid("AC", {"1": 0, "2": 1, "3": 2}) == "AC-"
    with pytest.raises(ValueError, match="prepared VH"):
        runtime.native_grid("ACD", {"1": 0, "2": 1})
    with pytest.raises(ValueError, match="unsupported"):
        runtime.native_grid("AC", {"1": 0})
    detail["chain_type"] = "L"
    with pytest.raises(ValueError, match="VH domain"):
        runtime.native_grid("AC", {"1": 0, "2": 1})


def test_worker_validates_controls_and_preserves_duplicate_attempts(
    tmp_path, monkeypatch
):
    """A no-op remains generation evidence and receives no fabricated native score."""
    monkeypatch.setattr(worker, "prepare_runtime", lambda: None)
    monkeypatch.setattr(worker, "SOURCE_ROOT", tmp_path)
    parent = VHHInput(sequence="ACDE", protected_indices=(1, 2))
    calls = []

    def execute(argv, **_kwargs):
        request = orjson.loads(Path(argv[argv.index("--input") + 1]).read_bytes())
        calls.append(request)
        output = {
            "input_sequence": "ACDE",
            "seed": request["seed"],
            "attempts": [
                {"attempt_index": n, "sequence": "ACDE", "error": None}
                for n in range(1, 3)
            ],
        }
        Path(argv[argv.index("--output") + 1]).write_bytes(orjson.dumps(output))

    monkeypatch.setattr(worker.subprocess, "run", execute)
    result = worker.humanize_vhh(parent=parent.model_dump(), candidate_count=2, seed=7)
    report = orjson.loads(result.outputs[0].storage.data)
    assert [row["sequence"] for row in report["attempts"]] == ["ACDE", "ACDE"]
    assert report["checkpoint_sha256"] == worker.CHECKPOINT.sha256
    assert report["patch_identity"] == patches.patch_identity()
    assert calls[0]["seed"] == 7
    for count in (0, 26, True):
        with pytest.raises(ValueError, match="candidate_count"):
            worker.humanize_vhh(parent=parent.model_dump(), candidate_count=count)
    assert len(calls) == 1
    with pytest.raises(ValueError, match="protected"):
        parent.validate_candidate("ACAE")
    parent.validate_candidate("ACDF")


def test_remote_modules_parse_with_python310():
    """The reused HuDiff image runs 3.10, independently of its coordinator."""
    for module in (runtime, worker, patches):
        ast.parse(Path(module.__file__).read_text(), feature_version=(3, 10))

"""Offline reconstruction and the exact production pooling/masking operations."""

import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from biomodals.app.design.mutation_ridge.inputs import parse_mutations
from biomodals.workflow.protein_optimization import features


def test_full_chain_deduplication_and_stable_identity_across_batches():
    """Changing batch size/order preserves exact rows and independent chain blocks."""
    parents = {"z": "AA", "A": "CMW"}
    variants = [(), parse_mutations("z:A1V"), parse_mutations("z:A2L,A:C1G")]
    calls = []

    def encode(sequences):
        calls.extend(sequences)
        return np.asarray(
            [[sum(map(ord, seq))] * features.FEATURE_WIDTH for seq in sequences],
            dtype=np.float32,
        )

    matrix, order = features.variant_features(parents, variants, encode, batch_size=2)
    assert order == ("A", "z")
    assert len(calls) == len(set(calls)) == 5
    np.testing.assert_array_equal(
        matrix[:, 0], [sum(map(ord, s)) for s in ("CMW", "CMW", "GMW")]
    )
    np.testing.assert_array_equal(
        matrix[:, -1], [sum(map(ord, s)) for s in ("AA", "VA", "AL")]
    )
    other, other_order = features.variant_features(
        dict(reversed(list(parents.items()))), variants, encode, batch_size=1
    )
    assert other_order == order
    np.testing.assert_array_equal(matrix, other)


def test_native_adapter_masks_bos_eos_padding_and_preserves_entire_chain(
    tmp_path, monkeypatch
):
    """Exercise the production arithmetic using a CPU tensor double, not real ESM."""

    class Tensor(np.ndarray):
        def to(self, device):
            return self

        def bool(self):
            return self.astype(bool)

        def float(self):
            return self.astype(np.float32)

        def unsqueeze(self, axis):
            return np.expand_dims(self, axis)

        def cpu(self):
            return self

        def numpy(self):
            return np.asarray(self)

    def tensor(value, device=None):
        return np.asarray(value).view(Tensor)

    class Tokenizer:
        def __call__(self, sequences, **kwargs):
            assert kwargs["truncation"] is False
            ids = [
                [0] + [4 + features.AMINO_ACIDS.index(aa) for aa in seq] + [2]
                for seq in sequences
            ]
            size = max(map(len, ids))
            padded = [row + [1] * (size - len(row)) for row in ids]
            return {
                "input_ids": tensor(padded),
                "attention_mask": tensor([[x != 1 for x in row] for row in padded]),
                "special_tokens_mask": tensor([[x < 4 for x in row] for row in padded]),
            }

    class Model:
        device = "cpu"

        @classmethod
        def from_pretrained(cls, directory, **kwargs):
            assert kwargs["local_files_only"] is True
            assert kwargs["attn_implementation"] == "sdpa"
            return cls()

        def eval(self):
            return self

        def __call__(self, input_ids, attention_mask):
            # Special/padded states would swamp the answer if erroneously included.
            values = np.where(input_ids < 4, 10000, input_ids)
            return SimpleNamespace(
                last_hidden_state=tensor(
                    np.repeat(values[:, :, None], features.FEATURE_WIDTH, axis=2)
                )
            )

    monkeypatch.setattr(features, "checkpoint_directory", lambda root: root)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            float32=np.float32,
            tensor=tensor,
            equal=np.array_equal,
            inference_mode=nullcontext,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "esm.models.esmc",
        SimpleNamespace(EsmcModel=Model, EsmcTokenizer=Tokenizer),
    )
    encode = features.native_encoder(tmp_path)
    batch = encode(["AC", "W"])
    expected = [
        np.mean([4 + features.AMINO_ACIDS.index(aa) for aa in seq])
        for seq in ("AC", "W")
    ]
    np.testing.assert_array_equal(batch[:, 0], expected)
    np.testing.assert_array_equal(batch[1], encode(["W"])[0])
    with pytest.raises(ValueError, match="2046"):
        encode(["A" * 2047])


def test_invalid_embedding_payload_is_not_published():
    """Bad native embeddings fail before they can become regression features."""
    with pytest.raises(ValueError, match="invalid full-chain"):
        features.variant_features(
            {"A": "AA"},
            [()],
            lambda rows: np.full((len(rows), features.FEATURE_WIDTH), np.nan),
        )

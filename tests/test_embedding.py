"""Numerical adapter tests with fake tensors; no torch or checkpoint imports."""
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from itercomp.embedding import BGEM3Encoder
from tests.test_core import config


class Tensor:
    def __init__(self, values):
        self.values = np.array(values)

    def float(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values.tolist()

    def squeeze(self, axis):
        return Tensor(np.squeeze(self.values, axis=axis))


class Batch(dict):
    def to(self, device):
        return self


class Tokenizer:
    cls_token_id, eos_token_id, pad_token_id, unk_token_id = 0, 2, 1, 3

    def __call__(self, texts, **kwargs):
        ids = [[0, 10, 10, 11, 2] for _ in texts]
        if kwargs.get("truncation"):
            ids = [row[:kwargs["max_length"]] for row in ids]
        if kwargs.get("return_tensors"):
            return Batch(input_ids=Tensor(ids))
        return {"input_ids": ids}


class Model:
    def __init__(self):
        self.calls = 0

    def __call__(self, inputs, **kwargs):
        self.calls += 1
        assert kwargs == {"return_dense": True, "return_sparse": True, "return_colbert_vecs": False}
        ids = inputs["input_ids"].values
        weights = [[[v] for v in [3, .2, .8, .5, 9][:len(row)]] for row in ids]
        return {"dense_vecs": Tensor([[.6, .8] for _ in ids]), "sparse_vecs": Tensor(weights)}


class EmbeddingTests(unittest.TestCase):
    def make_encoder(self):
        c = config().embedding
        c.backend = "bge_m3"
        c.batch_size = 2
        encoder = BGEM3Encoder(c)
        encoder.flag_model = SimpleNamespace(tokenizer=Tokenizer(), model=Model())
        return encoder

    def test_official_output_extraction_fixed_batches_no_extra_normalization(self):
        encoder = self.make_encoder()
        fake_torch = SimpleNamespace(inference_mode=nullcontext)
        with patch.dict("sys.modules", {"torch": fake_torch}), patch("itercomp.embedding.synchronize"):
            reps = encoder.encode(["a", "b", "c"], ["p0_s0", "p1_s0", "p2_s0"])
        self.assertEqual(len(reps), 3)
        self.assertEqual(reps[0].dense, (.6, .8))
        self.assertEqual(reps[0].lexical, {"10": .8, "11": .5})
        self.assertEqual(encoder.flag_model.model.calls, 2)
        self.assertEqual(encoder.forward_calls, 2)
        self.assertEqual(encoder.encoded_texts, 3)
        self.assertEqual(encoder.events, [])

    def test_truncation_error_logged_before_forward(self):
        encoder = self.make_encoder()
        encoder.config.max_length = 3
        with patch.dict("sys.modules", {"torch": SimpleNamespace(inference_mode=nullcontext)}), self.assertRaisesRegex(ValueError, "max_length"):
            encoder.encode(["a"], ["p0_s0"])
        self.assertEqual(encoder.flag_model.model.calls, 0)
        self.assertEqual(encoder.events[0]["input_tokens"], 5)
        self.assertEqual(encoder.events[0]["id"], "p0_s0")

    def test_explicit_log_policy_and_reset(self):
        encoder = self.make_encoder()
        encoder.config.max_length = 3
        encoder.config.truncation_policy = "log"
        with patch.dict("sys.modules", {"torch": SimpleNamespace(inference_mode=nullcontext)}), patch("itercomp.embedding.synchronize"):
            reps = encoder.encode(["a"], ["p0_s0"])
        self.assertEqual(len(reps), 1)
        self.assertEqual(encoder.events[0]["retained_tokens"], 3)
        encoder.reset_telemetry()
        self.assertEqual(encoder.events, [])
        self.assertEqual(encoder.forward_calls, 0)


if __name__ == "__main__":
    unittest.main()

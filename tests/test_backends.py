from contextlib import nullcontext
from copy import deepcopy
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np

from itercomp.backends import BackendPool
from itercomp.backends.base import GenerationFailure
from itercomp.backends.openai_compatible import OpenAICompatibleBackend
from itercomp.backends.transformers_backend import TransformersBackend
from itercomp.schemas import CallLedger, GenerationResult
from itercomp.telemetry import check_device
from tests.test_core import config


class FakeBatch(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    chat_template = "fake"
    model_max_length = 128
    pad_token_id = 0
    eos_token_id = 2

    def encode(self, text, **kwargs):
        return list(text.encode())

    def apply_chat_template(self, messages, **kwargs):
        return "<user>" + messages[0]["content"] + "<assistant>"

    def __call__(self, text, **kwargs):
        return FakeBatch(input_ids=np.array([self.encode(text)]), attention_mask=np.ones((1, len(text)), dtype=int))

    def decode(self, ids, **kwargs):
        return "Paris"


class FakeModel:
    config = SimpleNamespace(max_position_embeddings=128, _commit_hash="synthetic-model-revision")

    def eval(self):
        return self

    def requires_grad_(self, value):
        assert value is False

    def generate(self, **kwargs):
        self.generation_kwargs = kwargs
        return np.concatenate([kwargs["input_ids"], np.array([[5, 6, 2]])], axis=-1)


class BackendTests(unittest.TestCase):
    def test_api_usage_and_unknown_usage(self):
        c = config().reader
        c.backend = "openai_compatible"
        c.api_base = "http://example.invalid/v1"
        c.api_send_seed = True
        backend = OpenAICompatibleBackend(c)
        value = {"choices": [{"message": {"content": "Paris"}}], "usage": {"prompt_tokens": 42, "completion_tokens": 3}}
        with patch("itercomp.backends.openai_compatible.urlopen", return_value=io.BytesIO(json.dumps(value).encode())) as request:
            result = backend.generate([{"role": "user", "content": "Q?"}])
            payload = json.loads(request.call_args.args[0].data)
            self.assertEqual(payload["temperature"], 0)
            self.assertEqual(payload["seed"], 42)
            self.assertEqual(result.input_tokens, 42)
            self.assertEqual(result.output_tokens, 3)
        value.pop("usage")
        with patch("itercomp.backends.openai_compatible.urlopen", return_value=io.BytesIO(json.dumps(value).encode())):
            result = backend.generate([{"role": "user", "content": "Q?"}])
            self.assertIsNone(result.input_tokens)
            self.assertIsNone(result.output_tokens)
        self.assertIsNone(backend.count_text("abc"))
        with patch("itercomp.backends.openai_compatible.urlopen", side_effect=TimeoutError()), self.assertRaises(GenerationFailure):
            backend.generate([{"role": "user", "content": "Q?"}])

    def test_unknown_usage_propagates(self):
        ledger = CallLedger()
        self.assertEqual(ledger.total("input_tokens"), 0)
        ledger.add(GenerationResult("A", 10, 2, .1))
        ledger.add(GenerationResult("B", None, None, .2))
        self.assertIsNone(ledger.total("input_tokens"))

    def test_transformers_configuration_generation_and_context_limit_with_fake_modules(self):
        fake_torch = SimpleNamespace(float16="fp16", bfloat16="bf16", float32="fp32", inference_mode=nullcontext,
                                     cuda=SimpleNamespace(OutOfMemoryError=MemoryError))
        model, tokenizer = FakeModel(), FakeTokenizer()
        constructor = Mock(return_value=model)
        bnb = Mock(side_effect=lambda **kwargs: kwargs)
        fake_transformers = SimpleNamespace(AutoModelForCausalLM=SimpleNamespace(from_pretrained=constructor),
                                             AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=tokenizer)),
                                             BitsAndBytesConfig=bnb, set_seed=Mock())
        c = deepcopy(config().reader)
        c.backend = "transformers"
        c.quantization = "nf4"
        c.device = "cuda:0"
        c.dtype = "float16"
        c.compute_dtype = "float16"
        c.max_new_tokens = 4
        with patch.dict("sys.modules", {"torch": fake_torch, "transformers": fake_transformers}), patch("itercomp.backends.transformers_backend.check_device"), patch("itercomp.backends.transformers_backend.synchronize"), patch("itercomp.backends.transformers_backend.check_quantization_runtime", return_value={"native_cuda_backend": True}):
            backend = TransformersBackend(c)
            result = backend.generate([{"role": "user", "content": "Q?"}])
            self.assertEqual(result.text, "Paris")
            self.assertEqual(result.output_tokens, 3)
            kwargs = constructor.call_args.kwargs
            self.assertEqual(kwargs["device_map"], {"": "cuda:0"})
            self.assertEqual(kwargs["quantization_config"]["bnb_4bit_quant_type"], "nf4")
            self.assertEqual(kwargs["quantization_config"]["bnb_4bit_compute_dtype"], "fp16")
            self.assertNotIn("temperature", model.generation_kwargs)
            self.assertNotIn("top_p", model.generation_kwargs)
            self.assertEqual(model.generation_kwargs["pad_token_id"], 0)
            self.assertEqual(result.input_tokens, len("<user>Q?<assistant>"))
            with self.assertRaisesRegex(ValueError, "not truncated"):
                backend.generate([{"role": "user", "content": "x" * 200}])
            c.do_sample, c.temperature = True, .7
            backend.generate([{"role": "user", "content": "Q?"}])
            self.assertEqual(model.generation_kwargs["temperature"], .7)
            self.assertEqual(model.generation_kwargs["top_p"], 1)

    def test_identical_backend_pool_sharing(self):
        pool = BackendPool()
        c = config().reader
        self.assertIs(pool.get(c), pool.get(deepcopy(c)))
        pool.close()

    def test_hardware_errors_are_explicit_without_fallback(self):
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with patch.dict("sys.modules", {"torch": fake_torch}):
            with self.assertRaisesRegex(RuntimeError, "CPU fallback is disabled"):
                check_device("cuda:0", "float16")
            with self.assertRaises(ValueError):
                check_device("cpu", "float16")
            with self.assertRaises(ValueError):
                check_device("cpu", "float32", "nf4")
            check_device("cpu", "float32", "none")


if __name__ == "__main__":
    unittest.main()

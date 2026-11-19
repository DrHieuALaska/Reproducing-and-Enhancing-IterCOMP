"""Explicit synthetic responses; these are never research model outputs."""
import time
from .base import LLMBackend
from ..schemas import GenerationResult


class MockBackend(LLMBackend):
    def __init__(self, responses):
        self.script = list(responses)
        self.responses = list(responses)
        self.messages = []
        self.initialization = {"backend": "mock", "model_initialization_seconds": 0.0,
                               "weight_loading_seconds": 0.0, "tokenizer": "synthetic UTF-8 byte tokens"}

    def begin_sample(self):
        self.responses = list(self.script)

    def count_text(self, text):
        # Exact count for this explicitly synthetic tokenizer, never a pretrained estimate.
        return len(text.encode("utf-8"))

    def generate(self, messages):
        start = time.perf_counter()
        if not self.responses:
            raise RuntimeError("Mock response script exhausted")
        self.messages.append(messages)
        text = self.responses.pop(0)
        prompt = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages) + "<assistant>"
        return GenerationResult(text, self.count_text(prompt), self.count_text(text), time.perf_counter() - start)

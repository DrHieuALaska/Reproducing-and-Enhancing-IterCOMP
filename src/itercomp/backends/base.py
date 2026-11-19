from abc import ABC, abstractmethod
from ..schemas import GenerationResult


class GenerationFailure(RuntimeError):
    """A real generation/request was attempted, but usage may be incomplete."""
    def __init__(self, message, input_tokens=None, latency_seconds=0.0):
        super().__init__(message)
        self.result = GenerationResult("", input_tokens, None, latency_seconds)


class LLMBackend(ABC):
    initialization = None

    def prepare(self):
        pass

    def preflight(self):
        """Validate runtime dependencies before any checkpoint downloads."""
        pass

    def begin_sample(self):
        """Optional per-sample state reset, used only by deterministic synthetic mocks."""
        pass

    @abstractmethod
    def generate(self, messages) -> GenerationResult:
        raise NotImplementedError

    @abstractmethod
    def count_text(self, text) -> int | None:
        raise NotImplementedError

    def close(self):
        pass

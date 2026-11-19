"""Backends are imported only when explicitly selected."""
from dataclasses import asdict
from ..config import canonical_hash


def create_backend(config):
    if config.backend == "mock":
        from .mock import MockBackend
        return MockBackend(config.mock_responses)
    if config.backend == "transformers":
        from .transformers_backend import TransformersBackend
        return TransformersBackend(config)
    if config.backend == "openai_compatible":
        from .openai_compatible import OpenAICompatibleBackend
        return OpenAICompatibleBackend(config)
    raise ValueError(f"Unknown backend: {config.backend}")


class BackendPool:
    """Optional sharing for callers keeping identical controller/reader configs live."""
    def __init__(self):
        self.backends = {}

    def get(self, config):
        key = canonical_hash(asdict(config))
        if key not in self.backends:
            self.backends[key] = create_backend(config)
        return self.backends[key]

    def close(self):
        for backend in self.backends.values():
            backend.close()
        self.backends.clear()

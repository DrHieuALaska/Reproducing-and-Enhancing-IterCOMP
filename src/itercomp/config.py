"""Strict, serializable configuration and stage-specific identities."""
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
import hashlib
import json
import math


@dataclass
class EmbeddingConfig:
    backend: str = "bge_m3"
    model_id: str = "BAAI/bge-m3"
    revision: str | None = None
    device: str = "cuda:0"
    dtype: str = "float16"
    batch_size: int = 16
    max_length: int = 8192
    normalize_embeddings: bool = True
    include_titles: bool = False
    truncation_policy: str = "error"
    cache_dir: str | None = None
    local_files_only: bool = False
    mock_representations: dict = field(default_factory=dict)


@dataclass
class LLMConfig:
    backend: str = "transformers"
    model_id: str = "mistralai/Mistral-7B-Instruct-v0.3"
    revision: str | None = None
    device: str = "cuda:0"
    dtype: str = "float16"
    quantization: str = "nf4"
    compute_dtype: str = "float16"
    double_quantization: bool = True
    attention_implementation: str = "sdpa"
    max_new_tokens: int = 256
    max_context_tokens: int | None = None
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 42
    cache_dir: str | None = None
    local_files_only: bool = False
    api_base: str = "http://localhost:8000/v1"
    api_key_env: str = "ITERCOMP_API_KEY"
    timeout: float = 120.0
    api_send_seed: bool = False
    tokenizer_id: str | None = None
    tokenizer_revision: str | None = None
    chat_template_kwargs: dict = field(default_factory=dict)
    mock_responses: list[str] = field(default_factory=list)


@dataclass
class AlgorithmConfig:
    semantic_weight: float = 0.6
    percentile: float = 90.0
    percentile_method: str = "linear"
    max_iterations: int = 5
    controller_retries: int = 0
    answerable_follow_up_policy: str = "normalize_unused"
    full_rankings: bool = True


@dataclass
class DecompositionConfig:
    strategy: str = "spacy_sentencizer"
    language: str = "en"


@dataclass
class ContextConfig:
    include_titles: bool = False
    separator: str = "\n"
    order: str = "source"


@dataclass
class PromptConfig:
    controller: str | None = None
    reader: str | None = None


@dataclass
class ReferenceTokenizerConfig:
    model_id: str | None = None
    revision: str | None = None
    local_files_only: bool = False


@dataclass
class Config:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    controller: LLMConfig = field(default_factory=LLMConfig)
    reader: LLMConfig = field(default_factory=lambda: LLMConfig(
        model_id="meta-llama/Meta-Llama-3-8B-Instruct", max_new_tokens=64))
    itercomp: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    decomposition: DecompositionConfig = field(default_factory=DecompositionConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    prompts: PromptConfig = field(default_factory=PromptConfig)
    reference_tokenizer: ReferenceTokenizerConfig = field(default_factory=ReferenceTokenizerConfig)
    seed: int = 42
    profile: str = "paper-oriented; checkpoint variants are assumptions"

    def validate(self):
        def positive_int(value, label):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{label} must be a positive integer")
        for value, label in ((self.embedding.batch_size, "batch_size"),
                             (self.embedding.max_length, "embedding max_length"),
                             (self.itercomp.max_iterations, "max_iterations")):
            positive_int(value, label)
        if isinstance(self.itercomp.controller_retries, bool) or not isinstance(self.itercomp.controller_retries, int) or self.itercomp.controller_retries < 0:
            raise ValueError("controller_retries must be a nonnegative integer")
        for value, label in ((self.seed, "seed"), (self.controller.seed, "controller seed"), (self.reader.seed, "reader seed")):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{label} must be a nonnegative integer")
        for value, label in ((self.embedding.normalize_embeddings, "normalize_embeddings"),
                             (self.embedding.include_titles, "embedding include_titles"),
                             (self.context.include_titles, "context include_titles"),
                             (self.itercomp.full_rankings, "full_rankings")):
            if not isinstance(value, bool):
                raise ValueError(f"{label} must be a boolean")
        for value, low, high, label in ((self.itercomp.semantic_weight, 0, 1, "semantic_weight"),
                                        (self.itercomp.percentile, 0, 100, "percentile")):
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{label} outside [{low}, {high}]")
        if self.itercomp.percentile_method != "linear":
            raise ValueError("Original profile supports explicit NumPy 'linear' percentiles")
        if self.itercomp.answerable_follow_up_policy not in ("strict", "normalize_unused"):
            raise ValueError("answerable_follow_up_policy must be strict or normalize_unused")
        if self.context.order != "source" or not self.context.separator:
            raise ValueError("Use source ordering and a nonempty separator")
        if self.decomposition.strategy != "spacy_sentencizer":
            raise ValueError("Only non-overlapping spaCy sentencizer is implemented")
        if self.embedding.backend not in ("bge_m3", "mock"):
            raise ValueError("Unknown embedding backend")
        if self.embedding.dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError("Invalid embedding dtype")
        if self.embedding.truncation_policy not in ("error", "log"):
            raise ValueError("embedding truncation_policy must be error or log")
        for llm in (self.controller, self.reader):
            if not all(isinstance(value, bool) for value in (llm.do_sample, llm.double_quantization, llm.local_files_only, llm.api_send_seed)):
                raise ValueError("LLM sampling, quantization, local-file and API-seed flags must be boolean")
            if llm.backend not in ("transformers", "openai_compatible", "mock"):
                raise ValueError(f"Unknown LLM backend: {llm.backend}")
            if llm.quantization not in ("none", "nf4", "int8"):
                raise ValueError("quantization must be none, nf4, or int8")
            if llm.dtype not in ("float32", "float16", "bfloat16") or llm.compute_dtype not in ("float32", "float16", "bfloat16"):
                raise ValueError("Unsupported LLM precision")
            positive_int(llm.max_new_tokens, "max_new_tokens")
            if llm.max_context_tokens is not None:
                positive_int(llm.max_context_tokens, "max_context_tokens")
            if not math.isfinite(llm.temperature) or llm.temperature < 0 or not 0 < llm.top_p <= 1:
                raise ValueError("Invalid generation sampling configuration")
            if llm.do_sample and llm.temperature <= 0:
                raise ValueError("Sampling requires positive temperature")
            if not math.isfinite(llm.timeout) or llm.timeout <= 0:
                raise ValueError("API timeout must be positive")
        return self


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def merge_dict(base, update):
    result = deepcopy(base)
    for key, value in update.items():
        result[key] = merge_dict(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else deepcopy(value)
    return result


def config_from_dict(data):
    data = deepcopy(data)
    sections = {"embedding": EmbeddingConfig, "controller": LLMConfig, "reader": LLMConfig,
                "itercomp": AlgorithmConfig, "decomposition": DecompositionConfig,
                "context": ContextConfig, "prompts": PromptConfig,
                "reference_tokenizer": ReferenceTokenizerConfig}
    unknown = set(data) - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    for name, cls in sections.items():
        if name in data:
            if not isinstance(data[name], dict):
                raise ValueError(f"{name} must be a mapping")
            defaults = asdict(getattr(Config(), name))
            unknown = set(data[name]) - set(defaults)
            if unknown:
                raise ValueError(f"Unknown {name} keys: {sorted(unknown)}")
            data[name] = cls(**merge_dict(defaults, data[name]))
    return Config(**data).validate()


def load_config(path, overrides=None, _seen=None):
    import yaml
    path = Path(path).resolve()
    seen = set() if _seen is None else set(_seen)
    if path in seen:
        raise ValueError("Cyclic configuration inheritance")
    seen.add(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = data.pop("extends", None)
    if parent:
        data = merge_dict(asdict(load_config(path.parent / parent, _seen=seen)), data)
    return config_from_dict(merge_dict(data, overrides or {}))


def compression_settings(config, method, prompt_hashes):
    """Reader and its tokenizer cannot affect compression identity."""
    data = asdict(config)
    names = ("seed", "context") if method in ("raw", "oracle") else (
        "seed", "embedding", "controller", "itercomp", "decomposition", "context")
    return {"method": method, **{k: data[k] for k in names},
            "controller_prompt_sha256": prompt_hashes["controller"] if method == "itercomp" else None}

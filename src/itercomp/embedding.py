"""BGE-M3 official dense/sparse forward pass and a separate synthetic test encoder."""
import math
from pathlib import Path
import time
from .schemas import Representation
from .telemetry import check_device, synchronize, release_memory


class MockEncoder:
    def __init__(self, config):
        self.table = config.mock_representations
        self.config = config
        self.initialization = {"backend": "mock", "weight_loading_seconds": 0.0}
        self.reset_telemetry()

    def prepare(self):
        pass

    def reset_telemetry(self):
        self.events = []
        self.forward_calls = 0
        self.encoded_texts = 0

    def encode(self, texts, labels=None):
        self.forward_calls += math.ceil(len(texts) / self.config.batch_size)
        self.encoded_texts += len(texts)
        result = []
        for text in texts:
            if text not in self.table:
                raise ValueError(f"Synthetic representation not configured for {text!r}")
            value = self.table[text]
            result.append(Representation(tuple(value["dense"]), value.get("lexical", {})))
        return result

    def close(self):
        pass


class BGEM3Encoder:
    def __init__(self, config):
        self.config = config
        self.flag_model = None
        self.initialization = {}
        self.reset_telemetry()

    def reset_telemetry(self):
        self.events = []
        self.forward_calls = 0
        self.encoded_texts = 0

    def prepare(self):
        if self.flag_model is not None:
            return
        start = time.perf_counter()
        import torch
        from FlagEmbedding import BGEM3FlagModel
        from huggingface_hub import snapshot_download
        c = self.config
        check_device(c.device, c.dtype)
        loading_start = time.perf_counter()
        path = Path(c.model_id)
        if not path.is_dir():
            path = Path(snapshot_download(
                repo_id=c.model_id, revision=c.revision, cache_dir=c.cache_dir,
                local_files_only=c.local_files_only,
                allow_patterns=["*.json", "*.safetensors", "pytorch_model.bin", "*.pt",
                                "sentencepiece.bpe.model", "tokenizer.model", "*.txt"]))
        # FlagEmbedding otherwise initializes absent sparse heads randomly. Never allow it.
        for head in ("sparse_linear.pt", "colbert_linear.pt"):
            if not (path / head).is_file():
                raise FileNotFoundError(f"Missing official BGE-M3 head {head} in {path}; supply the complete BGE-M3 checkpoint")
        self.flag_model = BGEM3FlagModel(
            str(path), normalize_embeddings=c.normalize_embeddings,
            use_fp16=c.dtype == "float16", devices=[c.device],
            batch_size=c.batch_size, query_max_length=c.max_length, passage_max_length=c.max_length)
        self.flag_model.model.to(device=c.device, dtype=getattr(torch, c.dtype))
        self.flag_model.model.eval()
        self.flag_model.model.requires_grad_(False)
        synchronize(c.device)
        self.initialization = {"model_initialization_seconds": time.perf_counter() - start,
                               "weight_loading_seconds": time.perf_counter() - loading_start,
                               "resolved_revision": path.name if path.parent.name == "snapshots" else None,
                               "dense_normalized_by_flagembedding": c.normalize_embeddings,
                               "adapter": "official M3 forward; weighted max-per-token sparse extraction; fixed batches, no hidden retry"}

    def encode(self, texts, labels=None):
        self.prepare()
        import torch
        c = self.config
        tokenizer = self.flag_model.tokenizer
        labels = labels or [str(i) for i in range(len(texts))]
        if len(labels) != len(texts):
            raise ValueError("Embedding label count mismatch")
        result = []
        unused = {getattr(tokenizer, name + "_token_id", None) for name in ("cls", "eos", "pad", "unk")}
        for start in range(0, len(texts), c.batch_size):
            batch = texts[start:start + c.batch_size]
            # Measure lengths before optional explicit truncation; source text is always retained.
            untruncated = tokenizer(batch, truncation=False, padding=False)
            for offset, ids in enumerate(untruncated["input_ids"]):
                if len(ids) > c.max_length:
                    event = {"id": labels[start + offset], "input_tokens": len(ids),
                             "retained_tokens": c.max_length, "truncated": True}
                    self.events.append(event)
                    if c.truncation_policy == "error":
                        raise ValueError(f"BGE input exceeds max_length: {event}; explicit truncation_policy=log is required to truncate")
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=c.max_length,
                               return_tensors="pt").to(c.device)
            # Use the official model directly: encode() in FlagEmbedding 1.3.5 contains
            # an unbounded batch-size reduction retry. This adapter makes one forward
            # per configured batch and keeps exactly the official dense/sparse math.
            with torch.inference_mode():
                self.forward_calls += 1
                output = self.flag_model.model(inputs, return_dense=True, return_sparse=True,
                                               return_colbert_vecs=False)
            synchronize(c.device)
            dense = output["dense_vecs"].float().cpu().tolist()
            weights = output["sparse_vecs"].squeeze(-1).float().cpu().tolist()
            token_ids = inputs["input_ids"].cpu().tolist()
            for vector, token_weights, ids in zip(dense, weights, token_ids):
                sparse = {}
                for weight, idx in zip(token_weights, ids):
                    if idx not in unused and weight > 0:
                        token = str(idx)
                        sparse[token] = max(sparse.get(token, 0.0), weight)
                if not all(math.isfinite(v) for v in vector) or not all(math.isfinite(v) for v in token_weights):
                    raise ValueError("Nonfinite BGE-M3 representation")
                result.append(Representation(tuple(vector), sparse))
            self.encoded_texts += len(batch)
        return result

    def close(self):
        self.flag_model = None
        release_memory()


def create_encoder(config):
    return MockEncoder(config) if config.backend == "mock" else BGEM3Encoder(config)

"""Lazy, frozen Transformers generation, with explicit precision and placement."""
import time
from .base import LLMBackend, GenerationFailure
from .quantization import check_quantization_runtime
from ..telemetry import check_device, synchronize, release_memory


class TransformersBackend(LLMBackend):
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.initialization = {}
        self.quantization_runtime = None

    def preflight(self):
        self.quantization_runtime = check_quantization_runtime(self.config)

    def _tokenizer(self):
        if self.tokenizer is None:
            from transformers import AutoTokenizer
            c = self.config
            self.tokenizer = AutoTokenizer.from_pretrained(
                c.tokenizer_id or c.model_id, revision=c.tokenizer_revision or c.revision,
                cache_dir=c.cache_dir, local_files_only=c.local_files_only,
                trust_remote_code=False)
            if not self.tokenizer.chat_template:
                raise ValueError("Checkpoint has no chat template; configure an instruction-tuned tokenizer explicitly")
        return self.tokenizer

    def prepare(self):
        if self.model is not None:
            return
        start = time.perf_counter()
        self.preflight()
        import torch
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig, set_seed
        c = self.config
        check_device(c.device, c.dtype, c.quantization)
        if c.quantization != "none" and c.compute_dtype != c.dtype:
            check_device(c.device, c.compute_dtype, c.quantization)
        if c.attention_implementation == "flash_attention_2" and c.device.startswith("cuda"):
            if torch.cuda.get_device_capability(c.device)[0] < 8:
                raise RuntimeError("FlashAttention-2 requires a newer GPU; use sdpa or eager on T4")
        self._tokenizer()
        set_seed(c.seed)
        kwargs = {"revision": c.revision, "torch_dtype": getattr(torch, c.dtype),
                  "device_map": {"": c.device}, "local_files_only": c.local_files_only,
                  "cache_dir": c.cache_dir, "trust_remote_code": False,
                  "attn_implementation": c.attention_implementation}
        if c.quantization == "nf4":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=getattr(torch, c.compute_dtype),
                bnb_4bit_use_double_quant=c.double_quantization)
        elif c.quantization == "int8":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        loading_start = time.perf_counter()
        try:
            self.model = AutoModelForCausalLM.from_pretrained(c.model_id, **kwargs)
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError("GPU OOM loading model. Use separate compress/read stages, explicit device placement, or explicit NF4; no automatic CPU/precision fallback") from exc
        self.model.eval()
        self.model.requires_grad_(False)
        synchronize(c.device)
        self.initialization = {"model_initialization_seconds": time.perf_counter() - start,
                               "weight_loading_seconds": time.perf_counter() - loading_start,
                               "resolved_revision": getattr(self.model.config, "_commit_hash", None),
                               "resolved_tokenizer_revision": getattr(self.tokenizer, "init_kwargs", {}).get("_commit_hash"),
                               "quantization_runtime": self.quantization_runtime,
                               "device": c.device}

    def count_text(self, text):
        return len(self._tokenizer().encode(text, add_special_tokens=False, truncation=False))

    def begin_sample(self):
        if self.model is not None:
            from transformers import set_seed
            set_seed(self.config.seed)

    def generate(self, messages):
        self.prepare()
        import torch
        c = self.config
        # Templates supply their own special tokens. Never tokenize with truncation.
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **c.chat_template_kwargs)
        encoded = self.tokenizer(prompt, add_special_tokens=False, truncation=False, return_tensors="pt")
        input_tokens = encoded["input_ids"].shape[-1]
        limits = [v for v in (getattr(self.model.config, "max_position_embeddings", None),
                              self.tokenizer.model_max_length, c.max_context_tokens)
                  if isinstance(v, int) and 0 < v < 10**9]
        if not limits:
            raise ValueError("Unknown model context limit; set max_context_tokens explicitly")
        if input_tokens + c.max_new_tokens > min(limits):
            raise ValueError(f"Context limit exceeded: {input_tokens} input + {c.max_new_tokens} output > {min(limits)}. Prompt was not truncated")
        encoded = encoded.to(c.device)
        kwargs = {"max_new_tokens": c.max_new_tokens, "do_sample": c.do_sample,
                  "pad_token_id": self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id}
        if c.do_sample:
            kwargs.update(temperature=c.temperature, top_p=c.top_p)
        # Preserve model-specific EOS lists (notably Llama end-of-turn tokens).
        synchronize(c.device)
        start = time.perf_counter()
        try:
            with torch.inference_mode():
                output = self.model.generate(**encoded, **kwargs)
            synchronize(c.device)
        except Exception as exc:
            synchronize(c.device)
            message = "Generation failed; no automatic precision/device fallback: " + str(exc)
            raise GenerationFailure(message, input_tokens, time.perf_counter() - start) from exc
        generated = output[0, input_tokens:]
        from ..schemas import GenerationResult
        return GenerationResult(self.tokenizer.decode(generated, skip_special_tokens=True), input_tokens,
                                len(generated), time.perf_counter() - start)

    def close(self):
        self.model = None
        self.tokenizer = None
        release_memory()

"""Chat-completions HTTP interface. Provider usage is authoritative."""
import json
import os
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from .base import LLMBackend, GenerationFailure
from ..schemas import GenerationResult


class OpenAICompatibleBackend(LLMBackend):
    def __init__(self, config):
        self.config = config
        self.tokenizer = None
        self.initialization = {"weight_loading_seconds": 0.0, "backend": "remote API",
                               "context_limit_enforcement": "server; no client truncation"}

    def count_text(self, text):
        if not self.config.tokenizer_id:
            return None
        if self.tokenizer is None:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.tokenizer_id, revision=self.config.tokenizer_revision,
                cache_dir=self.config.cache_dir, local_files_only=self.config.local_files_only,
                trust_remote_code=False)
        return len(self.tokenizer.encode(text, add_special_tokens=False, truncation=False))

    def generate(self, messages):
        c = self.config
        payload = {"model": c.model_id, "messages": messages, "max_tokens": c.max_new_tokens,
                   "temperature": c.temperature if c.do_sample else 0}
        if c.do_sample:
            payload["top_p"] = c.top_p
        if c.api_send_seed:
            payload["seed"] = c.seed
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(c.api_key_env)
        if key:
            headers["Authorization"] = f"Bearer {key}"
        request = Request(c.api_base.rstrip("/") + "/chat/completions",
                          data=json.dumps(payload).encode(), headers=headers, method="POST")
        start = time.perf_counter()
        try:
            with urlopen(request, timeout=c.timeout) as response:
                result = json.load(response)
            text = result["choices"][0]["message"]["content"]
            if not isinstance(text, str):
                raise ValueError("Provider response content is not text")
            usage = result.get("usage") or {}
            input_tokens, output_tokens = usage.get("prompt_tokens"), usage.get("completion_tokens")
            for value in (input_tokens, output_tokens):
                if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                    raise ValueError("Invalid provider token usage")
            return GenerationResult(text, input_tokens, output_tokens, time.perf_counter() - start)
        except Exception as exc:
            # Do not echo headers, credentials, or arbitrary response bodies into logs.
            detail = f"HTTP {exc.code}" if isinstance(exc, HTTPError) else type(exc).__name__
            raise GenerationFailure(f"API request failed ({detail}); check endpoint, model access, and server context limit",
                                    latency_seconds=time.perf_counter() - start) from exc

    def close(self):
        self.tokenizer = None

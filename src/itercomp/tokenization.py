"""Optional fixed reference tokenizer; never loads model weights."""
import time


class ReferenceTokenizer:
    def __init__(self, config):
        self.config = config
        self.tokenizer = None
        self.initialization_seconds = None

    def prepare(self):
        if self.tokenizer is None:
            start = time.perf_counter()
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_id, revision=self.config.revision,
                local_files_only=self.config.local_files_only, trust_remote_code=False)
            self.initialization_seconds = time.perf_counter() - start

    def count_text(self, text):
        self.prepare()
        return len(self.tokenizer.encode(text, add_special_tokens=False, truncation=False))

    def close(self):
        self.tokenizer = None

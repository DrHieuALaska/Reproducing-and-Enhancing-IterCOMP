from pathlib import Path
from string import Template
import hashlib


def load_prompts(config):
    base = Path(__file__).parent / "prompts"
    return {name: Path(getattr(config.prompts, name) or base / f"{name}.txt").read_text(encoding="utf-8")
            for name in ("controller", "reader")}


def prompt_hashes(prompts):
    return {name: hashlib.sha256(value.encode()).hexdigest() for name, value in prompts.items()}


def messages(template, question, context):
    # Single user message also works with Mistral templates lacking system roles.
    return [{"role": "user", "content": Template(template).substitute(question=question, context=context)}]


def render_segments(segments, config):
    return config.separator.join(
        f"{s.title}: {s.text}" if config.include_titles and s.title else s.text for s in segments)


def render_paragraphs(paragraphs, config):
    return config.separator.join(
        f"{p.title}: {p.paragraph_text}" if config.include_titles and p.title else p.paragraph_text for p in paragraphs)

"""Data passed between components; inference inputs deliberately exclude gold."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Paragraph:
    idx: int
    title: str
    paragraph_text: str


@dataclass(frozen=True)
class InputSample:
    id: str
    question: str
    paragraphs: tuple[Paragraph, ...]


@dataclass(frozen=True)
class EvidenceSegment:
    id: str
    paragraph_idx: int
    sentence_idx: int
    text: str
    title: str
    start_char: int = 0
    end_char: int = 0


@dataclass(frozen=True)
class Representation:
    dense: tuple[float, ...]
    lexical: dict[str, float]


@dataclass(frozen=True)
class EvidenceScore:
    id: str
    semantic: float
    lexical: float
    dual: float


@dataclass
class GenerationResult:
    text: str
    input_tokens: int | None
    output_tokens: int | None
    latency_seconds: float


@dataclass(frozen=True)
class Judgment:
    answerable: bool
    follow_up_question: str | None
    format_warnings: tuple[str, ...] = ()


@dataclass
class CallLedger:
    """Unknown usage propagates; zero means a stage made no calls."""
    calls: list[dict] = field(default_factory=list)

    def add(self, result: GenerationResult, **extra):
        self.calls.append({"raw_response": result.text,
                           "input_tokens": result.input_tokens,
                           "output_tokens": result.output_tokens,
                           "latency_seconds": result.latency_seconds, **extra})

    def total(self, key):
        values = [c.get(key) for c in self.calls]
        return sum(values) if all(v is not None for v in values) else None

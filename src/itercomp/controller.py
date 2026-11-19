import json
import re
from .schemas import Judgment, CallLedger
from .prompt_builder import messages
from .backends.base import GenerationFailure


def parse_judgment(text, *, answerable_follow_up_policy="strict"):
    """Validate a judgment, optionally logging normalization of an unused field.

    Standalone parsing remains strict by default. The pipeline passes its explicit
    configured policy. Only a valid answerable label with a string follow-up can
    be normalized; no label is inferred and no missing/malformed field is repaired.
    """
    if answerable_follow_up_policy not in ("strict", "normalize_unused"):
        raise ValueError("Unknown answerable follow-up policy")
    body = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", body, flags=re.DOTALL | re.IGNORECASE)
    if match:
        body = match.group(1).strip()
    def unique_keys(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError("Duplicate controller JSON field")
            obj[key] = value
        return obj
    try:
        value = json.loads(body, object_pairs_hook=unique_keys)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("Controller did not return valid JSON") from exc
    if not isinstance(value, dict) or set(value) != {"answer", "follow_up_question"}:
        raise ValueError("Controller JSON must contain exactly answer and follow_up_question")
    label, follow = value["answer"], value["follow_up_question"]
    if label not in ("answerable", "unanswerable"):
        raise ValueError("Invalid controller answerability label")
    if follow is not None and not isinstance(follow, str):
        raise ValueError("Controller follow_up_question must be a string or null")
    warnings = ()
    if label == "answerable":
        if follow is not None:
            if answerable_follow_up_policy == "strict":
                raise ValueError("An answerable decision requires a null follow-up")
            # The answerable branch terminates before any follow-up is used.
            # Retain the original response in the call ledger and flag the mismatch.
            warnings = ("unused_follow_up_ignored_for_answerable",)
            follow = None
    elif not isinstance(follow, str) or not follow.strip():
        raise ValueError("An unanswerable decision requires a nonempty follow-up")
    return Judgment(label == "answerable", follow, warnings)


class Controller:
    def __init__(self, backend, template, retries=0, answerable_follow_up_policy="strict"):
        self.backend = backend
        self.template = template
        self.retries = retries
        self.answerable_follow_up_policy = answerable_follow_up_policy
        self.ledger = CallLedger()

    def reset(self):
        self.ledger = CallLedger()

    def judge(self, original_question, context):
        prompt = messages(self.template, original_question, context)
        for attempt in range(self.retries + 1):
            try:
                result = self.backend.generate(prompt)
            except GenerationFailure as exc:
                self.ledger.add(exc.result, error=str(exc), attempt=attempt + 1)
                raise
            self.ledger.add(result, attempt=attempt + 1)
            try:
                judgment = parse_judgment(result.text, answerable_follow_up_policy=self.answerable_follow_up_policy)
                self.ledger.calls[-1]["parsed_judgment"] = {
                    "answer": "answerable" if judgment.answerable else "unanswerable",
                    "follow_up_question": judgment.follow_up_question}
                if judgment.format_warnings:
                    self.ledger.calls[-1]["format_warnings"] = list(judgment.format_warnings)
                return judgment
            except ValueError as exc:
                self.ledger.calls[-1]["parse_error"] = str(exc)
                if attempt == self.retries:
                    raise

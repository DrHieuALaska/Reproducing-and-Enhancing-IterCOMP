from .schemas import CallLedger
from .prompt_builder import messages
from .backends.base import GenerationFailure

ABSTENTION = "INSUFFICIENT_INFORMATION"


class Reader:
    def __init__(self, backend, template):
        self.backend = backend
        self.template = template
        self.ledger = CallLedger()

    def answer(self, original_question, context):
        self.ledger = CallLedger()
        try:
            result = self.backend.generate(messages(self.template, original_question, context))
        except GenerationFailure as exc:
            self.ledger.add(exc.result, error=str(exc))
            raise
        self.ledger.add(result)
        answer = result.text.strip()
        # Explicit reader abstention convention; independent of controller judgment.
        answerable = bool(answer) and answer != ABSTENTION
        return (answer if answerable else ""), answerable, result.text

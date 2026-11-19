"""Independent paragraph segmentation; no context expansion or text rewriting."""
from .schemas import EvidenceSegment


class SentenceDecomposer:
    def __init__(self, config):
        self.config = config
        self.nlp = None

    def prepare(self):
        if self.nlp is None:
            import spacy
            # Tokenizer exceptions include abbreviations. No pretrained spaCy model.
            self.nlp = spacy.blank(self.config.language)
            self.nlp.add_pipe("sentencizer")

    def decompose(self, paragraphs):
        self.prepare()
        result = []
        for p in paragraphs:
            if not p.paragraph_text.strip():
                continue
            doc = self.nlp(p.paragraph_text)
            index = 0
            for span in doc.sents:
                if not span.text.strip():
                    continue
                # span.text is an exact source substring, including internal whitespace.
                result.append(EvidenceSegment(f"p{p.idx}_s{index}", p.idx, index,
                                              span.text, p.title, span.start_char, span.end_char))
                index += 1
        return result

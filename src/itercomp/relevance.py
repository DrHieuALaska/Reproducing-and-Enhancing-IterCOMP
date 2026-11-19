"""Equations 1, 3 and 4, without score normalization or rank fusion."""
import math
from .schemas import EvidenceScore


def semantic_score(query, evidence):
    if len(query) != len(evidence) or not query:
        raise ValueError("Dense dimensions must match and be nonempty")
    return math.fsum(float(a) * float(b) for a, b in zip(query, evidence))


def lexical_score(query, evidence):
    return math.fsum(float(weight) * float(evidence[token])
                     for token, weight in query.items() if token in evidence)


def score_evidence(query, representations, segments, semantic_weight=0.6):
    if len(representations) != len(segments):
        raise ValueError("Evidence representation count mismatch")
    if not math.isfinite(semantic_weight) or not 0 <= semantic_weight <= 1:
        raise ValueError("semantic_weight must be in [0, 1]")
    scores = []
    for segment, representation in zip(segments, representations):
        sem = semantic_score(query.dense, representation.dense)
        lex = lexical_score(query.lexical, representation.lexical)
        dual = semantic_weight * sem + (1 - semantic_weight) * lex
        if not all(math.isfinite(v) for v in (sem, lex, dual)):
            raise ValueError(f"Nonfinite relevance score for {segment.id}")
        scores.append(EvidenceScore(segment.id, sem, lex, dual))
    # Stable source-order tie-breaking, independent of arbitrary ID lexical order.
    return sorted(scores, key=lambda s: -s.dual)

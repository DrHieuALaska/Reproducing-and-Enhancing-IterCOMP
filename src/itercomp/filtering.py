"""Equation 5: global linear percentile, inclusive threshold, all ties retained."""
import math


def percentile_filter(ranked_scores, percentile=90.0, method="linear"):
    import numpy as np
    if not ranked_scores:
        raise ValueError("Cannot filter an empty sentence pool")
    if not math.isfinite(percentile) or not 0 <= percentile <= 100:
        raise ValueError("percentile must be in [0, 100]")
    values = [s.dual for s in ranked_scores]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Nonfinite filtering scores")
    threshold = float(np.percentile(values, percentile, method=method))
    return threshold, [s.id for s in ranked_scores if s.dual >= threshold]

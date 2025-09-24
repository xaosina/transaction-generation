from .metrics import (
    Levenshtein,
    Accuracy,
    MultisetF1Metric,
    CardinalityCoverage,
    Gini,
    ShannonEntropy,
)
from functools import partial


CardinalityCoverageUserwise = partial(CardinalityCoverage, userwise=True)
CardinalityCoverageSimple = partial(CardinalityCoverage, userwise=False)
F1Macro = partial(MultisetF1Metric, average="macro")
F1Micro = partial(MultisetF1Metric, average="micro")


__all__ = [
    "Levenshtein",
    "Accuracy",
    "F1Macro",
    "F1Micro",
    "CardinalityCoverageUserwise",
    "CardinalityCoverageSimple",
    "Gini",
    "ShannonEntropy",
    "BatchCutMetric"
]

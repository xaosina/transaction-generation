from .metrics import (
    Levenshtein,
    Accuracy,
    F1Metric,
    CardinalityCoverage,
    Gini,
    ShannonEntropy,
)
from functools import partial


CardinalityCoverageUserwise = partial(CardinalityCoverage, userwise=True)
CardinalityCoverageSimple = partial(CardinalityCoverage, userwise=False)
F1Macro = partial(F1Metric, average="macro")
F1Micro = partial(F1Metric, average="micro")


__all__ = [
    "Levenshtein",
    "Accuracy",
    "F1Macro",
    "F1Micro",
    "CardinalityCoverageUserwise",
    "CardinalityCoverageSimple",
    "Gini",
    "ShannonEntropy",
]

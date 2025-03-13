from .metrics import LevensteinMetric, Accuracy, F1Metric, Coverage, Gini, ShannonEntropy
from functools import partial

CoverageUserwise = partial(Coverage, userwise=True)
CoverageSimple = partial(Coverage, userwise=False)
F1Macro = partial(F1Metric, average='macro')
F1Micro = partial(F1Metric, average='micro')

METRICS = [
    LevensteinMetric, 
    Accuracy, 
    F1Macro,
    F1Micro, 
    CoverageUserwise,
    CoverageSimple, 
    Gini, 
    ShannonEntropy,
]
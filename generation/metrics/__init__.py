import importlib
from .metrics import LevensteinMetric, Accuracy, F1Metric, Coverage, Gini, ShannonEntropy
from functools import partial


CoverageUserwise = partial(Coverage, userwise=True)
CoverageSimple = partial(Coverage, userwise=False)
F1Macro = partial(F1Metric, average='macro')
F1Micro = partial(F1Metric, average='micro')


__all__ = [
    "LevensteinMetric", 
    "Accuracy", 
    "F1Macro",
    "F1Micro", 
    "CoverageUserwise",
    "CoverageSimple", 
    "Gini", 
    "ShannonEntropy",
]

_LOADED = {}

def __getattr__(name: str):
    if name in __all__:
        # Если уже загружен, вернём из кэша
        if name in _LOADED:
            return _LOADED[name]
        
        # Иначе загружаем из модуля metrics.metrics
        module = importlib.import_module("metrics.metrics")  
        attr = getattr(module, name)  # например, Accuracy
        _LOADED[name] = attr
        return attr

    raise AttributeError(f"module {__name__} has no attribute {name}")
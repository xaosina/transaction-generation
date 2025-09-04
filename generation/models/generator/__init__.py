import os
from .base import BaseGenerator, ModelConfig
from ..import autoencoders
from ..autoencoders import BaseAE
from ...utils import _auto_import_subclasses

_auto_import_subclasses(os.path.dirname(__file__), __name__, globals(), BaseGenerator)

for name in dir(autoencoders):
    obj = getattr(autoencoders, name)
    if isinstance(obj, type) and issubclass(obj, BaseAE) and obj != BaseAE:
        globals()[name] = obj
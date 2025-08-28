import os

from ...utils import _auto_import_subclasses
from .base import BaseAE

_auto_import_subclasses(os.path.dirname(__file__), __name__, globals(), BaseAE)

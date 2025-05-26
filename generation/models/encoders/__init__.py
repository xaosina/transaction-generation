import os

# from .transformer.ar import AR
from ebes.model import BaseSeq2Seq
from .base import EncoderConfig
from ...utils import _auto_import_subclasses

_auto_import_subclasses(os.path.dirname(__file__), __name__, globals(), BaseSeq2Seq)

import logging
import sys
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def log_to_file(filename: Path, file_lvl="info", cons_lvl="warning"):
    if isinstance(file_lvl, str):
        file_lvl = getattr(logging, file_lvl.upper())
    if isinstance(cons_lvl, str):
        cons_lvl = getattr(logging, cons_lvl.upper())

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(cons_lvl)
    cfmt = logging.Formatter("{levelname:8} - {asctime} - {message}", style="{")
    ch.setFormatter(cfmt)

    fh = logging.FileHandler(filename)
    fh.setLevel(file_lvl)
    ffmt = logging.Formatter(
        "{name: ^16} - {asctime} - {message}",
        style="{",
    )
    fh.setFormatter(ffmt)
    logger = logging.getLogger()
    logger.setLevel(min(file_lvl, cons_lvl))
    logger.addHandler(fh)
    logger.addHandler(ch)

    try:
        yield
    finally:
        fh.close()
        logger.removeHandler(fh)
        logger.removeHandler(ch)

import logging
import sys
from typing import Optional

class Logger:

    def __init__(
        self,
        name: str,
        level: int = logging.INFO,
        log_to_file: Optional[str] = None,
        log_to_stdout: bool = True
    ):
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)

        if self._logger.hasHandlers():
            self._logger.handlers.clear()

        self.formatter = logging.Formatter(
            fmt='[%(asctime)s]:%(message)s',
            datefmt='%H:%M:%S'
        )

        if log_to_stdout:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(self.formatter)
            self._logger.addHandler(console_handler)

        if log_to_file is not None:
            self.set_file(log_to_file)

    def debug(self, msg: str) -> None:
        self._logger.debug(msg)

    def info(self, *args, **kwargs) -> None:
        msg = self._format_message(*args)
        self._logger.info(msg)

    def warning(self, msg: str) -> None:
        self._logger.warning(msg)

    def error(self, msg: str) -> None:
        self._logger.error(msg)

    def critical(self, msg: str) -> None:
        self._logger.critical(msg)

    def set_file(self, log_to_file) -> None:
        file_handler = logging.FileHandler(log_to_file, encoding='utf-8')
        file_handler.setLevel(self._logger.level)
        file_handler.setFormatter(self.formatter)
        self._logger.addHandler(file_handler)

    def _format_message(self, *args) -> str:
        return '\n' + ' '.join(str(arg) for arg in args)
"""Structured logging configuration for the bridge.

Uses stdlib logging with a compact, key=value formatter so logs can be
ingested by ``journalctl``, ``logfmt``-aware tools, or whatever the user
already has, without dragging in a heavy dependency.
"""

from __future__ import annotations

import logging
import sys
from typing import Any


class KeyValueFormatter(logging.Formatter):
    default_fields = ("ts", "level", "logger", "msg")

    def format(self, record: logging.LogRecord) -> str:
        base: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k
            not in {
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "asctime",
                "message",
                "taskName",
            }
        }
        merged = {**base, **extras}
        parts = [f"{key}={_format_value(value)}" for key, value in merged.items()]
        line = " ".join(parts)
        if record.exc_info:
            line = f"{line}\n{self.formatException(record.exc_info)}"
        return line


def _format_value(value: Any) -> str:
    text = str(value)
    if any(c.isspace() for c in text) or "=" in text:
        return '"' + text.replace('"', '\\"') + '"'
    return text


def configure_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(KeyValueFormatter())

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)

    for noisy in ("httpx", "httpcore", "uvicorn.error"):
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))

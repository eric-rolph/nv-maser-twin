"""Tests for structured JSON logging in nv_maser.api.server."""
import json
import logging
import os

import pytest

from nv_maser.api.server import _JsonFormatter, _configure_logging


def _make_record(msg: str = "hello", level: int = logging.INFO) -> logging.LogRecord:
    record = logging.LogRecord(
        name="test_logger",
        level=level,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=(),
        exc_info=None,
    )
    return record


def test_json_formatter_produces_valid_json():
    formatter = _JsonFormatter()
    record = _make_record("hello")
    output = formatter.format(record)
    parsed = json.loads(output)
    for key in ("ts", "level", "logger", "msg"):
        assert key in parsed, f"Missing key: {key}"


def test_json_formatter_level_and_msg():
    formatter = _JsonFormatter()
    record = _make_record("hello", level=logging.INFO)
    parsed = json.loads(formatter.format(record))
    assert parsed["level"] == "INFO"
    assert parsed["msg"] == "hello"


def test_configure_logging_text_format(monkeypatch):
    monkeypatch.setenv("LOG_FORMAT", "text")
    # Remove existing handlers so _configure_logging can add one
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    root.handlers.clear()
    try:
        _configure_logging()
        assert len(root.handlers) >= 1
    finally:
        root.handlers[:] = original_handlers


def test_configure_logging_json_format(monkeypatch):
    monkeypatch.setenv("LOG_FORMAT", "json")
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    root.handlers.clear()
    try:
        _configure_logging()
        assert len(root.handlers) >= 1
        assert any(isinstance(h.formatter, _JsonFormatter) for h in root.handlers)
    finally:
        root.handlers[:] = original_handlers

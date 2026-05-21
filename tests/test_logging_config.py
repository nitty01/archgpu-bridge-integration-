import logging
from io import StringIO

from archgpu_ollama_bridge.logging_config import KeyValueFormatter, configure_logging


def test_key_value_formatter_includes_extras() -> None:
    formatter = KeyValueFormatter()
    record = logging.LogRecord(
        name="archgpu_ollama_bridge.test",
        level=logging.INFO,
        pathname="x",
        lineno=1,
        msg="hello world",
        args=None,
        exc_info=None,
    )
    record.method = "POST"
    record.path = "/v1/chat/completions"
    record.duration_ms = 12.34

    formatted = formatter.format(record)

    assert "level=INFO" in formatted
    assert 'msg="hello world"' in formatted
    assert "method=POST" in formatted
    assert "path=/v1/chat/completions" in formatted
    assert "duration_ms=12.34" in formatted


def test_configure_logging_replaces_handlers() -> None:
    buffer = StringIO()
    configure_logging(level=logging.DEBUG)

    root = logging.getLogger()
    assert len(root.handlers) == 1

    root.handlers[0].stream = buffer
    logging.getLogger("archgpu_ollama_bridge.test").info("ping", extra={"path": "/health"})

    output = buffer.getvalue()
    assert "msg=ping" in output
    assert "path=/health" in output

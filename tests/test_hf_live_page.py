import pytest

from archgpu_ollama_bridge.hf_index import resolve_hf_search_query


def test_resolve_hf_search_query_defaults_to_gguf() -> None:
    assert resolve_hf_search_query("") == "GGUF"
    assert resolve_hf_search_query("capability:coding") == "GGUF"
    assert resolve_hf_search_query("repo:ggml-org/gemma-4") == "ggml-org/gemma-4"
    assert resolve_hf_search_query("gemma-4") == "gemma-4"

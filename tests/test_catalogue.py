from pathlib import Path

import pytest

from archgpu_ollama_bridge.catalogue import (
    Catalogue,
    CatalogueError,
    HFRef,
)


def _write_catalogue(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "catalogue.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def test_load_returns_empty_when_path_missing(tmp_path: Path) -> None:
    cat = Catalogue.load(tmp_path / "missing.yaml")
    assert cat.list_entries() == []


def test_resolve_alias_returns_hfref(tmp_path: Path) -> None:
    cat = Catalogue.load(
        _write_catalogue(
            tmp_path,
            """
models:
  - alias: qwen-coder
    repo: Qwen/Qwen2.5-Coder-3B-Instruct-GGUF
    filename: qwen2.5-coder-3b-instruct-q4_k_m.gguf
    context_length: 8192
    tags: [code]
""".strip(),
        )
    )

    ref = cat.resolve("qwen-coder")
    assert isinstance(ref, HFRef)
    assert ref.repo == "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF"
    assert ref.filename == "qwen2.5-coder-3b-instruct-q4_k_m.gguf"
    assert ref.revision == "main"
    assert ref.suggested_context_length == 8192
    assert ref.tags == ("code",)


def test_resolve_full_hf_ref_does_not_require_alias(tmp_path: Path) -> None:
    cat = Catalogue.load(tmp_path / "missing.yaml")
    ref = cat.resolve("Qwen/Qwen2.5-3B-Instruct-GGUF:qwen2.5-3b-instruct-q4_k_m.gguf")
    assert ref.repo == "Qwen/Qwen2.5-3B-Instruct-GGUF"
    assert ref.filename == "qwen2.5-3b-instruct-q4_k_m.gguf"


def test_resolve_rejects_path_traversal_in_filename(tmp_path: Path) -> None:
    cat = Catalogue.load(tmp_path / "missing.yaml")
    with pytest.raises(CatalogueError):
        cat.resolve("Qwen/Qwen2.5:../../../etc/passwd.gguf")


def test_resolve_rejects_non_gguf_filename(tmp_path: Path) -> None:
    cat = Catalogue.load(tmp_path / "missing.yaml")
    with pytest.raises(CatalogueError):
        cat.resolve("Qwen/Qwen2.5:model.bin")


def test_resolve_rejects_invalid_repo_format(tmp_path: Path) -> None:
    cat = Catalogue.load(tmp_path / "missing.yaml")
    with pytest.raises(CatalogueError):
        cat.resolve("not-a-repo:model.gguf")


def test_resolve_rejects_unknown_alias(tmp_path: Path) -> None:
    cat = Catalogue.load(tmp_path / "missing.yaml")
    with pytest.raises(CatalogueError):
        cat.resolve("totally-unknown-alias")


def test_allowlist_blocks_unlisted_orgs(tmp_path: Path) -> None:
    cat = Catalogue.load(tmp_path / "missing.yaml", allow_orgs=("qwen",))
    with pytest.raises(CatalogueError, match="not on the configured allowlist"):
        cat.resolve("bartowski/Some-GGUF:file.gguf")


def test_allowlist_allows_listed_org_case_insensitive(tmp_path: Path) -> None:
    cat = Catalogue.load(tmp_path / "missing.yaml", allow_orgs=("Qwen",))
    ref = cat.resolve("Qwen/Foo-GGUF:file.gguf")
    assert ref.repo.startswith("Qwen/")


def test_safe_local_filename_collapses_org_repo() -> None:
    ref = HFRef(repo="Qwen/Repo", filename="file.gguf")
    assert ref.safe_local_filename == "Qwen__Repo__file.gguf"


def test_load_rejects_duplicate_alias(tmp_path: Path) -> None:
    body = """
models:
  - alias: dup
    repo: A/B
    filename: a.gguf
  - alias: dup
    repo: C/D
    filename: c.gguf
""".strip()
    with pytest.raises(CatalogueError):
        Catalogue.load(_write_catalogue(tmp_path, body))


def test_resolve_rejects_empty_name(tmp_path: Path) -> None:
    cat = Catalogue.load(tmp_path / "missing.yaml")
    with pytest.raises(CatalogueError):
        cat.resolve("")
    with pytest.raises(CatalogueError):
        cat.resolve("   ")

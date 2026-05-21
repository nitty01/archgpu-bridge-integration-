"""Catalogue of pullable Hugging Face GGUF models.

A catalogue entry binds a short, user-friendly alias (e.g. ``qwen2.5-coder-3b``)
to a concrete Hugging Face reference (repo + filename + optional revision and
suggested context length). The pull endpoint accepts either an alias or a raw
HF reference of the form ``<org>/<repo>:<filename.gguf>``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


_HF_ORG_REPO_RE = re.compile(r"^[A-Za-z0-9._\-]+/[A-Za-z0-9._\-]+$")
_FILENAME_RE = re.compile(r"^[A-Za-z0-9._\-]+\.gguf$", re.IGNORECASE)
_ALIAS_RE = re.compile(r"^[A-Za-z0-9._\-]+(:[A-Za-z0-9._\-]+)?$")


class CatalogueError(ValueError):
    """Raised when a pull name cannot be resolved or is unsafe."""


@dataclass(slots=True, frozen=True)
class HFRef:
    """A fully-resolved Hugging Face reference."""

    repo: str
    filename: str
    revision: str = "main"
    suggested_context_length: int | None = None
    tags: tuple[str, ...] = ()
    display_name: str | None = None

    @property
    def safe_local_filename(self) -> str:
        """A flat filename safe to drop into ``models_host_dir``.

        We collapse the HF ``<org>/<repo>`` into the filename to avoid
        collisions when two repos publish the same GGUF basename.
        """

        org, repo = self.repo.split("/", 1)
        base = self.filename
        return f"{org}__{repo}__{base}"


class CatalogueEntry(BaseModel):
    alias: str = Field(min_length=1)
    repo: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    revision: str = Field(default="main")
    context_length: int | None = Field(default=None, gt=0)
    tags: list[str] = Field(default_factory=list)
    display_name: str | None = None


class CatalogueDocument(BaseModel):
    models: list[CatalogueEntry] = Field(default_factory=list)


class Catalogue:
    """In-memory view of curated aliases + parser for free-form HF refs."""

    def __init__(
        self,
        entries: dict[str, CatalogueEntry],
        *,
        allow_orgs: tuple[str, ...] = (),
    ) -> None:
        self._entries = entries
        self._allow_orgs = tuple(o.lower() for o in allow_orgs)

    @classmethod
    def load(
        cls,
        path: str | Path | None,
        *,
        allow_orgs: tuple[str, ...] = (),
    ) -> "Catalogue":
        entries: dict[str, CatalogueEntry] = {}
        if path is not None:
            p = Path(path)
            if p.exists():
                payload = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                document = CatalogueDocument.model_validate(payload)
                for entry in document.models:
                    if entry.alias in entries:
                        raise CatalogueError(f"Duplicate catalogue alias: {entry.alias}")
                    _validate_repo(entry.repo)
                    _validate_filename(entry.filename)
                    entries[entry.alias] = entry
        return cls(entries=entries, allow_orgs=allow_orgs)

    def list_entries(self) -> list[CatalogueEntry]:
        return list(self._entries.values())

    def resolve(self, name: str) -> HFRef:
        """Resolve a pull name into a concrete HF reference.

        Accepts an alias (with optional ``:tag`` suffix that we currently
        ignore) or a raw HF reference ``<org>/<repo>:<filename.gguf>``.
        """

        if not name or not isinstance(name, str):
            raise CatalogueError("Empty pull name")

        cleaned = name.strip()
        if not cleaned:
            raise CatalogueError("Empty pull name")

        if "/" in cleaned:
            ref = self._parse_hf_ref(cleaned)
        else:
            ref = self._resolve_alias(cleaned)

        self._enforce_allowlist(ref.repo)
        return ref

    def _resolve_alias(self, raw: str) -> HFRef:
        head = raw.split(":", 1)[0]
        if not _ALIAS_RE.fullmatch(raw):
            raise CatalogueError(f"Invalid alias: {raw!r}")
        entry = self._entries.get(head)
        if entry is None:
            raise CatalogueError(
                f"Unknown alias {head!r}. Use a configured alias from "
                f"config/catalogue.yaml or a full HF ref like "
                f"'org/repo:file.gguf'."
            )
        return HFRef(
            repo=entry.repo,
            filename=entry.filename,
            revision=entry.revision,
            suggested_context_length=entry.context_length,
            tags=tuple(entry.tags),
            display_name=entry.display_name or entry.alias,
        )

    def _parse_hf_ref(self, raw: str) -> HFRef:
        if ":" not in raw:
            raise CatalogueError(
                "HF reference must be 'org/repo:filename.gguf'"
            )
        repo_part, filename = raw.rsplit(":", 1)
        revision = "main"
        if "@" in repo_part:
            repo_part, revision = repo_part.split("@", 1)
        _validate_repo(repo_part)
        _validate_filename(filename)
        if not revision or "/" in revision or ".." in revision:
            raise CatalogueError(f"Invalid revision: {revision!r}")
        return HFRef(
            repo=repo_part,
            filename=filename,
            revision=revision,
            display_name=raw,
        )

    def _enforce_allowlist(self, repo: str) -> None:
        if not self._allow_orgs:
            return
        org = repo.split("/", 1)[0].lower()
        if org not in self._allow_orgs:
            raise CatalogueError(
                f"Hugging Face org {org!r} is not on the configured allowlist"
            )


def _validate_repo(repo: str) -> None:
    if not _HF_ORG_REPO_RE.fullmatch(repo):
        raise CatalogueError(f"Invalid HF repo: {repo!r}")
    if ".." in repo:
        raise CatalogueError(f"Invalid HF repo: {repo!r}")


def _validate_filename(filename: str) -> None:
    if "/" in filename or "\\" in filename or ".." in filename or "\0" in filename:
        raise CatalogueError(f"Unsafe filename: {filename!r}")
    if not _FILENAME_RE.fullmatch(filename):
        raise CatalogueError(f"Filename must match *.gguf: {filename!r}")

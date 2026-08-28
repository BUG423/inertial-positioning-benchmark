"""Repository-wide documentation consistency checks."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")
EXTERNAL_SCHEMES = ("http://", "https://", "mailto:", "tel:")


def _local_target(raw_target: str) -> str | None:
    """Return a local link target, excluding URLs and page fragments."""

    target = raw_target.strip()
    if target.startswith("<") and ">" in target:
        target = target[1 : target.index(">")]
    else:
        # Markdown permits an optional title after the destination.
        target = target.split(maxsplit=1)[0]

    if not target or target.startswith("#") or target.startswith(EXTERNAL_SCHEMES):
        return None
    return unquote(target.split("#", 1)[0].split("?", 1)[0])


def test_relative_markdown_links_resolve() -> None:
    missing: list[str] = []

    for document in sorted(ROOT.rglob("*.md")):
        if ".git" in document.parts:
            continue
        for match in MARKDOWN_LINK.finditer(document.read_text(encoding="utf-8")):
            target = _local_target(match.group(1))
            if target is None:
                continue
            resolved = (document.parent / target).resolve()
            if not resolved.exists():
                missing.append(f"{document.relative_to(ROOT)} -> {target}")

    assert not missing, "Broken relative Markdown links:\n" + "\n".join(missing)

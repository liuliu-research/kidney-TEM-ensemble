"""Fail-closed privacy and release audit for this repository."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
POLICY = json.loads((ROOT / "privacy_policy.json").read_text(encoding="utf-8"))
EXCLUDED_DIRS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache"}
EXCLUDED_FILES = {"PRIVACY_AUDIT.py", "privacy_policy.json"}
TEXT_SUFFIXES = {".cff", ".cfg", ".csv", ".ini", ".json", ".md", ".py", ".r", ".toml", ".txt", ".yaml", ".yml"}
TEXT_NAMES = {".gitattributes", ".gitignore", "LICENSE", "requirements.txt"}
SECRET_PATTERNS = {
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "GitHub token": re.compile(r"(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}"),
    "OpenAI-style key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "IPv4 address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "Windows absolute path": re.compile(r"(?i)\b[A-Z]:\\(?:Users|Python|Data)\\"),
    "Linux home path": re.compile(r"/(?:home|Users)/[^/\s]+/"),
}


def main() -> int:
    problems: list[str] = []
    forbidden_suffixes = {item.casefold() for item in POLICY["forbidden_suffixes"]}
    banned = [item.casefold() for item in POLICY.get("banned_content", [])]
    max_bytes = int(POLICY.get("max_file_bytes", 20 * 1024 * 1024))
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED_DIRS for part in relative.parts):
            continue
        if path.name in EXCLUDED_FILES:
            continue
        if path.suffix.casefold() in forbidden_suffixes:
            problems.append(f"forbidden file type: {relative.as_posix()}")
        if path.stat().st_size > max_bytes:
            problems.append(f"oversized file: {relative.as_posix()}")
        if path.name not in TEXT_NAMES and path.suffix.casefold() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            problems.append(f"non-UTF-8 text: {relative.as_posix()}")
            continue
        folded = text.casefold()
        for token in banned:
            if token and token in folded:
                problems.append(f"banned content in {relative.as_posix()}: {token}")
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                problems.append(f"{label} in {relative.as_posix()}")
    if problems:
        print("PRIVACY AUDIT FAILED")
        for problem in problems:
            print(f"- {problem}")
        return 1
    print("PRIVACY AUDIT PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

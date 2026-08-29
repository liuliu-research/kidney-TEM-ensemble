"""Fail-closed, project-specific privacy audit for the public release tree."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
POLICY = json.loads((ROOT / "privacy_policy.json").read_text(encoding="utf-8"))
EXCLUDED_DIRS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache"}
EXCLUDED_ROOT_FILES = {"PRIVACY_AUDIT.py", "privacy_policy.json"}
TEXT_SUFFIXES = {
    ".cff",
    ".cfg",
    ".csv",
    ".ini",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".r",
    ".sh",
    ".sha256",
    ".toml",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}
TEXT_NAMES = {".gitattributes", ".gitignore", "LICENSE", "requirements.txt"}
SECRET_PATTERNS = {
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "GitHub token": re.compile(r"(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}"),
    "OpenAI-style key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "IPv4 address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "Windows absolute path": re.compile(r"(?i)\b[A-Z]:\\(?:Users|Python|Data)\\"),
    "Linux home path": re.compile(r"/(?:home|Users)/[^/\s]+/"),
}


def _figure_hashes(problems: list[str]) -> dict[str, str]:
    """Read and validate the exact path-to-SHA-256 figure allowlist."""

    value = POLICY.get("allowed_figure_sha256")
    if not isinstance(value, dict):
        problems.append("invalid allowed_figure_sha256 policy")
        return {}
    result: dict[str, str] = {}
    for path, digest in value.items():
        if not isinstance(path, str) or not isinstance(digest, str):
            problems.append("invalid allowed_figure_sha256 entry")
            continue
        digest = digest.casefold()
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            problems.append(f"invalid figure SHA-256: {path}")
            continue
        result[path] = digest
    return result


def _banned_hashes(problems: list[str]) -> dict[int, set[str]]:
    """Validate hashed private literals without exposing them in the release."""

    result: dict[int, set[str]] = {}
    try:
        for item in POLICY.get("banned_content_hashes", []):
            length = int(item["length"])
            digest = str(item["sha256"]).casefold()
            if length <= 0 or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError
            result.setdefault(length, set()).add(digest)
    except (KeyError, TypeError, ValueError):
        problems.append("invalid banned_content_hashes policy")
    return result


def main() -> int:
    """Audit the complete release tree and return non-zero on any uncertainty."""

    problems: list[str] = []
    figure_hashes = _figure_hashes(problems)
    allowed_figures = set(figure_hashes)
    expected_count = int(POLICY.get("expected_figure_count", -1))
    if len(allowed_figures) != expected_count:
        problems.append(
            "allowed_figure_paths count mismatch: "
            f"actual={len(allowed_figures)} expected={expected_count}"
        )
    for relative in sorted(allowed_figures):
        path = Path(relative)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in relative
            or len(path.parts) != 2
            or path.parts[0] != "figures"
            or path.suffix.casefold() not in {".png", ".pdf"}
        ):
            problems.append(f"invalid allowlisted figure path: {relative}")

    forbidden_suffixes = {
        str(item).casefold() for item in POLICY.get("forbidden_suffixes", [])
    }
    forbidden_names = {
        str(item).casefold() for item in POLICY.get("forbidden_science_filenames", [])
    }
    patient_patterns: list[re.Pattern[str]] = []
    try:
        patient_patterns = [
            re.compile(str(pattern))
            for pattern in POLICY.get("patient_identifier_patterns", [])
        ]
    except re.error:
        problems.append("invalid patient_identifier_patterns policy")
    banned_hashes = _banned_hashes(problems)
    max_bytes = int(POLICY.get("max_file_bytes", 20 * 1024 * 1024))
    max_json_bytes = int(POLICY.get("max_json_bytes", 2 * 1024 * 1024))
    present_figures: set[str] = set()

    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(ROOT)
        if any(part in EXCLUDED_DIRS for part in relative_path.parts):
            continue
        if relative_path.as_posix() in EXCLUDED_ROOT_FILES:
            continue
        relative = relative_path.as_posix()
        suffix = path.suffix.casefold()

        if path.name.casefold() in forbidden_names and relative != "MANIFEST.json":
            problems.append(f"forbidden scientific file: {relative}")
        if relative in allowed_figures:
            present_figures.add(relative)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != figure_hashes[relative]:
                problems.append(f"allowlisted figure SHA-256 mismatch: {relative}")
        elif suffix in forbidden_suffixes:
            problems.append(f"forbidden file type: {relative}")
        elif path.name not in TEXT_NAMES and suffix not in TEXT_SUFFIXES:
            problems.append(f"unapproved non-text file: {relative}")

        size = path.stat().st_size
        if size > max_bytes:
            problems.append(f"oversized file: {relative}")
        if suffix == ".json" and size > max_json_bytes:
            problems.append(f"oversized JSON: {relative}")
        if relative in allowed_figures:
            continue
        if path.name not in TEXT_NAMES and suffix not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            problems.append(f"non-UTF-8 text: {relative}")
            continue

        surfaces = [relative.casefold(), text.casefold()]
        for pattern in patient_patterns:
            if any(pattern.search(surface) for surface in surfaces):
                problems.append(f"patient identifier pattern in {relative}")
        for length, digests in banned_hashes.items():
            matched = next(
                (
                    digest
                    for surface in surfaces
                    for start in range(max(0, len(surface) - length + 1))
                    for digest in [
                        hashlib.sha256(
                            surface[start : start + length].encode("utf-8")
                        ).hexdigest()
                    ]
                    if digest in digests
                ),
                None,
            )
            if matched:
                problems.append(f"banned content hash in {relative}: {matched[:12]}")
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                problems.append(f"{label} in {relative}")

    missing = sorted(allowed_figures - present_figures)
    if missing:
        problems.append(f"allowlisted figures missing: {missing}")
    if problems:
        print("PRIVACY AUDIT FAILED")
        for problem in problems:
            print(f"- {problem}")
        return 1
    print(
        "PRIVACY AUDIT PASS: exact figure allowlist, scientific filenames, "
        "patient patterns, binary types, sizes, credentials and host paths"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

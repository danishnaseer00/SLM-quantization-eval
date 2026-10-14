"""Verify the artifact without installing anything.

    python analysis/verify_artifact.py

Checks every entry in ARTIFACT_MANIFEST.sha256 against the bytes on disk and
reports any file shipped in the repository that the manifest does not cover.

Exit status is 0 only when every hash matches and nothing is uncovered.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "ARTIFACT_MANIFEST.sha256"


def git_tracked() -> set[str]:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=ROOT,
            capture_output=True,
        ).stdout.decode("utf-8")
    except OSError:
        return set()
    return {p for p in out.split("\0") if p}


def main() -> int:
    if not MANIFEST.is_file():
        print(f"missing manifest: {MANIFEST}")
        return 1

    listed: dict[str, str] = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        listed[name.strip()] = digest

    ok, bad, missing = 0, [], []
    for name, expected in listed.items():
        path = ROOT / name
        if not path.is_file():
            missing.append(name)
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual == expected:
            ok += 1
        else:
            bad.append(name)

    tracked = git_tracked()
    uncovered = sorted(
        t
        for t in tracked
        if t not in listed and t != "ARTIFACT_MANIFEST.sha256"
    )

    for name in missing:
        print(f"MISSING   {name}")
    for name in bad:
        print(f"MISMATCH  {name}")
    for name in uncovered:
        print(f"UNCOVERED {name}")

    print()
    print(f"manifest entries : {len(listed)}")
    print(f"verified         : {ok}")
    print(f"mismatched       : {len(bad)}")
    print(f"missing          : {len(missing)}")
    print(f"uncovered files  : {len(uncovered)}")

    if bad or missing:
        print("\nFAILED: the artifact on disk does not match its manifest.")
        return 1
    if uncovered:
        print(
            "\nWARNING: files are shipped without hashes. "
            "Rerun analysis/reanalyze.py to refresh the manifest."
        )
        return 1
    print("\nOK: every shipped file matches its recorded SHA-256.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
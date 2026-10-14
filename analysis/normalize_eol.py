"""Normalise tracked TEXT files in the working tree to LF.

Pairs with .gitattributes (`* -text`) so that the bytes git stores are the
bytes ARTIFACT_MANIFEST.sha256 hashes, on every platform.

    python analysis/normalize_eol.py

Binary files are skipped: rewriting \\r\\n inside a zip or JSON blob corrupts
it. Text-ness is decided by the null-byte test, not by extension.
"""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".gitattributes"}

tracked = subprocess.run(
    ["git", "ls-files", "-z"], capture_output=True, cwd=ROOT
).stdout.decode("utf-8").split("\0")

changed, skipped_binary = [], []

for name in tracked:
    if not name or name in SKIP:
        continue
    p = ROOT / name
    if not p.is_file():
        continue
    raw = p.read_bytes()
    if b"\x00" in raw[:65536]:
        skipped_binary.append(name)
        continue
    if b"\r\n" not in raw:
        continue
    p.write_bytes(raw.replace(b"\r\n", b"\n"))
    changed.append(name)

print(f"normalised {len(changed)} text files to LF")
for n in changed:
    print("  ", n)
print(f"\nskipped {len(skipped_binary)} binary files (left byte-identical)")
for n in skipped_binary:
    print("  ", n)
"""Verify the engine against the v0.4 archive manifest without needing the original ZIP."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / 'docs/engine-v0.4-manifest.json').read_text())
for relative, expected in manifest['files'].items():
    actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f'Preservation check failed: {relative}')
print(f"Verified {len(manifest['files'])} unchanged v0.4 source/test files")

"""Compute reproducible content digests for evaluation inputs."""

import json
from collections.abc import Iterable
from hashlib import sha256


def content_digest(files: Iterable[tuple[str, bytes]]) -> str:
    manifest = sorted((name, sha256(data).hexdigest()) for name, data in files)
    return sha256(
        json.dumps(manifest, separators=(",", ":")).encode("ascii")
    ).hexdigest()

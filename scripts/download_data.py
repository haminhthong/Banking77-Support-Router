"""Script tải tập dữ liệu BANKING77 với mã hóa checksum SHA-256 để đảm bảo tính tái lập."""

from __future__ import annotations

import hashlib
from pathlib import Path

import requests

PINNED_COMMIT = "master"
URLS = {
    "train.csv": f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{PINNED_COMMIT}/banking_data/train.csv",
    "test.csv": f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{PINNED_COMMIT}/banking_data/test.csv",
}

EXPECTED_SHA256 = {
    "train.csv": "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
    "test.csv": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
}


def compute_sha256(content: bytes) -> str:
    h = hashlib.sha256()
    h.update(content)
    return h.hexdigest()


def main() -> None:
    out = Path("data/raw")
    out.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        content = r.content
        computed_sha = compute_sha256(content)
        (out / name).write_bytes(content)
        print(f"Downloaded {name}: {len(content):,} bytes | SHA256: {computed_sha}")
        expected = EXPECTED_SHA256.get(name)
        if expected and computed_sha != expected:
            print(f"CẢNH BÁO: Checksum của {name} không khớp với bản gốc đã kiểm thử!")


if __name__ == "__main__":
    main()

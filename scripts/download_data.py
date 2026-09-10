"""Tải Banking77 CSV nếu checkout chưa có dữ liệu."""

from __future__ import annotations

from pathlib import Path

import requests

SOURCE_REF = "master"
URLS = {
    "train.csv": f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{SOURCE_REF}/banking_data/train.csv",
    "test.csv": f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{SOURCE_REF}/banking_data/test.csv",
}


def main() -> None:
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in URLS.items():
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        path = output_dir / filename
        path.write_bytes(response.content)
        print(f"Đã tải {filename}: {len(response.content):,} bytes")


if __name__ == "__main__":
    main()

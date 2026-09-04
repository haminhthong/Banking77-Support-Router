from pathlib import Path

import requests

URLS = {
    "train.csv": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv",
    "test.csv": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv",
}


def main():
    out = Path("data/raw")
    out.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        (out / name).write_bytes(r.content)
        print(f"Downloaded {name}: {len(r.content):,} bytes")


if __name__ == "__main__":
    main()

"""
verify.py
---------
Run at the START of every session. Checks that all prerequisites are in place
before you spend time training or debugging.

Checks:
  1. Python version
  2. Required packages importable
  3. Processed dataset splits exist and are non-empty
  4. Best model file exists (only warns if missing — OK on first session)
  5. SQLite inference log directory exists
  6. API reachable (optional — only if --api flag passed)

Usage:
    python scripts/verify.py
    python scripts/verify.py --api          (also pings the API health endpoint)
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import urllib.error
from pathlib import Path

REQUIRED_PACKAGES = [
    "torch",
    "torchvision",
    "fastapi",
    "uvicorn",
    "streamlit",
    "mlflow",
    "sklearn",
    "albumentations",
    "PIL",
    "cv2",
    "tqdm",
    "dotenv",
]

SPLITS = ["train", "val", "test"]
PROC_DIR  = Path("data/processed")
MODEL_PATH = Path("models/best_model.pth")
LOGS_DIR   = Path("logs")

PASS = "✓"
FAIL = "✗"
WARN = "⚠"


def check_python() -> bool:
    major, minor = sys.version_info[:2]
    ok = (major == 3 and minor >= 10)
    status = PASS if ok else FAIL
    print(f"  {status}  Python {major}.{minor} {'(OK)' if ok else '(need 3.10+)'}")
    return ok


def check_packages() -> bool:
    all_ok = True
    for pkg in REQUIRED_PACKAGES:
        try:
            __import__(pkg)
            print(f"  {PASS}  {pkg}")
        except ImportError:
            print(f"  {FAIL}  {pkg}  ← pip install -r requirements.txt")
            all_ok = False
    return all_ok


def check_dataset() -> bool:
    all_ok = True
    for split in SPLITS:
        split_dir = PROC_DIR / split
        if not split_dir.exists():
            print(f"  {FAIL}  data/processed/{split}/ missing — run: python scripts/prepare_dataset.py")
            all_ok = False
            continue

        count = sum(1 for _ in split_dir.rglob("*.jpg"))
        if count == 0:
            print(f"  {FAIL}  data/processed/{split}/ is empty")
            all_ok = False
        else:
            print(f"  {PASS}  data/processed/{split}/  ({count} images)")
    return all_ok


def check_model() -> bool:
    if MODEL_PATH.exists():
        size_mb = MODEL_PATH.stat().st_size / 1024 / 1024
        import torch
        ckpt = torch.load(str(MODEL_PATH), map_location="cpu")
        model_name = ckpt.get("model_name", "unknown")
        epoch      = ckpt.get("epoch", "?")
        val_f1     = ckpt.get("val_f1", 0.0)
        print(f"  {PASS}  {MODEL_PATH}  ({size_mb:.1f} MB | {model_name} | epoch {epoch} | val F1 {val_f1:.4f})")
        return True
    else:
        print(f"  {WARN}  {MODEL_PATH} not found (OK on first session — run train.py)")
        return True  # Not a blocking failure


def check_logs_dir() -> bool:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  {PASS}  logs/ directory ready")
    return True


def check_api(host: str = "http://localhost:8000") -> bool:
    try:
        with urllib.request.urlopen(f"{host}/health", timeout=3) as resp:
            body = json.loads(resp.read())
            print(f"  {PASS}  API reachable at {host}/health  — status: {body.get('status', 'ok')}")
            return True
    except urllib.error.URLError:
        print(f"  {WARN}  API not running at {host}  (start with: make api)")
        return True  # Not blocking — may not be needed this session
    except Exception as e:
        print(f"  {WARN}  API check failed: {e}")
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", action="store_true", help="Also check API health endpoint")
    args = parser.parse_args()

    print("\n" + "─" * 50)
    print("  Pipeline Integrity Monitor — Session Check")
    print("─" * 50)

    results = []

    print("\n[1] Python version")
    results.append(check_python())

    print("\n[2] Required packages")
    results.append(check_packages())

    print("\n[3] Processed dataset splits")
    results.append(check_dataset())

    print("\n[4] Best model checkpoint")
    results.append(check_model())

    print("\n[5] Logs directory")
    results.append(check_logs_dir())

    if args.api:
        print("\n[6] API health")
        results.append(check_api())

    print("\n" + "─" * 50)
    if all(results):
        print(f"  {PASS}  All checks passed. Ready to work.\n")
    else:
        print(f"  {FAIL}  Some checks failed — fix the issues above before proceeding.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
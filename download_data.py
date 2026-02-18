#!/usr/bin/env python3
"""
download_data.py  –  Download and prepare the ElectroSense PSD dataset

Dataset: ElectroSense PSD Spectrum Dataset
Zenodo:  https://zenodo.org/records/7521246

This script:
1. Downloads the dataset metadata from Zenodo
2. Lists available files and their sizes
3. Downloads a chosen file (or all files)
4. Converts to the .npz format expected by run_surveillance.py

Usage
-----
    # Show available files without downloading:
    python download_data.py --list

    # Download the first available file:
    python download_data.py --download 0

    # Download all files (warning: may be several GB):
    python download_data.py --download all
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ZENODO_RECORD_ID = "7521246"
ZENODO_API_URL   = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
DATA_DIR         = Path("data")


def fetch_record_metadata() -> dict:
    print(f"Fetching metadata from Zenodo record {ZENODO_RECORD_ID} …")
    req = urllib.request.Request(ZENODO_API_URL,
                                  headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read())


def list_files(record: dict):
    files = record.get("files", [])
    print(f"\nAvailable files in record {ZENODO_RECORD_ID}:")
    print(f"  {'#':>3}  {'Filename':<50}  {'Size (MB)':>10}")
    print("  " + "-"*70)
    for i, f in enumerate(files):
        size_mb = f.get("size", 0) / 1024**2
        print(f"  {i:>3}  {f['key']:<50}  {size_mb:>10.1f}")
    return files


def download_file(file_info: dict, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    url      = file_info["links"]["self"]
    filename = file_info["key"]
    dest     = dest_dir / filename
    size_mb  = file_info.get("size", 0) / 1024**2

    if dest.exists():
        print(f"  Already exists: {dest}  ({size_mb:.1f} MB) – skipping.")
        return dest

    print(f"  Downloading {filename}  ({size_mb:.1f} MB) …")
    print(f"  URL: {url}")

    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, 100 * downloaded / total_size)
            print(f"\r  Progress: {pct:.1f}%  ({downloaded/1024**2:.1f}/{total_size/1024**2:.1f} MB)",
                  end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=progress)
    print(f"\n  Saved → {dest}")
    return dest


def convert_to_npz(raw_path: Path):
    """
    Convert the downloaded ElectroSense file to .npz if necessary.
    The actual conversion logic depends on the dataset's internal format.
    This function provides a template – adapt based on the actual file structure.
    """
    import numpy as np

    suffix = raw_path.suffix.lower()
    npz_path = raw_path.with_suffix(".npz")

    if npz_path.exists():
        print(f"  .npz already exists: {npz_path}")
        return npz_path

    if suffix == ".npz":
        print(f"  File is already .npz: {raw_path}")
        return raw_path

    elif suffix == ".npy":
        psd = np.load(raw_path)
        n_time, n_freq = psd.shape
        frequencies = np.linspace(24e6, 1700e6, n_freq)
        timestamps  = np.arange(n_time, dtype=np.float64) * 30.0
        np.savez_compressed(npz_path, psd=psd,
                            frequencies=frequencies, timestamps=timestamps)
        print(f"  Converted .npy → {npz_path}")
        return npz_path

    elif suffix in (".csv", ".txt"):
        import pandas as pd
        df = pd.read_csv(raw_path, index_col=0)
        psd = df.values.astype(np.float32)
        try:
            frequencies = df.columns.astype(np.float64).values
        except Exception:
            n_freq = psd.shape[1]
            frequencies = np.linspace(24e6, 1700e6, n_freq)
        timestamps = np.arange(psd.shape[0], dtype=np.float64) * 30.0
        np.savez_compressed(npz_path, psd=psd,
                            frequencies=frequencies, timestamps=timestamps)
        print(f"  Converted .csv → {npz_path}")
        return npz_path

    else:
        print(f"  WARNING: Unknown format {suffix}.")
        print("  Inspect the file manually and update convert_to_npz() accordingly.")
        print("  Expected keys for .npz: psd (2D, dB), frequencies (Hz), timestamps (s)")
        return raw_path


def main():
    p = argparse.ArgumentParser(
        description="Download ElectroSense PSD dataset from Zenodo"
    )
    p.add_argument("--list", action="store_true",
                   help="List available files without downloading.")
    p.add_argument("--download", type=str, default=None,
                   metavar="INDEX_OR_all",
                   help="Download file by index (0, 1, …) or 'all'.")
    args = p.parse_args()

    try:
        record = fetch_record_metadata()
    except Exception as e:
        print(f"ERROR: Could not fetch Zenodo metadata: {e}")
        print("Check your internet connection or visit:")
        print(f"  https://zenodo.org/records/{ZENODO_RECORD_ID}")
        sys.exit(1)

    files = list_files(record)

    if not files:
        print("\nNo files found in this Zenodo record.")
        sys.exit(0)

    if args.list or args.download is None:
        print("\nRun with --download <index> to download a file.")
        print("Example: python download_data.py --download 0")
        return

    if args.download == "all":
        targets = files
    else:
        try:
            idx = int(args.download)
            targets = [files[idx]]
        except (ValueError, IndexError):
            print(f"ERROR: Invalid index '{args.download}'. "
                  f"Choose 0–{len(files)-1} or 'all'.")
            sys.exit(1)

    print()
    for file_info in targets:
        raw_path = download_file(file_info, DATA_DIR)
        npz_path = convert_to_npz(raw_path)
        print(f"\n  Ready to use:  python run_surveillance.py --input {npz_path}\n")


if __name__ == "__main__":
    main()

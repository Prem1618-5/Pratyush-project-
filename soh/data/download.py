"""Stage 1: download the raw data (MIT batches ~8.3 GB, NASA zip ~210 MB).

Usage:  python -m soh.data.download            # everything
        python -m soh.data.download --nasa-only
Downloads resume if interrupted (HTTP Range), and file sizes are verified.
"""
import argparse
import zipfile

import requests

from soh import config


def _fetch(url, dest, expected_size=None):
    if dest.exists() and expected_size and dest.stat().st_size == expected_size:
        print(f"[ok] {dest.name} already complete")
        return
    have = dest.stat().st_size if dest.exists() else 0
    if expected_size and have > expected_size:          # larger than the real file: corrupt, restart
        print(f"[bad] {dest.name} is {have:,} bytes > expected {expected_size:,}; restarting")
        dest.unlink()
        have = 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    print(f"[get] {dest.name} (resume from {have:,} bytes)")
    with requests.get(url, stream=True, headers=headers, timeout=60) as r:
        if r.status_code == 416:                        # server cannot resume from here: restart
            print(f"[bad] {dest.name}: server refused to resume; restarting")
            dest.unlink()
            return _fetch(url, dest, expected_size)
        r.raise_for_status()
        mode = "ab" if r.status_code == 206 else "wb"
        with open(dest, mode) as fh:
            for chunk in r.iter_content(chunk_size=1 << 22):
                fh.write(chunk)
    if expected_size and dest.stat().st_size != expected_size:
        raise IOError(f"{dest.name}: size {dest.stat().st_size} != expected {expected_size}")


def download_mit():
    for _, (fname, url, size) in config.MIT_FILES.items():
        _fetch(url, config.RAW_DIR / fname, size)


def download_nasa():
    zpath = config.RAW_DIR / "nasa_battery.zip"
    _fetch(config.NASA_URL, zpath, config.NASA_ZIP_BYTES)
    out = config.RAW_DIR / "nasa"
    if not all((out / f"{c}.mat").exists() for c in config.NASA_CELLS):   # repairs a partial unzip
        out.mkdir(exist_ok=True)
        # outer zip contains inner zips; B0005/6/7/18 are in "1. BatteryAgingARC-FY08Q4.zip"
        with zipfile.ZipFile(zpath) as z:
            inner = [n for n in z.namelist() if "FY08Q4" in n][0]
            z.extract(inner, out)
        with zipfile.ZipFile(out / inner) as z:
            for n in z.namelist():
                if n.endswith(".mat"):
                    (out / n.split("/")[-1]).write_bytes(z.read(n))
    print("[ok] NASA cells:", sorted(p.name for p in out.glob("B00*.mat")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nasa-only", action="store_true")
    args = ap.parse_args()
    config.ensure_dirs()
    download_nasa()
    if not args.nasa_only:
        download_mit()

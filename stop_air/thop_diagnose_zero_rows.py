#!/usr/bin/env python3
"""
thop_diagnose_zero_rows.py

For every TARGET case whose extracted CSV had 0 data rows (per
target_manifest.csv from thop_extract_target.py), lists EVERY .log file
that actually exists in that case's folder (not just the one picked as
TARGET), with each file's data-row / event count and Start Date. This
lets you see whether a different file in the same folder -- excluded only
because its Start Date didn't exactly match the folder's date -- actually
holds the real treatment data that the exact-date-match rule missed.

Usage:
  python3 thop_diagnose_zero_rows.py target_manifest.csv --root "/Volumes/Extreme Pro/元データ"
  -> writes zero_rows_diagnosis.csv next to target_manifest.csv
"""

import argparse
import csv
import datetime
import os
import struct
import sys

EPOCH_2000 = datetime.datetime(2000, 1, 1)


def iter_records(data: bytes, start: int):
    off = start
    n = len(data)
    while off + 12 <= n:
        tag = data[off : off + 4]
        length = struct.unpack("<H", data[off + 6 : off + 8])[0]
        ts = struct.unpack("<I", data[off + 8 : off + 12])[0]
        if length < 12 or off + length > n:
            break
        yield tag, ts
        off += length


def find_records_start(data: bytes) -> int:
    candidates = [i for i in (data.find(b"EVNT"), data.find(b"LOGA")) if i != -1]
    if not candidates:
        raise ValueError("no EVNT/LOGA found")
    return min(candidates)


def ts_to_dt(ts: int, tz_offset_hours: float = 9.0) -> datetime.datetime:
    return EPOCH_2000 + datetime.timedelta(seconds=ts + tz_offset_hours * 3600)


def scan_file(path: str):
    """Returns (n_data_rows, n_events, first_ts) without full decoding."""
    data = open(path, "rb").read()
    start = find_records_start(data)
    n_data, n_evt = 0, 0
    first_ts = None
    for tag, ts in iter_records(data, start):
        if first_ts is None:
            first_ts = ts
        if tag == b"LOGA":
            n_data += 1
        elif tag == b"EVNT":
            n_evt += 1
    return n_data, n_evt, first_ts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest_csv", help="target_manifest.csv from thop_extract_target.py")
    ap.add_argument("--root", required=True, help="root folder the 'Folder' column is relative to")
    ap.add_argument("-o", "--out", help="output path (default: zero_rows_diagnosis.csv next to manifest_csv)")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.manifest_csv, encoding="utf-8-sig")))
    zero_folders = sorted({r["Folder"] for r in rows if r.get("n_data_rows") in ("0", "", None)})
    # only keep folders whose TARGET row(s) are ALL zero (a folder can have 2
    # TARGET rows on a split-session day; if either has data we don't flag it)
    by_folder = {}
    for r in rows:
        by_folder.setdefault(r["Folder"], []).append(r)
    zero_folders = [
        f for f in zero_folders
        if all((r.get("n_data_rows") in ("0", "", None)) for r in by_folder[f])
    ]

    print(f"{len(zero_folders)} folder(s) have TARGET row(s) with 0 data rows. Scanning every file in each...")

    out_rows = []
    rescue_candidates = 0
    for folder in zero_folders:
        target_files = {r["File"] for r in by_folder[folder]}
        folder_path = os.path.join(args.root, folder)
        try:
            all_files = sorted(
                fn for fn in os.listdir(folder_path)
                if fn.lower().endswith(".log") and not fn.startswith(".")
            )
        except FileNotFoundError:
            out_rows.append({"Folder": folder, "File": "(folder not found)", "is_current_target": "", "n_data_rows": "", "n_events": "", "start_date": ""})
            continue

        best_alt = None
        for fn in all_files:
            fpath = os.path.join(folder_path, fn)
            try:
                n_data, n_evt, first_ts = scan_file(fpath)
            except Exception as e:
                out_rows.append({"Folder": folder, "File": fn, "is_current_target": fn in target_files, "n_data_rows": f"ERROR: {e}", "n_events": "", "start_date": ""})
                continue
            start_date = ts_to_dt(first_ts).strftime("%Y-%m-%d") if first_ts is not None else ""
            is_target = fn in target_files
            out_rows.append({"Folder": folder, "File": fn, "is_current_target": is_target, "n_data_rows": n_data, "n_events": n_evt, "start_date": start_date})
            if not is_target and n_data > 0 and (best_alt is None or n_data > best_alt):
                best_alt = n_data

        marker = f"  <-- possible rescue: non-TARGET file has {best_alt} data rows" if best_alt else ""
        print(f"  {folder}: {len(all_files)} file(s) total{marker}")
        if best_alt:
            rescue_candidates += 1

    out_path = args.out or os.path.join(os.path.dirname(os.path.abspath(args.manifest_csv)), "zero_rows_diagnosis.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["Folder", "File", "is_current_target", "n_data_rows", "n_events", "start_date"])
        w.writeheader()
        w.writerows(out_rows)

    print(f"\n{len(zero_folders)} zero-row folders scanned; {rescue_candidates} have a non-TARGET file with real data (possible rescue).")
    print(f"Full listing -> {out_path}")


if __name__ == "__main__":
    main()

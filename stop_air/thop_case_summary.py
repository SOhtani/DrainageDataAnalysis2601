#!/usr/bin/env python3
"""
thop_case_summary.py

Phase 2 of the STOP_AIR initial analysis (see PLAN_STOP_AIR.md).

Reads target_manifest.csv (from thop_extract_target.py), keeps only rows
with n_data_rows > 0 (the 69 usable logs / 63 cases), groups multi-segment
same-day folders into a single case, and for each case computes:

  - combined drainage duration (first LOGA timestamp -> last LOGA
    timestamp across all of that case's segments)
  - a DIAMOND-style proxy PAL label from that duration:
      >=120h  -> positive (prolonged)
      24-120h -> negative
      <24h    -> excluded
  - basic descriptive features: total data rows, mean/max air leak,
    mean pressure

Writes one row per case to stop_air_case_summary.csv and prints the
proxy-label breakdown (positive/negative/excluded counts).

Usage:
  python3 thop_case_summary.py ~/Desktop/stop_air_target_csv/target_manifest.csv
"""

import argparse
import csv
import datetime
import os
import statistics
import sys
from collections import defaultdict


def parse_dt(s: str) -> datetime.datetime:
    # format written by thop_log_to_csv.py / thop_extract_target.py:
    # "3/27/23 10:15" (no leading zeros on month/day/hour, 2-digit year)
    date_part, time_part = s.split(" ")
    m, d, y = date_part.split("/")
    h, mi = time_part.split(":")
    yy = int(y)
    yy += 2000 if yy < 70 else 1900
    return datetime.datetime(yy, int(m), int(d), int(h), int(mi))


def read_data_csv(path: str):
    """Returns list of (dt, air_leak, pressure) from a *_data.csv file."""
    rows = []
    with open(path, encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                dt = parse_dt(row["Date"])
                air_leak = float(row["Air leak [ml/min]"])
                pressure = float(row["Pressure [kPa]"])
            except (KeyError, ValueError):
                continue
            rows.append((dt, air_leak, pressure))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest_csv", help="target_manifest.csv from thop_extract_target.py")
    ap.add_argument("-o", "--out", help="output path (default: stop_air_case_summary.csv next to manifest_csv)")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.manifest_csv, encoding="utf-8-sig")))

    def n_rows(r):
        try:
            return int(r.get("n_data_rows") or 0)
        except ValueError:
            return 0

    usable = [r for r in rows if n_rows(r) > 0]
    print(f"Loaded {len(rows)} manifest rows; {len(usable)} have data rows (n_data_rows > 0).")

    by_folder = defaultdict(list)
    for r in usable:
        by_folder[r["Folder"]].append(r)

    print(f"Grouped into {len(by_folder)} case(s) (folders); "
          f"{sum(1 for v in by_folder.values() if len(v) > 1)} case(s) have multiple segments.")

    out_rows = []
    label_counts = {"positive": 0, "negative": 0, "excluded": 0}
    for folder, segs in sorted(by_folder.items()):
        combined = []
        for seg in segs:
            combined.extend(read_data_csv(seg["data_csv"]))
        if not combined:
            print(f"  WARNING: {folder} has n_data_rows>0 in manifest but no rows parsed from CSV -- skipping", file=sys.stderr)
            continue
        combined.sort(key=lambda x: x[0])
        start_dt, end_dt = combined[0][0], combined[-1][0]
        duration_h = (end_dt - start_dt).total_seconds() / 3600.0

        if duration_h >= 120:
            label = "positive"
        elif duration_h >= 24:
            label = "negative"
        else:
            label = "excluded"
        label_counts[label] += 1

        air_leaks = [x[1] for x in combined]
        pressures = [x[2] for x in combined]

        out_rows.append(
            {
                "Folder": folder,
                "n_segments": len(segs),
                "n_data_rows_total": len(combined),
                "start_dt": start_dt.isoformat(sep=" "),
                "end_dt": end_dt.isoformat(sep=" "),
                "duration_hours": round(duration_h, 1),
                "proxy_pal_label": label,
                "mean_air_leak": round(statistics.mean(air_leaks), 1),
                "max_air_leak": round(max(air_leaks), 1),
                "mean_pressure": round(statistics.mean(pressures), 2),
            }
        )

    out_path = args.out or os.path.join(os.path.dirname(os.path.abspath(args.manifest_csv)), "stop_air_case_summary.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "Folder", "n_segments", "n_data_rows_total", "start_dt", "end_dt",
                "duration_hours", "proxy_pal_label", "mean_air_leak", "max_air_leak", "mean_pressure",
            ],
        )
        w.writeheader()
        w.writerows(out_rows)

    print(f"\nWrote {len(out_rows)} case(s) -> {out_path}\n")
    print("Proxy PAL label breakdown:")
    total = sum(label_counts.values())
    for label in ("positive", "negative", "excluded"):
        n = label_counts[label]
        pct = (100.0 * n / total) if total else 0
        print(f"  {label:10s}: {n:3d} ({pct:.1f}%)")
    if label_counts["excluded"] > 0:
        print(
            "\n  NOTE: 'excluded' (<24h) count is > 0. This was expected to be 0 "
            "(all usable logs should span >=24h) -- worth double-checking those "
            "specific case(s) in the output CSV.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()

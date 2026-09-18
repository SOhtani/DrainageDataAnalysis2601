#!/usr/bin/env python3
"""
thop_identify_target.py

Reads the thop_start_times.csv produced by `thop_log_to_csv.py --summary-only`
and, for each session folder, figures out which .log file is the actual
target patient's case and which ones are leftover sessions from OTHER
patients who used the same physical device earlier.

Rule (reverse-engineered and verified against a real 928-file dataset,
89% exact-match rate):

  - Filename pattern:  <device_serial>_<session_index>.log
    device_serial is the physical device's fixed serial number (the same
    device is reused across ~20 patients). session_index is an internal
    counter the device assigns per session and wraps around (e.g. ~99 ->
    1), so a single memory dump can contain sessions from many prior
    patients still sitting in the device's buffer.

  - Folder name pattern:  <case_number>_<YYYY.M.D>(remarks)
    The date is the date the device was retrieved for that case. The
    leading case number is a hospital case ID, unrelated to the device
    serial.

  - Target log = the file(s) in the folder whose Start Date exactly
    matches the folder's date. If none match exactly, the file with the
    closest date is used instead (flagged as an approximate match).
    Everything else in the folder is a residual session from a different
    patient who used the same device previously.

Usage:
  python3 thop_identify_target.py thop_start_times.csv
  -> writes thop_target_cases.csv next to it
"""

import argparse
import csv
import datetime
import os
import re
import sys
from collections import defaultdict

FOLDER_RE = re.compile(r"^(?P<case>\d+)_(?P<y>\d{4})\.(?P<m>\d{1,2})\.(?P<d>\d{1,2})")


def parse_mdy(s: str):
    if not s:
        return None
    m, rest = s.split("/", 1)
    d, y = rest.split("/")
    yy = int(y)
    yy += 2000 if yy < 70 else 1900
    try:
        return datetime.date(yy, int(m), int(d))
    except ValueError:
        return None


def device_serial(filename: str) -> str:
    return filename.split("_", 1)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("summary_csv", help="path to thop_start_times.csv (from --summary-only)")
    ap.add_argument("-o", "--out", help="output path (default: thop_target_cases.csv next to input)")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.summary_csv, encoding="utf-8-sig")))
    groups = defaultdict(list)
    for r in rows:
        if "/" not in r["File"]:
            print(f"  skip (no folder): {r['File']}", file=sys.stderr)
            continue
        folder, fname = r["File"].split("/", 1)
        groups[folder].append((fname, r["Start Date"], r["Start Time"], r["Note"]))

    out_rows = []
    unmatched_folders = []
    for folder, files in sorted(groups.items()):
        m = FOLDER_RE.match(folder)
        if not m:
            for fname, sd, st, note in files:
                out_rows.append((folder, fname, device_serial(fname), sd, st, "UNKNOWN", "folder name didn't match <case>_<date> pattern"))
            unmatched_folders.append(folder)
            continue
        fdate = datetime.date(int(m.group("y")), int(m.group("m")), int(m.group("d")))

        dated = [(fname, sd, st, note, parse_mdy(sd)) for fname, sd, st, note in files]
        exact = [x for x in dated if x[4] == fdate]

        if exact:
            target_names = {x[0] for x in exact}
            role_note = "exact date match" if len(exact) == 1 else f"exact date match ({len(exact)} segments same day)"
        else:
            with_date = [x for x in dated if x[4] is not None]
            if with_date:
                closest = min(with_date, key=lambda x: abs((x[4] - fdate).days))
                target_names = {closest[0]}
                diff = (closest[4] - fdate).days
                role_note = f"no exact match; closest date is {abs(diff)} day(s) {'after' if diff > 0 else 'before'} folder date -- verify manually"
            else:
                target_names = set()
                role_note = "no dated files in this folder at all -- verify manually"

        for fname, sd, st, note, _d in dated:
            if fname in target_names:
                out_rows.append((folder, fname, device_serial(fname), sd, st, "TARGET", role_note))
            else:
                out_rows.append((folder, fname, device_serial(fname), sd, st, "other patient (residual)", ""))

    out_path = args.out or os.path.join(os.path.dirname(os.path.abspath(args.summary_csv)), "thop_target_cases.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Folder", "File", "Device Serial", "Start Date", "Start Time", "Role", "Note"])
        for row in out_rows:
            w.writerow(row)

    n_target = sum(1 for r in out_rows if r[5] == "TARGET")
    n_other = sum(1 for r in out_rows if r[5].startswith("other"))
    n_unknown = sum(1 for r in out_rows if r[5] == "UNKNOWN")
    print(f"Folders processed: {len(groups)}")
    print(f"TARGET rows: {n_target}   other-patient (residual) rows: {n_other}   unresolved: {n_unknown}")
    if unmatched_folders:
        print(f"Folders whose name didn't match <case>_<date> pattern ({len(unmatched_folders)}): {unmatched_folders}", file=sys.stderr)
    print(f"Wrote -> {out_path}")


if __name__ == "__main__":
    main()

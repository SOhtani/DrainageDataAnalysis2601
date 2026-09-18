#!/usr/bin/env python3
"""
thop_extract_target.py

Converts ONLY the TARGET-classified logs (from thop_target_cases.csv,
produced by thop_identify_target.py) into per-case CSV pairs, skipping the
other-patient residual sessions entirely. Also writes one manifest.csv
summarizing every TARGET case in a single table (folder/case id, device
serial, start date/time, row/event counts, output paths) -- handy as the
input list for the next step (proxy-label PAL calculation + DIAMOND model
scoring).

Self-contained: embeds the same binary-format decoder as
thop_log_to_csv.py (no import needed, so this can be pasted standalone).

Usage:
  python3 thop_extract_target.py thop_target_cases.csv \
      --root "/Volumes/Extreme Pro/元データ" \
      --output-dir ~/Desktop/stop_air_target_csv

  --root defaults to the folder that contains thop_target_cases.csv.
  --output-dir defaults to a new "target_csv" folder next to
  thop_target_cases.csv (mirrors <folder>/<file> structure so multiple
  logs in the same case folder don't collide).
"""

import argparse
import csv
import datetime
import os
import struct
import sys

EPOCH_2000 = datetime.datetime(2000, 1, 1)

EVENT_MESSAGES = {
    ("WARN", 306): "Canister full",
    ("INFO", 511): "Canister change was confirmed.",
}


def ascii_str(b: bytes) -> str:
    end = b.find(b"\x00")
    if end != -1:
        b = b[:end]
    return b.decode("ascii", errors="replace").strip()


def iter_records(data: bytes, start: int):
    off = start
    n = len(data)
    while off + 12 <= n:
        tag = data[off : off + 4]
        length = struct.unpack("<H", data[off + 6 : off + 8])[0]
        ts = struct.unpack("<I", data[off + 8 : off + 12])[0]
        if length < 12 or off + length > n:
            break
        payload = data[off + 12 : off + length]
        yield off, tag, ts, payload
        off += length


def find_records_start(data: bytes) -> int:
    candidates = [i for i in (data.find(b"EVNT"), data.find(b"LOGA")) if i != -1]
    if not candidates:
        raise ValueError("Could not locate first EVNT/LOGA record in file")
    return min(candidates)


def ts_to_dt(ts: int, tz_offset_hours: float) -> datetime.datetime:
    return EPOCH_2000 + datetime.timedelta(seconds=ts + tz_offset_hours * 3600)


def fmt_dt(dt: datetime.datetime) -> str:
    return f"{dt.month}/{dt.day}/{dt.strftime('%y')} {dt.hour}:{dt.minute:02d}"


def decode_loga(payload: bytes):
    air_leak = struct.unpack("<f", payload[2:6])[0]
    pressure = struct.unpack("<f", payload[14:18])[0]
    preset = struct.unpack("<f", payload[26:30])[0]
    fluid = struct.unpack("<H", payload[37:39])[0]
    return {
        "preset_pressure": round(preset, 1),
        "pressure": round(pressure, 1),
        "air_leak": round(air_leak),
        "fluid": fluid,
    }


def decode_evnt(payload: bytes, include_diagnostics: bool = False, verbose: bool = False):
    cls = payload[6]
    code = struct.unpack("<H", payload[8:10])[0]
    if cls == 0x01:
        return "ON:Device is running"
    if cls == 0x02:
        return "STDBY:Standby"
    if cls == 0x04:
        msg = EVENT_MESSAGES.get(("WARN", code), "(unknown message)")
        return f"WARN {code}:{msg}"
    if cls == 0x05:
        return f"CLR {code}:Event {code} resolved"
    if cls == 0x08:
        msg = EVENT_MESSAGES.get(("INFO", code))
        if msg is None:
            if verbose:
                print(f"  (suppressed unknown INFO code {code})", file=sys.stderr)
            return None
        return f"INFO {code}:{msg}"
    if cls == 0x09:
        if not include_diagnostics:
            return None
        text = ascii_str(payload[10:])
        return f"[diagnostic] {text}" if text else None
    return None


def convert(path: str, tz_offset_hours: float, include_diagnostics: bool = False, verbose: bool = False):
    data = open(path, "rb").read()
    start = find_records_start(data)
    data_rows = []
    event_rows = []
    for off, tag, ts, payload in iter_records(data, start):
        dt = ts_to_dt(ts, tz_offset_hours)
        if tag == b"LOGA":
            data_rows.append((dt, decode_loga(payload)))
        elif tag == b"EVNT":
            msg = decode_evnt(payload, include_diagnostics, verbose)
            if msg is not None:
                event_rows.append((dt, msg))
    return data_rows, event_rows


def write_csv(data_rows, event_rows, out_prefix: str):
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    data_path = f"{out_prefix}_data.csv"
    with open(data_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Date", "Preset Pressure [kPa]", "Pressure [kPa]", "Air leak [ml/min]", "Fluid [ml]"])
        for dt, v in data_rows:
            w.writerow([fmt_dt(dt), v["preset_pressure"], v["pressure"], v["air_leak"], v["fluid"]])

    events_path = f"{out_prefix}_events.csv"
    with open(events_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Date", "Event"])
        for dt, msg in event_rows:
            w.writerow([fmt_dt(dt), msg])

    return data_path, events_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target_csv", help="path to thop_target_cases.csv (from thop_identify_target.py)")
    ap.add_argument("--root", help="root folder the 'Folder' column is relative to (default: folder containing target_csv)")
    ap.add_argument("--output-dir", help="where to write the per-case CSVs (default: '<target_csv dir>/target_csv/', mirroring <Folder>/<File>)")
    ap.add_argument("--tz-offset", type=float, default=9.0, help="hours to add to UTC log timestamp (default 9 = JST)")
    ap.add_argument("--include-diagnostics", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    target_csv_abspath = os.path.abspath(args.target_csv)
    root = args.root or os.path.dirname(target_csv_abspath)
    out_root = args.output_dir or os.path.join(os.path.dirname(target_csv_abspath), "target_csv")

    rows = list(csv.DictReader(open(args.target_csv, encoding="utf-8-sig")))
    target_rows = [r for r in rows if r["Role"].startswith("TARGET")]
    print(f"Loaded {len(rows)} rows from {args.target_csv}; {len(target_rows)} are TARGET.")

    manifest = []
    ok, failed = 0, []
    for r in target_rows:
        folder, fname = r["Folder"], r["File"]
        src = os.path.join(root, folder, fname)
        out_prefix = os.path.join(out_root, folder, os.path.splitext(fname)[0])
        try:
            data_rows, event_rows = convert(src, args.tz_offset, args.include_diagnostics, args.verbose)
            data_path, events_path = write_csv(data_rows, event_rows, out_prefix)
            ok += 1
            manifest.append(
                {
                    "Folder": folder,
                    "File": fname,
                    "Device Serial": r.get("Device Serial", ""),
                    "Start Date": r.get("Start Date", ""),
                    "Start Time": r.get("Start Time", ""),
                    "Note": r.get("Note", ""),
                    "n_data_rows": len(data_rows),
                    "n_events": len(event_rows),
                    "data_csv": data_path,
                    "events_csv": events_path,
                    "status": "OK",
                }
            )
            print(f"  OK   {folder}/{fname}  ({len(data_rows)} rows, {len(event_rows)} events)")
        except Exception as e:
            failed.append((folder, fname, str(e)))
            manifest.append(
                {
                    "Folder": folder,
                    "File": fname,
                    "Device Serial": r.get("Device Serial", ""),
                    "Start Date": r.get("Start Date", ""),
                    "Start Time": r.get("Start Time", ""),
                    "Note": r.get("Note", ""),
                    "n_data_rows": "",
                    "n_events": "",
                    "data_csv": "",
                    "events_csv": "",
                    "status": f"FAILED: {e}",
                }
            )
            print(f"  FAIL {folder}/{fname}  ({e})", file=sys.stderr)

    manifest_path = os.path.join(out_root, "target_manifest.csv")
    os.makedirs(out_root, exist_ok=True)
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "Folder", "File", "Device Serial", "Start Date", "Start Time", "Note",
                "n_data_rows", "n_events", "data_csv", "events_csv", "status",
            ],
        )
        w.writeheader()
        w.writerows(manifest)

    print(f"\nDone: {ok} succeeded, {len(failed)} failed.")
    print(f"Per-case CSVs written under: {out_root}")
    print(f"Manifest (1 row per TARGET case): {manifest_path}")
    if failed:
        sys.exit(2)


if __name__ == "__main__":
    main()

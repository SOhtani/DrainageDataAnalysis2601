#!/usr/bin/env python3
"""
thop_log_to_csv.py

Converts a raw ThopEasyPlus device log (".log", the file thopeasy imports)
directly into CSV, without needing the thopeasy application.

Reverse-engineered format (verified against two known-good thopeasy xlsx
exports; Data-sheet values matched 100% for 663/663 rows, Events matched
8/8 by timestamp+code):

  Header (91 bytes):
    "ThopEasyPlus" magic, device code, firmware version string, serial
    number. Not needed for conversion, only printed for info.

  Then a sequence of fixed 64-byte records:
    offset 0:  4-byte ASCII tag  ("EVNT" or "LOGA")
    offset 4:  flag (observed always 0x01)
    offset 5:  reserved (observed always 0x00)
    offset 6:  uint16 LE record length (observed always 64)
    offset 8:  uint32 LE timestamp = seconds since 2000-01-01 00:00:00 UTC
    offset 12: 52-byte payload (meaning depends on tag)

  LOGA payload (data sample), offsets relative to payload start:
    [0:2]   uint16  sequence/internal counter (unused)
    [2:6]   float32 Air leak [ml/min]
    [14:18] float32 Pressure (actual) [kPa]
    [26:30] float32 Preset Pressure [kPa]
    [37:39] uint16  Fluid [ml]

  EVNT payload (event), offsets relative to payload start:
    [0:2]  uint16  sequence/internal counter (unused)
    [6]    class byte:
             0x01 = ON      -> "ON:Device is running"
             0x02 = STDBY   -> "STDBY:Standby"
             0x04 = WARN    -> code at [8:10] (uint16 LE)
             0x05 = CLR     -> code at [8:10] (uint16 LE)
             0x08 = INFO    -> code at [8:10] (uint16 LE)
             0x09 = free-text diagnostic message, null-terminated ASCII
                    starting ~[10:] (self-test results, POST, canister
                    fluid summaries, ...)
             0x06 = internal/heartbeat event (fires every 1-4h, code
                    always 0)
    [8:10] uint16 LE event code (for WARN/CLR/INFO classes)

  Verified against two thopeasy-exported xlsx files (663+8 and a shorter
  one): class 0x06 heartbeats and class 0x09 free-text records NEVER
  appear in thopeasy's own "Events" sheet -- they are internal/diagnostic
  only, so this script skips them by default (pass --include-diagnostics
  to see class 0x09 text anyway).

  class 0x08 (INFO) is trickier: of two INFO codes seen in the sample
  data (511 and 514), only 511 showed up in the reference Events sheet;
  514 did not, and no rule distinguishing them was found. To stay
  faithful to thopeasy's real output, INFO events are only emitted when
  their code has a *known* human message (currently just 511); unknown
  INFO codes are logged to stderr with -v but omitted from the CSV.
  WARN/CLR/ON/STDBY are always emitted since every instance seen so far
  appeared in the reference output.

  A code->message table is only known for the codes seen so far (306,
  511). Unknown WARN codes are still emitted (with "(unknown message)")
  since state-change/warning events appear to always be user-visible;
  only INFO is filtered by known-code as described above.

Caveats / things NOT yet verified:
  - The +9h (JST) timestamp offset is hardcoded based on the two Japanese
    sample files seen so far. Pass --tz-offset to override for other
    timezones.
  - Patient identity fields (Hospital/Name/DOB) are NOT present in the raw
    log at all -- thopeasy's "Therapy" sheet is filled in manually at
    export time, so this script leaves it blank.
  - Rounding uses Python's round() (banker's rounding); Excel may round
    differently on exact .5 boundaries. Cosmetic only, last-digit level.
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


def parse_header(data: bytes):
    magic_end = data.find(b"\x00", 0)
    info = {"magic": ascii_str(data[0:magic_end]) if magic_end != -1 else ""}
    # device code / firmware version / serial are variable-length ASCII
    # runs inside the header; extract greedily rather than by fixed offset.
    printable_runs = []
    cur = []
    for b in data[: data.find(b"EVNT") if b"EVNT" in data[:200] else 91]:
        if 32 <= b < 127:
            cur.append(chr(b))
        else:
            if len(cur) >= 3:
                printable_runs.append("".join(cur))
            cur = []
    if len(cur) >= 3:
        printable_runs.append("".join(cur))
    info["strings"] = printable_runs
    return info


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
    # Matches thopeasy's exported style: "4/17/19 18:52" (no leading
    # zeros on month/day/hour, 2-digit year, no seconds).
    return f"{dt.month}/{dt.day}/{dt.strftime('%y')} {dt.hour}:{dt.minute:02d}"


def fmt_date(dt: datetime.datetime) -> str:
    return f"{dt.month}/{dt.day}/{dt.strftime('%y')}"


def fmt_time(dt: datetime.datetime) -> str:
    return f"{dt.hour}:{dt.minute:02d}"


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
    return None  # class 0x06 and anything else: internal heartbeat, never user-visible


def convert(path: str, tz_offset_hours: float, include_diagnostics: bool = False, verbose: bool = False):
    data = open(path, "rb").read()
    start = find_records_start(data)
    header = parse_header(data)
    if verbose:
        print("Header strings:", header["strings"], file=sys.stderr)

    data_rows = []
    event_rows = []
    for off, tag, ts, payload in iter_records(data, start):
        dt = ts_to_dt(ts, tz_offset_hours)
        if tag == b"LOGA":
            vals = decode_loga(payload)
            data_rows.append((dt, vals))
        elif tag == b"EVNT":
            msg = decode_evnt(payload, include_diagnostics, verbose)
            if msg is not None:
                event_rows.append((dt, msg))
    return header, data_rows, event_rows


def write_csv(data_rows, events_rows, out_prefix: str):
    data_path = f"{out_prefix}_data.csv"
    with open(data_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["Date", "Preset Pressure [kPa]", "Pressure [kPa]", "Air leak [ml/min]", "Fluid [ml]"]
        )
        for dt, v in data_rows:
            w.writerow([fmt_dt(dt), v["preset_pressure"], v["pressure"], v["air_leak"], v["fluid"]])

    events_path = f"{out_prefix}_events.csv"
    with open(events_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Date", "Event"])
        for dt, msg in events_rows:
            w.writerow([fmt_dt(dt), msg])

    return data_path, events_path


def find_log_files(root: str, pattern_ext: str):
    """Recursively walk `root` (which may contain many session subfolders,
    each with one or more logs) and yield every file whose extension
    matches pattern_ext (case-insensitive, e.g. '.log')."""
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in sorted(filenames):
            if fn.startswith("._") or fn.startswith("."):
                continue  # skip macOS AppleDouble/resource-fork junk files
            if fn.lower().endswith(pattern_ext.lower()):
                yield os.path.join(dirpath, fn)


def get_start_info(data_rows, event_rows):
    """Returns (start_dt, note) for the summary table: the timestamp of
    the first Data-sheet row (first LOGA record), which is what
    thopeasy's Data sheet itself starts with. Falls back to the first
    event if a file has no data rows at all (e.g. a self-test-only log)."""
    if data_rows:
        return data_rows[0][0], ""
    if event_rows:
        return event_rows[0][0], "(no data rows -- first event time shown)"
    return None, "(no records found)"


def process_one(path: str, args, out_dir_root: str | None):
    if out_dir_root is not None:
        # mirror the source tree under --output-dir instead of writing
        # next to the source file
        rel = os.path.relpath(path, args.batch_root)
        out_prefix = os.path.join(out_dir_root, os.path.splitext(rel)[0])
        os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    else:
        out_prefix = os.path.splitext(path)[0]
    header, data_rows, event_rows = convert(path, args.tz_offset, args.include_diagnostics, args.verbose)
    data_path, events_path = write_csv(data_rows, event_rows, out_prefix)
    return data_path, events_path, len(data_rows), len(event_rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "path",
        help="a single .log file, OR a folder to scan recursively (all session subfolders and every .log inside them are processed in one go)",
    )
    ap.add_argument("-o", "--out-prefix", help="(single-file mode only) output file prefix; default: same path as input, minus extension")
    ap.add_argument("--output-dir", help="(folder mode only) write CSVs into this directory instead of next to each source .log, mirroring the folder structure")
    ap.add_argument("--ext", default=".log", help="(folder mode only) file extension to look for, case-insensitive (default: .log)")
    ap.add_argument("--tz-offset", type=float, default=9.0, help="hours to add to the UTC log timestamp (default 9 = JST)")
    ap.add_argument("--include-diagnostics", action="store_true", help="also include class-0x09 free-text diagnostic records (self-test, POST, canister summaries) -- these do NOT appear in thopeasy's own Events sheet")
    ap.add_argument(
        "--summary-only",
        action="store_true",
        help="don't write per-file _data/_events CSVs; instead scan every log and write ONE combined table of filename + start date + start time (the timestamp of each log's first Data row)",
    )
    ap.add_argument("--summary-out", help="where to write the combined summary table (default: <scanned folder>/thop_start_times.csv)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if args.summary_only:
        root = args.path
        if os.path.isdir(root):
            log_files = list(find_log_files(root, args.ext))
        else:
            log_files = [root]
            root = os.path.dirname(root) or "."
        if not log_files:
            print(f"No *{args.ext} files found under {root}", file=sys.stderr)
            sys.exit(1)
        print(f"Found {len(log_files)} file(s); extracting start date/time...")
        rows = []
        for p in log_files:
            try:
                _header, data_rows, event_rows = convert(p, args.tz_offset, args.include_diagnostics, args.verbose)
                start_dt, note = get_start_info(data_rows, event_rows)
            except Exception as e:
                start_dt, note = None, f"(error: {e})"
            rel = os.path.relpath(p, root)
            rows.append((rel, start_dt, note))
            shown = fmt_dt(start_dt) if start_dt else "N/A"
            print(f"  {rel}: {shown} {note}")

        rows.sort(key=lambda r: (r[1] is None, r[1]))
        out_path = args.summary_out or os.path.join(root, "thop_start_times.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["File", "Start Date", "Start Time", "Note"])
            for rel, dt, note in rows:
                w.writerow([rel, fmt_date(dt) if dt else "", fmt_time(dt) if dt else "", note])
        print(f"\nWrote summary table ({len(rows)} files) -> {out_path}")
        return

    if os.path.isdir(args.path):
        args.batch_root = args.path
        log_files = list(find_log_files(args.path, args.ext))
        if not log_files:
            print(f"No *{args.ext} files found under {args.path}", file=sys.stderr)
            sys.exit(1)
        print(f"Found {len(log_files)} {args.ext} file(s) under {args.path}")
        ok, failed = 0, []
        total_rows, total_events = 0, 0
        for p in log_files:
            try:
                data_path, events_path, n_rows, n_events = process_one(p, args, args.output_dir)
                total_rows += n_rows
                total_events += n_events
                ok += 1
                print(f"  OK   {p}  ({n_rows} rows, {n_events} events) -> {data_path}")
            except Exception as e:
                failed.append((p, str(e)))
                print(f"  FAIL {p}  ({e})", file=sys.stderr)
        print(f"\nDone: {ok} succeeded, {len(failed)} failed, {total_rows} total data rows, {total_events} total events.")
        if failed:
            print("Failed files:", file=sys.stderr)
            for p, err in failed:
                print(f"  {p}: {err}", file=sys.stderr)
            sys.exit(2)
    else:
        out_prefix = args.out_prefix or os.path.splitext(args.path)[0]
        header, data_rows, event_rows = convert(args.path, args.tz_offset, args.include_diagnostics, args.verbose)
        data_path, events_path = write_csv(data_rows, event_rows, out_prefix)
        print(f"Wrote {len(data_rows)} data rows -> {data_path}")
        print(f"Wrote {len(event_rows)} events   -> {events_path}")


if __name__ == "__main__":
    main()

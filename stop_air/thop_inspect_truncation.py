#!/usr/bin/env python3
"""
thop_inspect_truncation.py

Checks whether the "0 data rows" TARGET logs are genuinely empty of LOGA
records, or whether our record-iterator is stopping early because it hit
a record with an unexpected length (a different firmware/device variant
might not use exactly 64-byte records) -- iter_records currently BREAKS
THE WHOLE LOOP the moment length/bounds look wrong, which would silently
truncate parsing and hide any real data sitting later in the file.

For each TARGET file with n_data_rows == 0 in target_manifest.csv, prints:
  - total file size on disk
  - the byte offset where parsing actually stopped
  - how many bytes of the file were NOT reached (0 = genuinely parsed to EOF)
  - the tag/length/timestamp of the last successfully parsed record
  - a hex+ascii dump of the next 64 bytes after the stopping point, so we
    can see by eye what record type/length the parser choked on

Usage:
  python3 thop_inspect_truncation.py target_manifest.csv --root "/Volumes/Extreme Pro/元データ"
"""

import argparse
import csv
import os
import struct
import sys


def find_records_start(data: bytes) -> int:
    candidates = [i for i in (data.find(b"EVNT"), data.find(b"LOGA")) if i != -1]
    if not candidates:
        raise ValueError("no EVNT/LOGA found")
    return min(candidates)


def hexdump(b: bytes) -> str:
    hex_part = " ".join(f"{c:02x}" for c in b)
    ascii_part = "".join(chr(c) if 32 <= c < 127 else "." for c in b)
    return f"{hex_part}\n  ascii: {ascii_part}"


def inspect(path: str):
    data = open(path, "rb").read()
    size = len(data)
    start = find_records_start(data)
    off = start
    n = size
    last_tag, last_len, last_ts, last_off = None, None, None, start
    stop_reason = "reached EOF cleanly"
    while off + 12 <= n:
        tag = data[off : off + 4]
        length = struct.unpack("<H", data[off + 6 : off + 8])[0]
        ts = struct.unpack("<I", data[off + 8 : off + 12])[0]
        if length < 12 or off + length > n:
            stop_reason = f"bad record: tag={tag!r} length={length} at offset={off} (off+length={off+length} vs file size {n})"
            break
        last_tag, last_len, last_ts, last_off = tag, length, ts, off
        off += length
    else:
        # loop ended because off + 12 > n (not enough bytes left for a header)
        if off < n:
            stop_reason = f"trailing {n - off} byte(s) too short for another record header"

    bytes_unreached = n - off
    next_bytes = data[off : off + 64]
    return {
        "size": size,
        "header_start": start,
        "stopped_at_offset": off,
        "bytes_unreached": bytes_unreached,
        "last_record": f"{last_tag!r} len={last_len} ts={last_ts}" if last_tag else "(none -- broke on very first record)",
        "stop_reason": stop_reason,
        "next_bytes_hex": hexdump(next_bytes) if next_bytes else "(EOF)",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest_csv", help="target_manifest.csv from thop_extract_target.py")
    ap.add_argument("--root", required=True, help="root folder the 'Folder' column is relative to")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.manifest_csv, encoding="utf-8-sig")))
    zero_rows = [r for r in rows if r.get("n_data_rows") in ("0", 0)]
    print(f"Inspecting {len(zero_rows)} zero-data-row TARGET file(s)...\n")

    truncated_count = 0
    for r in zero_rows:
        path = os.path.join(args.root, r["Folder"], r["File"])
        print(f"=== {r['Folder']}/{r['File']} ===")
        try:
            info = inspect(path)
        except Exception as e:
            print(f"  ERROR: {e}\n")
            continue
        print(f"  file size:          {info['size']} bytes")
        print(f"  parsed up to offset: {info['stopped_at_offset']}")
        print(f"  bytes NOT reached:   {info['bytes_unreached']}")
        print(f"  last good record:    {info['last_record']}")
        print(f"  stop reason:         {info['stop_reason']}")
        if info["bytes_unreached"] > 0:
            truncated_count += 1
            print(f"  next 64 bytes after stop point:\n  {info['next_bytes_hex']}")
        print()

    print(f"Summary: {truncated_count}/{len(zero_rows)} file(s) stopped BEFORE reaching end of file "
          f"(i.e. parsing was cut short -- real data may be sitting unread beyond the stop point).")


if __name__ == "__main__":
    main()

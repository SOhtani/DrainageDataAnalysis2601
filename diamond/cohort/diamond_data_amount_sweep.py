#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diamond_data_amount_sweep.py

目的:
  「学習データ量とAUCの関係」を、単施設(DIAMOND)の固定held-out testコホートと、
  外部(順天堂RCT-2)コホートの両方について新規に測定し、2枚のグラフ(PNG)を作る。

  ⚠️ 2026-08-19時点の前提: 過去に実際使っていた学習スクリプトは失われているため、
  本スクリプトは同じ目的の別パイプライン(diamond_freeze_dev_model.pyと同一の
  データ読込・前処理・1D-CNNアーキテクチャ)を土台にした「新規実験」である。
  過去の内部AUC学習曲線(添付画像、6h/12h/18h/24h)の再現・厳密な接続ではない。

  設計:
    1. 単施設の生時系列ファイル(--input_dir、totalleak/totalnoleak配下)を全件読み込み、
       horizon(既定24h)に到達した症例のみ残す。
    2. その中から stratified に固定held-out test(既定10%)を1回だけ切り出す。
       これが全fraction共通で使う「単施設testコホート」（曲線1本目の評価対象）。
    3. 残り(train pool)から、--fractionsで指定した割合ごとにstratified抽出し、
       その都度、MinMaxScalerのfit・パディング長(max_rows)を「その抽出データのみ」から
       決定してモデルを学習する（リーク防止、diamond_dev_cohort_cv.pyと同じ設計）。
    4. 外部(RCT)の生時系列Excel(--external_raw_dir、1症例1ファイル、'Data'シート)を
       読み込み、症例IDをdiamond_build_true_pal_labels.pyと同じ正規化ロジックで
       --external_labels_csv(ge5境界で修正済みのtrue_pal_labels.csv)と突合する。
       これが全fraction共通で使う「外部コホート」（曲線2本目の評価対象）。
    5. fractionごとに学習したモデルを、3の固定内部testと4の固定外部コホート
       両方に適用してAUCを記録し、CSV+2枚のPNG(横軸=学習データ数, 縦軸=AUC)を出力する。

  ⚠️ 各fractionの学習サブセットはtrain poolからのstratified再抽出であり、
    小さいfractionが大きいfractionの厳密な部分集合とは限らない（同一seedで独立抽出）。
    学習曲線としての傾向を見る目的では実用上問題ないが、解釈時に留意すること。

  ⚠️ まず --fractions 1.0 だけを指定して1回計測し、elapsed_sec（CSVに出力）を見てから、
    本番の複数fraction実行に進むことを推奨する。

使い方（例、まず1回だけ計測）:
  python diamond_data_amount_sweep.py \
      --input_dir ~/Documents/DIAMOND/DIAMOND/Data-fix \
      --external_raw_dir "/Volumes/Extreme Pro/Thop easy data/..." \
      --external_labels_csv ~/Documents/DIAMOND/DIAMOND/split_models_external/labels_ge5/true_pal_labels.csv \
      --output_dir ~/Documents/DIAMOND/DIAMOND/split_models_external/data_amount_sweep \
      --horizon_hours 24 --fractions 1.0

使い方（本番、5チェックポイント）:
  python diamond_data_amount_sweep.py \
      --input_dir ~/Documents/DIAMOND/DIAMOND/Data-fix \
      --external_raw_dir "/Volumes/Extreme Pro/Thop easy data/..." \
      --external_labels_csv ~/Documents/DIAMOND/DIAMOND/split_models_external/labels_ge5/true_pal_labels.csv \
      --output_dir ~/Documents/DIAMOND/DIAMOND/split_models_external/data_amount_sweep \
      --horizon_hours 24 --fractions 0.2,0.4,0.6,0.8,1.0
"""

from __future__ import annotations

import argparse
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout
from tensorflow.keras.optimizers import Adam

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


LABEL_POSITIVE_DIR = "totalleak"
LABEL_NEGATIVE_DIR = "totalnoleak"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True, help="単施設(DIAMOND)の生データディレクトリ")
    p.add_argument("--external_raw_dir", required=True, help="外部(RCT)の生時系列Excelディレクトリ(1症例1ファイル)")
    p.add_argument("--external_labels_csv", required=True, help="diamond_build_true_pal_labels.pyの出力(ge5修正後)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--horizon_hours", type=int, default=24)
    p.add_argument(
        "--fractions", default="0.2,0.4,0.6,0.8,1.0",
        help="train pool(単施設held-out test除く)に対する割合、カンマ区切り。"
             "まず'1.0'だけで1回計測してから本番の複数fractionに進むことを推奨",
    )
    p.add_argument("--test_size", type=float, default=0.1, help="単施設側の固定held-out testの割合")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--recursive", action="store_true", default=True)
    return p.parse_args()


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


def normalize_colname(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def detect_label_from_path(file_path: Path) -> Optional[int]:
    parts = [p.lower() for p in file_path.parts]
    if LABEL_POSITIVE_DIR in parts:
        return 1
    if LABEL_NEGATIVE_DIR in parts:
        return 0
    return None


def find_candidate_files(input_dir: Path, recursive: bool) -> List[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
    files = [p for p in iterator if p.is_file()]
    allowed_ext = {".csv", ".txt", ".tsv"}
    return [p for p in files if p.suffix.lower() in allowed_ext]


_LAST_GOOD_FORMAT: Dict[str, Optional[str]] = {"encoding": None, "sep": None}


def try_read_table(file_path: Path) -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[str]]:
    """⚠️ 2026-08-19高速化: 旧版は毎ファイル5encoding×5sep(sep=None/python engineを
    最優先で試す最悪の順序)=最大25回、しかも最も遅い自動判定を必ず先に試していたため、
    1000件超のバッチで体感「固まった」ほど遅くなっていた(バグではなく性能問題)。
    同一バッチ内のファイルはほぼ同じ形式である前提で、
    ①直前に成功した(encoding,sep)をまず試す高速パス
    ②Cエンジン+主要区切り文字(明示指定、python engineのsep=Noneより桁違いに速い)
    ③本当にダメな場合のみ低速な自動判定にフォールバック
    の順に変更。結果は変えず、速度だけ改善する。"""
    global _LAST_GOOD_FORMAT
    if _LAST_GOOD_FORMAT["encoding"] is not None:
        enc, sep = _LAST_GOOD_FORMAT["encoding"], _LAST_GOOD_FORMAT["sep"]
        try:
            if sep == "auto":
                df = pd.read_csv(file_path, encoding=enc, sep=None, engine="python")
            else:
                df = pd.read_csv(file_path, encoding=enc, sep=sep, engine="c")
            if df is not None and len(df.columns) >= 2 and len(df) >= 1:
                return df, enc, sep
        except Exception:
            pass

    encodings = ["utf-8-sig", "utf-8", "cp932", "shift_jis", "latin1"]
    fast_seps = [",", "\t", ";"]
    for enc in encodings:
        for sep in fast_seps:
            try:
                df = pd.read_csv(file_path, encoding=enc, sep=sep, engine="c")
                if df is not None and len(df.columns) >= 2 and len(df) >= 1:
                    _LAST_GOOD_FORMAT["encoding"], _LAST_GOOD_FORMAT["sep"] = enc, sep
                    return df, enc, sep
            except Exception:
                continue
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc, sep=None, engine="python")
            if df is not None and len(df.columns) >= 2 and len(df) >= 1:
                _LAST_GOOD_FORMAT["encoding"], _LAST_GOOD_FORMAT["sep"] = enc, "auto"
                return df, enc, "auto"
        except Exception:
            continue
    return None, None, None


def resolve_datetime_column(df: pd.DataFrame) -> Optional[str]:
    normalized = {c: normalize_colname(c) for c in df.columns}
    priority_exact = {"date", "datetime", "timestamp", "time", "recordedtime", "recordtime"}
    for c, n in normalized.items():
        if n in priority_exact:
            return c
    for c, n in normalized.items():
        if "date" in n or "time" in n:
            return c
    return None


def resolve_airleak_column(df: pd.DataFrame) -> Optional[str]:
    normalized = {c: normalize_colname(c) for c in df.columns}
    priority_exact = {"airleak", "airleakmlmin", "airleakml", "leak", "flow"}
    for c, n in normalized.items():
        if n in priority_exact:
            return c
    for c, n in normalized.items():
        if "airleak" in n:
            return c
    for c, n in normalized.items():
        if n == "leak":
            return c
    return None


def clean_and_standardize(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], str]:
    dt_col = resolve_datetime_column(df)
    air_col = resolve_airleak_column(df)
    if dt_col is None:
        return None, "datetime column not found"
    if air_col is None:
        return None, "air leak column not found"
    out = df[[dt_col, air_col]].copy()
    out.columns = ["Date", "Air leak"]
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["Air leak"] = pd.to_numeric(out["Air leak"], errors="coerce")
    out = out.dropna(subset=["Date", "Air leak"]).sort_values("Date").reset_index(drop=True)
    if out.empty:
        return None, "no valid rows after cleaning"
    return out, "ok"


def extract_horizon_series(df: pd.DataFrame, horizon_hours: int) -> Tuple[Optional[np.ndarray], str]:
    """legacy_style/diamond_freeze_dev_model.pyと同一ロジック:
    horizon未到達(drain_removed_before_horizon)は除外。"""
    start_time = df["Date"].min()
    end_required = start_time + pd.Timedelta(hours=horizon_hours)
    end_time = df["Date"].max()
    if end_time < end_required:
        return None, "drain_removed_before_horizon"
    sub = df[df["Date"] <= end_required].copy()
    if sub.empty:
        return None, "empty_horizon_slice"
    return sub[["Air leak"]].to_numpy(dtype=np.float32), "ok"


def pad_data(data: List[np.ndarray], max_rows: int) -> np.ndarray:
    padded = []
    for item in data:
        if item.shape[0] < max_rows:
            item = np.pad(item, ((0, max_rows - item.shape[0]), (0, 0)), "constant")
        else:
            item = item[:max_rows]
        padded.append(item)
    return np.array(padded, dtype=np.float32)


def create_cnn_model(input_shape: Tuple[int, int], learning_rate: float) -> tf.keras.Model:
    """diamond_freeze_dev_model.py と同一アーキテクチャ。"""
    model = Sequential()
    model.add(Conv1D(64, 3, activation="relu", input_shape=input_shape))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(32, 3, activation="relu"))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def normalize_case_id(raw_id: object) -> Tuple[Optional[str], Optional[str], str]:
    """diamond_build_true_pal_labels.py と同一ロジック（ズレ防止のため重複実装）。"""
    s = str(raw_id).strip()
    cleaned = re.sub(r"[（(].*?[）)]", "", s)
    cleaned = re.sub(r"[①②③④⑤⑥⑦⑧⑨⑩]", "", cleaned).strip()
    m = re.match(r"^(\d+)\s*[-_]\s*(\d+)\s*([a-zA-Z]?)", cleaned)
    if not m:
        return None, None, cleaned
    facility = str(int(m.group(1)))
    case_no = str(int(m.group(2)))
    suffix = m.group(3).lower()
    loose_key = f"{facility}-{case_no}"
    return f"{loose_key}{suffix}", loose_key, cleaned


def load_dev_cases(input_dir: Path, recursive: bool, horizon_hours: int) -> pd.DataFrame:
    files = find_candidate_files(input_dir, recursive)
    print(f"[INFO] 単施設: 候補ファイル数 {len(files)}件を読み込み開始")
    records = []
    for i, fp in enumerate(sorted(files)):
        if i > 0 and i % 100 == 0:
            print(f"[INFO] 進捗: {i}/{len(files)}件処理済み")
        label = detect_label_from_path(fp)
        if label is None:
            continue
        raw_df, enc, sep = try_read_table(fp)
        if raw_df is None:
            records.append({"case_id": fp.stem, "label": label, "status": "read_failed"})
            continue
        df, status = clean_and_standardize(raw_df)
        if df is None:
            records.append({"case_id": fp.stem, "label": label, "status": status})
            continue
        series, status2 = extract_horizon_series(df, horizon_hours)
        records.append({"case_id": fp.stem, "label": label, "status": status2, "series": series})
    df_all = pd.DataFrame(records)
    included = df_all[df_all["status"] == "ok"].reset_index(drop=True)
    print(f"[INFO] 単施設: 候補{len(df_all)}件中、horizon到達=included {len(included)}件 "
          f"(陽性{int(included['label'].sum())})")
    return included


def load_external_cases(external_raw_dir: Path, horizon_hours: int, labels_csv: Path) -> pd.DataFrame:
    if not external_raw_dir.exists():
        raise RuntimeError(
            f"--external_raw_dir が存在しません: {external_raw_dir}\n"
            f"実在するフォルダを指定してください（例のパスの'...'は本物のパスではなく省略記号です）。"
        )
    files = sorted(external_raw_dir.rglob("*.xlsx"))
    print(f"[INFO] 外部: xlsxファイル{len(files)}件")
    if len(files) == 0:
        raise RuntimeError(
            f"--external_raw_dir 配下に.xlsxファイルが1件も見つかりません: {external_raw_dir}\n"
            f"パスが正しいか（サブフォルダの奥にxlsxがある場合はそちらを指定）確認してください。"
        )
    records = []
    for i, fp in enumerate(files):
        if i > 0 and i % 50 == 0:
            print(f"[INFO] 外部進捗: {i}/{len(files)}件処理済み")
        try:
            df = pd.read_excel(fp, sheet_name="Data")
        except Exception as e:
            records.append({"case_id": fp.stem, "status": f"read_failed:{e}"})
            continue
        air_col = resolve_airleak_column(df)
        dt_col = resolve_datetime_column(df)
        if air_col is None or dt_col is None:
            records.append({"case_id": fp.stem, "status": "column_not_found"})
            continue
        out = df[[dt_col, air_col]].copy()
        out.columns = ["Date", "Air leak"]
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        out["Air leak"] = pd.to_numeric(out["Air leak"], errors="coerce")
        out = out.dropna(subset=["Date", "Air leak"]).sort_values("Date").reset_index(drop=True)
        if out.empty:
            records.append({"case_id": fp.stem, "status": "no_valid_rows"})
            continue
        series, status2 = extract_horizon_series(out, horizon_hours)
        strict, loose, _cleaned = normalize_case_id(fp.stem)
        records.append({"case_id": fp.stem, "status": status2, "series": series, "_strict_key": strict, "_loose_key": loose})

    df_all = pd.DataFrame(records)
    included = df_all[df_all["status"] == "ok"].reset_index(drop=True)
    print(f"[INFO] 外部: horizon到達=included {len(included)}件")
    if included.empty:
        return included

    labels = pd.read_csv(labels_csv)
    if "case_id_strict_key" not in labels.columns:
        raise RuntimeError("external_labels_csvに case_id_strict_key 列がありません。true_pal_labels.csvを指定してください。")
    lab = labels.rename(columns={"case_id_strict_key": "_strict_key", "case_id_loose_key": "_loose_key"})

    lab_strict = lab.dropna(subset=["_strict_key"]).drop_duplicates("_strict_key", keep=False)
    merged_strict = included.merge(lab_strict[["_strict_key", "true_pal"]], on="_strict_key", how="inner")
    matched_ids = set(merged_strict["case_id"])

    remaining = included[~included["case_id"].isin(matched_ids)]
    lab_loose = lab.dropna(subset=["_loose_key"])
    loose_counts = lab_loose["_loose_key"].value_counts()
    lab_loose_unique = lab_loose[lab_loose["_loose_key"].isin(loose_counts[loose_counts == 1].index)]
    merged_loose = remaining.merge(lab_loose_unique[["_loose_key", "true_pal"]], on="_loose_key", how="inner")

    merged = pd.concat([merged_strict, merged_loose], ignore_index=True, sort=False)
    merged = merged.dropna(subset=["true_pal"])
    merged = merged[merged["true_pal"].isin([0, 1])].copy()
    merged["true_pal"] = merged["true_pal"].astype(int)
    print(f"[INFO] 外部: labels突合成功={len(merged)}件 (陽性{int(merged['true_pal'].sum())})")
    return merged


def eval_at_fraction(
    train_pool: List[np.ndarray], y_pool: np.ndarray,
    fixed_internal: Tuple[List[np.ndarray], np.ndarray],
    fixed_external: Tuple[List[np.ndarray], np.ndarray],
    fraction: float, args: argparse.Namespace,
) -> Dict[str, object]:
    n_pool = len(y_pool)
    if fraction >= 0.999:
        idx = np.arange(n_pool)
    else:
        idx, _ = train_test_split(np.arange(n_pool), train_size=fraction, stratify=y_pool, random_state=args.seed)
    train_series = [train_pool[i] for i in idx]
    train_y = y_pool[idx]

    max_rows = max(item.shape[0] for item in train_series)
    x_padded = pad_data(train_series, max_rows)
    scaler = MinMaxScaler()
    scaler.fit(x_padded.reshape(-1, 1))
    x_scaled = np.array([scaler.transform(item) for item in x_padded], dtype=np.float32)

    set_seeds(args.seed)
    model = create_cnn_model((x_scaled.shape[1], x_scaled.shape[2]), args.learning_rate)
    model.fit(x_scaled, train_y.astype(np.float32), epochs=args.epochs, batch_size=args.batch_size, verbose=0, shuffle=True)

    def score(series_list: List[np.ndarray]) -> np.ndarray:
        xp = pad_data(series_list, max_rows)
        xs = np.array([scaler.transform(item) for item in xp], dtype=np.float32)
        return model.predict(xs, verbose=0).flatten()

    int_series, int_y = fixed_internal
    int_prob = score(int_series)
    int_auc = float(roc_auc_score(int_y, int_prob)) if len(np.unique(int_y)) > 1 else float("nan")

    ext_series, ext_y = fixed_external
    ext_prob = score(ext_series)
    ext_auc = float(roc_auc_score(ext_y, ext_prob)) if len(np.unique(ext_y)) > 1 else float("nan")

    return {
        "fraction": fraction, "n_train": len(idx), "n_train_pos": int(train_y.sum()),
        "internal_test_auc": int_auc, "internal_test_n": len(int_y), "internal_test_n_pos": int(np.sum(int_y)),
        "external_auc": ext_auc, "external_n": len(ext_y), "external_n_pos": int(np.sum(ext_y)),
    }


def main() -> int:
    args = parse_args()
    set_seeds(args.seed)

    input_dir = Path(args.input_dir).expanduser()
    external_raw_dir = Path(args.external_raw_dir).expanduser()
    labels_csv = Path(args.external_labels_csv).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    dev = load_dev_cases(input_dir, args.recursive, args.horizon_hours)
    if dev.empty:
        raise RuntimeError("単施設側でhorizonに到達した症例がありません。")
    y_all = dev["label"].to_numpy(dtype=int)
    series_all = dev["series"].tolist()

    idx_all = np.arange(len(y_all))
    idx_train_pool, idx_test = train_test_split(idx_all, test_size=args.test_size, stratify=y_all, random_state=args.seed)
    train_pool = [series_all[i] for i in idx_train_pool]
    y_pool = y_all[idx_train_pool]
    fixed_internal_series = [series_all[i] for i in idx_test]
    fixed_internal_y = y_all[idx_test]
    print(f"[INFO] 固定held-out test(単施設): n={len(idx_test)} (陽性{int(fixed_internal_y.sum())}) / "
          f"train pool: n={len(idx_train_pool)}")

    ext_df = load_external_cases(external_raw_dir, args.horizon_hours, labels_csv)
    if ext_df.empty:
        raise RuntimeError("外部側でhorizonに到達し、ラベル突合できた症例がありません。")
    fixed_external_series = ext_df["series"].tolist()
    fixed_external_y = ext_df["true_pal"].to_numpy(dtype=int)

    fractions = [float(x) for x in args.fractions.split(",")]
    rows = []
    for i, frac in enumerate(fractions):
        print(f"\n===== fraction={frac} ({i + 1}/{len(fractions)}) =====")
        t0 = time.time()
        row = eval_at_fraction(
            train_pool, y_pool,
            (fixed_internal_series, fixed_internal_y),
            (fixed_external_series, fixed_external_y),
            frac, args,
        )
        dt = time.time() - t0
        row["elapsed_sec"] = dt
        rows.append(row)
        print(
            f"[結果] fraction={frac}: n_train={row['n_train']}, "
            f"internal_auc={row['internal_test_auc']:.3f}, external_auc={row['external_auc']:.3f}, "
            f"elapsed={dt:.1f}秒"
        )

    result_df = pd.DataFrame(rows)
    csv_path = output_dir / f"data_amount_sweep_{args.horizon_hours}h.csv"
    result_df.to_csv(csv_path, index=False)
    total_elapsed = result_df["elapsed_sec"].sum()
    print(f"\n[OK] 保存: {csv_path}")
    print(f"[INFO] 合計所要時間: {total_elapsed:.1f}秒 ({total_elapsed / 60:.1f}分)")
    if len(fractions) == 1:
        print(
            f"[NEXT] 1点あたり約{total_elapsed:.0f}秒。本番(--fractions 0.2,0.4,0.6,0.8,1.0、5点)なら"
            f"おおよそ{total_elapsed * 5:.0f}秒(約{total_elapsed * 5 / 60:.1f}分)が目安です。"
        )

    if HAS_MPL:
        for metric, color, fname, title in [
            ("internal_test_auc", "#2a78d6", f"data_amount_vs_internal_auc_{args.horizon_hours}h.png",
             "Single-center (DIAMOND) held-out test AUC vs training data amount"),
            ("external_auc", "#1baf7a", f"data_amount_vs_external_auc_{args.horizon_hours}h.png",
             "External (RCT-2) AUC vs training data amount"),
        ]:
            fig, ax = plt.subplots(figsize=(6, 4.5))
            ax.plot(result_df["n_train"], result_df[metric], marker="o", color=color, linewidth=2,
                    label=f"AUC_{args.horizon_hours}h")
            ax.set_xlabel("Amount of training data (n)")
            ax.set_ylabel("AUC")
            ax.set_ylim(0.0, 1.0)
            ax.set_title(title)
            ax.legend(loc="lower right")
            fig.tight_layout()
            out_path = output_dir / fname
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"[OK] 保存: {out_path}")
    else:
        print("[WARN] matplotlib未インストールのため画像は保存しません")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

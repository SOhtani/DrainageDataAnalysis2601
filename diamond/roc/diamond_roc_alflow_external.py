#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diamond_roc_alflow_external.py

目的:
  外部(順天堂RCT-2)コホートについて、REDCapの[alflow]（POD1時点のThopaz流量、
  1点測定。「24時間平均リーク量」の代用として2026-08-19に確定）を単一のスコアとして、
  真のPALラベル(true_pal, --labels_csvから、ge5境界で修正済みのものを渡すこと)に対する
  ROC曲線・AUCを算出する。AIモデルを介さない「ルールベースの参考線」。

  コホートは--labels_csvに含まれるsubjid(=rand非欠損+dsdecod=1の192例)にそのまま従う。
  alflowはREDCap上、患者ごとに1回のみ記録される非repeatingフィールドと想定し、
  他の連続変数(fev1,vc,pao2等)と同様「最初の非欠損値」で集約する。

⚠️ 出力はAUC・n・陽性数などの集計値とROC画像のみ。患者単位の行は一切表示しない。

使い方（例）:
  python diamond_roc_alflow_external.py \
      --redcap_csv "/Volumes/Extreme Pro/Red Cap data/ThopazRCT2_raw.csv" \
      --labels_csv ~/Documents/DIAMOND/DIAMOND/split_models_external/labels_ge5/true_pal_labels.csv \
      --output_dir ~/Documents/DIAMOND/DIAMOND/split_models_external/scored_ge5
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--redcap_csv", required=True)
    p.add_argument("--labels_csv", required=True, help="diamond_build_true_pal_labels.py の出力(修正後のもの)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--subjid_col", default="subjid")
    p.add_argument("--alflow_col", default="alflow", help="POD1時点のThopaz流量列。見つからなければキーワード検索する")
    p.add_argument("--encoding", default="utf-8")
    return p.parse_args()


def normalize_colname(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def resolve_col(df: pd.DataFrame, wanted: str) -> Optional[str]:
    if wanted in df.columns:
        return wanted
    norm_wanted = normalize_colname(wanted)
    for c in df.columns:
        if normalize_colname(c) == norm_wanted:
            return c
    return None


def read_redcap_csv(path: Path, encoding: str) -> pd.DataFrame:
    tried = [encoding, "utf-8-sig", "utf-8", "cp932", "shift_jis"]
    last_err = None
    for enc in dict.fromkeys(tried):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except (UnicodeDecodeError, UnicodeError) as e:
            last_err = e
            continue
    raise RuntimeError(f"{path} を読めるエンコーディングが見つかりませんでした: {last_err}")


def aggregate_redcap_long_to_wide(df: pd.DataFrame, subjid_col: str) -> pd.DataFrame:
    def first_non_null(s: pd.Series):
        s2 = s.dropna()
        return s2.iloc[0] if len(s2) else np.nan
    return df.groupby(subjid_col, dropna=False).agg(first_non_null).reset_index()


def main() -> int:
    args = parse_args()
    redcap_path = Path(args.redcap_csv).expanduser()
    labels_path = Path(args.labels_csv).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    df_raw = read_redcap_csv(redcap_path, args.encoding)
    subjid_col = resolve_col(df_raw, args.subjid_col)
    if subjid_col is None:
        raise RuntimeError(f"subjid列が見つかりません。列名候補: {[c for c in df_raw.columns if 'subj' in normalize_colname(c)]}")

    wide = aggregate_redcap_long_to_wide(df_raw, subjid_col)
    print(f"[INFO] subjid単位に集約(全イベント): {len(wide)}症例")

    alflow_col = resolve_col(wide, args.alflow_col)
    if alflow_col is None:
        candidates = [c for c in wide.columns if "flow" in normalize_colname(c)]
        raise RuntimeError(f"alflow列が見つかりません。'flow'を含む列候補: {candidates}")
    print(f"[INFO] alflow列: {alflow_col}")

    labels = pd.read_csv(labels_path)
    if "subjid" not in labels.columns or "true_pal" not in labels.columns:
        raise RuntimeError("labels_csvにsubjid/true_pal列がありません。true_pal_labels.csvを指定してください。")
    print(f"[INFO] labels_csvコホート数: {len(labels)}")

    merged = labels[["subjid", "true_pal"]].merge(
        wide[[subjid_col, alflow_col]].rename(columns={subjid_col: "subjid", alflow_col: "alflow"}),
        on="subjid", how="left",
    )
    merged["alflow"] = pd.to_numeric(merged["alflow"], errors="coerce")
    n_missing_alflow = int(merged["alflow"].isna().sum())
    print(f"[INFO] alflow欠測: {n_missing_alflow}/{len(merged)}")

    scored = merged.dropna(subset=["alflow", "true_pal"]).copy()
    scored = scored[scored["true_pal"].isin([0, 1])]
    scored["true_pal"] = scored["true_pal"].astype(int)
    y_true = scored["true_pal"].to_numpy()
    y_score = scored["alflow"].to_numpy(dtype=float)

    n = len(y_true)
    n_pos = int(y_true.sum())
    if len(np.unique(y_true)) < 2:
        print("[ERROR] 陽性/陰性が揃っていないためAUC計算不可")
        return 1
    auc = float(roc_auc_score(y_true, y_score))
    print(f"[結果] 外部(RCT)コホート: alflow(POD1流量)によるROC AUC={auc:.3f} (n={n}, 陽性{n_pos})")

    scored[["true_pal"]].assign(alflow=y_score).describe().to_csv(output_dir / "roc_alflow_external_score_summary.csv")

    if HAS_MPL:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.plot(fpr, tpr, color="#eb6834", linewidth=2, label=f"alflow (POD1) rule (AUC={auc:.3f}, n={n}, pos={n_pos})")
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Reference (AUC=0.5)")
        ax.set_xlabel("1 - specificity (FPR)")
        ax.set_ylabel("sensitivity (TPR)")
        ax.set_title("External (RCT-2): true PAL vs POD1 airflow(alflow) rule")
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.legend(loc="lower right", fontsize=9)
        ax.set_aspect("equal")
        fig.tight_layout()
        out_path = output_dir / "roc_alflow_external.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[OK] 保存: {out_path}")
    else:
        print("[WARN] matplotlib未インストールのためROC画像は保存しません")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

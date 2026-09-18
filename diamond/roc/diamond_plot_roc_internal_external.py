#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diamond_plot_roc_internal_external.py

目的:
  9:1分割・多ホライズン(full/12h/18h/24h)モデルの internal test / external の
  ROC曲線を、horizonごとに1枚の図に重ね書きして出力する（計4枚）。

  入力:
    --internal_predictions_dir: internal_test_predictions_{horizon}.csv
      (列: case_id, label, ai_probability。diamond_build_true_pal_labels.py系列とは
      無関係、9:1分割学習時の internal test 予測ファイルそのもの)
    --external_merged_dir: external_merged_{horizon}.csv
      (diamond_score_external_predictions.py の出力。列: case_id, ai_probability,
      true_pal 等。true_palが0/1の行のみをスコアリング対象とする)

  出力: roc_internal_external_{horizon}.png を --output_dir に保存（4枚）。
  各図に internal(train90%の残り10%) と external(真のPAL) のROC曲線を重ね、
  それぞれのAUC・n・陽性数を凡例に表示する。参考として対角線(AUC=0.5)も描画。

  ⚠️ 出力するのはROC曲線(集約された性能曲線)のみで、患者単位の行は一切
  表示・保存しない。

使い方（例）:
  python diamond_plot_roc_internal_external.py \
      --internal_predictions_dir ~/Documents/DIAMOND/DIAMOND/split_models \
      --external_merged_dir ~/DIAMOND-local/output/scored \
      --output_dir ~/DIAMOND-local/output/scored \
      --horizons full 12 18 24
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import roc_auc_score, roc_curve
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--internal_predictions_dir", required=True,
                   help="internal_test_predictions_{horizon}.csv があるディレクトリ")
    p.add_argument("--external_merged_dir", required=True,
                   help="diamond_score_external_predictions.py が出力した external_merged_{horizon}.csv があるディレクトリ")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--horizons", nargs="+", default=["full", "12", "18", "24"])
    return p.parse_args()


def internal_predictions_filename(horizon: str) -> str:
    return f"internal_test_predictions_{horizon}.csv" if horizon == "full" else f"internal_test_predictions_{horizon}h.csv"


def load_internal(path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df.dropna(subset=["label", "ai_probability"])
    y = df["label"].astype(int).to_numpy()
    s = df["ai_probability"].to_numpy(dtype=float)
    return y, s


def load_external_merged(path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df.dropna(subset=["true_pal"])
    df = df[df["true_pal"].isin([0, 1])].copy()
    if df.empty:
        return None
    y = df["true_pal"].astype(int).to_numpy()
    s = df["ai_probability"].to_numpy(dtype=float)
    return y, s


def plot_one_horizon(
    horizon: str,
    internal_data: Optional[Tuple[np.ndarray, np.ndarray]],
    external_data: Optional[Tuple[np.ndarray, np.ndarray]],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))

    if internal_data is not None and len(np.unique(internal_data[0])) >= 2:
        y, s = internal_data
        fpr, tpr, _ = roc_curve(y, s)
        auc = roc_auc_score(y, s)
        n_pos = int(y.sum())
        ax.plot(fpr, tpr, color="#2a78d6", linewidth=2,
                label=f"Internal test (AUC={auc:.3f}, n={len(y)}, pos={n_pos})")
    else:
        print(f"[WARN] horizon={horizon}: internalデータが無いか陽性/陰性が揃っていないためinternal曲線は描画しません")

    if external_data is not None and len(np.unique(external_data[0])) >= 2:
        y, s = external_data
        fpr, tpr, _ = roc_curve(y, s)
        auc = roc_auc_score(y, s)
        n_pos = int(y.sum())
        ax.plot(fpr, tpr, color="#eb6834", linewidth=2,
                label=f"External (AUC={auc:.3f}, n={len(y)}, pos={n_pos})")
    else:
        print(f"[WARN] horizon={horizon}: externalデータが無いか陽性/陰性が揃っていないためexternal曲線は描画しません")

    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Reference (AUC=0.5)")
    ax.set_xlabel("1 - specificity (FPR)")
    ax.set_ylabel("sensitivity (TPR)")
    # ⚠️ 日本語グリフ非対応フォント環境での警告/文字化けを避けるため、タイトルは英語表記に統一
    # （diamond_score_external_predictions.py の plot_roc() と同じ方針）。
    horizon_label_en = {"full": "full", "12": "12h", "18": "18h", "24": "24h"}.get(horizon, horizon)
    ax.set_title(f"ROC: internal test vs external — horizon={horizon_label_en}")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[OK] 保存: {out_path}")


def main() -> int:
    args = parse_args()
    if not HAS_SKLEARN:
        print("[ERROR] scikit-learnがインストールされていません。'pip install scikit-learn' を実行してください。")
        return 1
    if not HAS_MPL:
        print("[ERROR] matplotlibがインストールされていません。'pip install matplotlib' を実行してください。")
        return 1

    internal_dir = Path(args.internal_predictions_dir).expanduser()
    external_dir = Path(args.external_merged_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    for horizon in args.horizons:
        internal_path = internal_dir / internal_predictions_filename(horizon)
        external_path = external_dir / f"external_merged_{horizon}.csv"

        internal_data = load_internal(internal_path)
        if internal_data is None:
            print(f"[WARN] horizon={horizon}: internal予測ファイルが見つかりません -> {internal_path}")
        external_data = load_external_merged(external_path)
        if external_data is None:
            print(f"[WARN] horizon={horizon}: external_mergedファイルが見つかりません -> {external_path}")

        if internal_data is None and external_data is None:
            print(f"[ERROR] horizon={horizon}: internal/external両方ともデータが無いためスキップします")
            continue

        out_path = output_dir / f"roc_internal_external_{horizon}.png"
        plot_one_horizon(horizon, internal_data, external_data, out_path)

    print("[OK] 全horizon処理完了。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diamond_plot_roc_by_cohort.py

目的:
  9:1分割・多ホライズン(full/12h/18h/24h)モデルのROC曲線を、コホートごとに
  1枚の図にまとめて出力する（計2枚）:
    - roc_internal_all_horizons.png: internal test の full/12h/18h/24h を重ね書き
    - roc_external_all_horizons.png: external の full/12h/18h/24h を重ね書き

  diamond_plot_roc_internal_external.py（horizonごとにinternal/externalを重ねる版、
  計4枚）とは軸の切り方が逆（こちらはコホートごとに1枚、horizonを重ねる）。

  入力:
    --internal_predictions_dir: internal_test_predictions_{horizon}.csv
      (列: case_id, label, ai_probability)
    --external_merged_dir: external_merged_{horizon}.csv
      (diamond_score_external_predictions.py の出力。列: case_id, ai_probability,
      true_pal 等。true_palが0/1の行のみをスコアリング対象とする)

  ⚠️ 出力するのはROC曲線(集約された性能曲線)のみで、患者単位の行は一切
  表示・保存しない。

使い方（例）:
  python diamond_plot_roc_by_cohort.py \
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

# horizonごとの色（既存のdataviz検証済みカテゴリカルパレットに準拠、4色目を追加）
HORIZON_COLORS = {
    "full": "#2a78d6",  # blue
    "12": "#eb6834",    # orange
    "18": "#1baf7a",    # aqua/green
    "24": "#8b5cf6",    # purple
}
HORIZON_LABEL_EN = {"full": "full", "12": "12h", "18": "18h", "24": "24h"}


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
    if df.empty:
        return None
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


def plot_cohort(
    cohort_label: str,
    horizon_data: dict,  # horizon -> (y, s) or None
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))

    any_plotted = False
    for horizon, data in horizon_data.items():
        if data is None or len(np.unique(data[0])) < 2:
            print(f"[WARN] {cohort_label}/horizon={horizon}: データが無いか陽性/陰性が揃っていないためスキップします")
            continue
        y, s = data
        fpr, tpr, _ = roc_curve(y, s)
        auc = roc_auc_score(y, s)
        n_pos = int(y.sum())
        ax.plot(
            fpr, tpr, color=HORIZON_COLORS.get(horizon, "#666666"), linewidth=2,
            label=f"{HORIZON_LABEL_EN.get(horizon, horizon)} (AUC={auc:.3f}, n={len(y)}, pos={n_pos})",
        )
        any_plotted = True

    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Reference (AUC=0.5)")
    ax.set_xlabel("1 - specificity (FPR)")
    ax.set_ylabel("sensitivity (TPR)")
    ax.set_title(f"ROC by horizon — {cohort_label}")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_aspect("equal")
    fig.tight_layout()

    if not any_plotted:
        print(f"[ERROR] {cohort_label}: 描画できるhorizonが1つもありませんでした。保存をスキップします。")
        plt.close(fig)
        return

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

    internal_by_horizon = {}
    external_by_horizon = {}
    for horizon in args.horizons:
        internal_path = internal_dir / internal_predictions_filename(horizon)
        internal_by_horizon[horizon] = load_internal(internal_path)
        if internal_by_horizon[horizon] is None:
            print(f"[WARN] internal/horizon={horizon}: ファイルが見つかりません -> {internal_path}")

        external_path = external_dir / f"external_merged_{horizon}.csv"
        external_by_horizon[horizon] = load_external_merged(external_path)
        if external_by_horizon[horizon] is None:
            print(f"[WARN] external/horizon={horizon}: ファイルが見つかりません -> {external_path}")

    plot_cohort("Internal test", internal_by_horizon, output_dir / "roc_internal_all_horizons.png")
    plot_cohort("External", external_by_horizon, output_dir / "roc_external_all_horizons.png")

    print("[OK] 処理完了。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

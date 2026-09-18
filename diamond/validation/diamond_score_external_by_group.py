#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diamond_score_external_by_group.py

目的:
  diamond_score_external_predictions.py が保存した external_merged_{horizon}.csv
  （case_id, ai_probability, true_pal, rand 等を含む）を使い、外部(RCT-2)検証結果を
  割付群別（rand: 1=Group A [-8 cmH2O] / 2=Group B [-15 cmH2O]）に分けて算出する。

  各群について:
    - 外部AUC（ブートストラップ95%CI付き）
    - その群自身のYouden最適閾値での感度・特異度・PPV・NPV・TP/FP/FN/TN
  を出す。

  ⚠️ 群ごとにn・陽性数がさらに減る（全体165前後 -> 各群90前後）ため、
  Group A/Bの差が統計的に有意かは慎重に解釈すること（区間の重なりを必ず確認する）。

⚠️ 出力は集計値とROC画像のみ。症例単位の行は一切表示しない。

使い方（例）:
  python diamond_score_external_by_group.py \
      --scored_dir ~/Documents/DIAMOND/DIAMOND/split_models_external/scored_ge5 \
      --output_dir ~/Documents/DIAMOND/DIAMOND/split_models_external/scored_ge5/by_group \
      --horizons full 12 18 24
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


GROUP_LABELS = {1.0: "Group A (-8 cmH2O)", 2.0: "Group B (-15 cmH2O)"}
GROUP_COLORS = {1.0: "#2a78d6", 2.0: "#eb6834"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--scored_dir", required=True, help="diamond_score_external_predictions.pyのoutput_dir(external_merged_{horizon}.csvがある場所)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--horizons", nargs="+", default=["full", "12", "18", "24"])
    p.add_argument("--bootstrap_n", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def bootstrap_auc_ci(y_true: np.ndarray, y_score: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, ys))
    if not aucs:
        return float("nan"), float("nan")
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def youden_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y_true, y_score)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


def confusion_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    def sd(a, b):
        return a / b if b else float("nan")
    return {
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "sensitivity": sd(tp, tp + fn), "specificity": sd(tn, tn + fp),
        "PPV": sd(tp, tp + fp), "NPV": sd(tn, tn + fn),
        "accuracy": sd(tp + tn, tp + tn + fp + fn),
    }


def main() -> int:
    args = parse_args()
    scored_dir = Path(args.scored_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[dict] = []

    for horizon in args.horizons:
        merged_path = scored_dir / f"external_merged_{horizon}.csv"
        if not merged_path.exists():
            print(f"[WARN] horizon={horizon}: {merged_path} が見つかりません（スキップ）")
            continue
        df = pd.read_csv(merged_path)
        if "rand" not in df.columns:
            print(f"[ERROR] horizon={horizon}: rand列がありません。diamond_score_external_predictions.pyの出力か確認してください")
            continue

        scored = df.dropna(subset=["true_pal"])
        scored = scored[scored["true_pal"].isin([0, 1])].copy()
        scored["true_pal"] = scored["true_pal"].astype(int)
        scored["rand"] = pd.to_numeric(scored["rand"], errors="coerce")

        print(f"\n===== horizon={horizon} =====")
        rand_counts = scored["rand"].value_counts(dropna=True).to_dict()
        print(f"[INFO] 群別内訳(rand非欠損のみ): {rand_counts}")

        if HAS_MPL:
            fig, ax = plt.subplots(figsize=(5.5, 5.5))

        for rand_val, group_name in GROUP_LABELS.items():
            sub = scored[scored["rand"] == rand_val]
            n = len(sub)
            n_pos = int(sub["true_pal"].sum())
            if n == 0 or sub["true_pal"].nunique() < 2:
                print(f"[{group_name}] n={n}, 陽性={n_pos} -> AUC計算不可")
                continue
            y = sub["true_pal"].to_numpy()
            x = sub["ai_probability"].to_numpy(dtype=float)

            auc = float(roc_auc_score(y, x))
            ci_lo, ci_hi = bootstrap_auc_ci(y, x, args.bootstrap_n, args.seed)
            thr = youden_threshold(y, x)
            pred = (x >= thr).astype(int)
            cm = confusion_metrics(y, pred)

            print(
                f"[{group_name}] n={n} (陽性{n_pos}), AUC={auc:.3f} (95%CI {ci_lo:.3f}-{ci_hi:.3f}), "
                f"Youden閾値={thr:.4f}, 感度={cm['sensitivity']:.3f}, 特異度={cm['specificity']:.3f}, "
                f"PPV={cm['PPV']:.3f}, NPV={cm['NPV']:.3f}"
            )

            summary_rows.append({
                "horizon": horizon, "group": group_name, "rand": rand_val,
                "n": n, "n_positive": n_pos,
                "auc": auc, "auc_ci_lo": ci_lo, "auc_ci_hi": ci_hi,
                "youden_threshold": thr,
                **cm,
            })

            if HAS_MPL:
                fpr, tpr, _ = roc_curve(y, x)
                ax.plot(fpr, tpr, color=GROUP_COLORS[rand_val], linewidth=2,
                         label=f"{group_name} (AUC={auc:.3f}, n={n}, pos={n_pos})")

        if HAS_MPL:
            ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Reference (AUC=0.5)")
            ax.set_xlabel("1 - specificity (FPR)")
            ax.set_ylabel("sensitivity (TPR)")
            horizon_label = {"full": "full", "12": "12h", "18": "18h", "24": "24h"}.get(horizon, horizon)
            ax.set_title(f"External ROC by group — horizon={horizon_label}")
            ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
            ax.legend(loc="lower right", fontsize=8)
            ax.set_aspect("equal")
            fig.tight_layout()
            out_path = output_dir / f"roc_external_by_group_{horizon}.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"[OK] 保存: {out_path}")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        csv_path = output_dir / "external_auc_by_group_summary.csv"
        summary_df.to_csv(csv_path, index=False)
        print(f"\n[OK] 全horizon横断サマリを保存: {csv_path}")
    else:
        print("\n[ERROR] 1件も計算できませんでした。")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diamond_score_external_predictions.py

目的:
  diamond_build_true_pal_labels.py が出力した true_pal_labels.csv と、
  9:1分割モデル(full/12h/18h/24h)を外部コホートに適用して得た
  external_predictions_{horizon}.csv を症例IDで突合し、
  「3つ目のAUC」＝外部検証AUC（真のPALラベルに対するAI予測のAUROC）を
  horizonごとに算出する。

  あわせて:
    - 95%信頼区間（bootstrap, デフォルト2000回）
    - Youden最適閾値での混同行列・感度/特異度/PPV/NPV
    - ルールベース比較（POD air leak > 100mL/min, pod_gt100列）のAUROC・感度特異度
    - train(90%, in-sample) / internal test(10%, held-out) / external の3AUC統合表
      (train・internal testの値は学習ログから読み取れないため、
       --internal_metrics_json で与えるか、本スクリプト既定値
       ＝2026-08-15に貼っていただいたログの値を使う)

  IDの突合は diamond_build_true_pal_labels.py と同じ正規化ロジック
  （厳密一致 → 接尾辞/注記を無視した緩い一致 の2段階）を用いる。
  緩い一致で複数候補が生じた場合は "ambiguous" としてマージせず、
  unmatched_ids_{horizon}.csv に出力するので手動確認すること。

使い方（例）:
  python diamond_score_external_predictions.py \
      --labels_csv ~/Documents/DIAMOND/DIAMOND/split_models_external/true_pal_labels.csv \
      --predictions_dir ~/Documents/DIAMOND/DIAMOND/split_models_external \
      --output_dir ~/Documents/DIAMOND/DIAMOND/split_models_external/scored \
      --horizons full 12 18 24
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


# 2026-08-15にログを貼っていただいた9:1分割・4モデルのtrain/internal test AUC。
# 学習スクリプト自体はこれらの値をファイルに保存していない可能性があるため、
# ここに既定値として埋め込む。--internal_metrics_json で上書き可能。
DEFAULT_INTERNAL_METRICS = {
    "full": {"train_auc_insample": 0.992, "internal_test_auc": 0.982, "internal_test_n": 102, "internal_test_n_pos": 10},
    "12": {"train_auc_insample": 0.838, "internal_test_auc": 0.840, "internal_test_n": 102, "internal_test_n_pos": 10},
    "18": {"train_auc_insample": 0.862, "internal_test_auc": 0.807, "internal_test_n": 101, "internal_test_n_pos": 10},
    "24": {"train_auc_insample": 0.899, "internal_test_auc": 0.901, "internal_test_n": 92, "internal_test_n_pos": 10},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--labels_csv", required=True, help="diamond_build_true_pal_labels.py の出力")
    p.add_argument("--predictions_dir", required=True, help="external_predictions_{horizon}.csv があるディレクトリ")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--horizons", nargs="+", default=["full", "12", "18", "24"])
    p.add_argument("--internal_metrics_json", default=None, help="train/internal test AUC等を上書きするJSON")
    p.add_argument(
        "--internal_predictions_dir", default=None,
        help="2026-08-16追加: internal_test_predictions_{horizon}.csv(列:case_id,label,"
             "ai_probability)があるディレクトリ。指定すると内部test(10%)側のAUC・95%CI・"
             "Youden閾値・感度/特異度/PPV/NPVも直接計算する(--internal_metrics_jsonの"
             "AUC固定値だけでなく、生の予測から再計算した完全な指標セットが手に入る)",
    )
    p.add_argument("--bootstrap_n", type=int, default=2000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


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
    strict_key = f"{loose_key}{suffix}"
    return strict_key, loose_key, cleaned


def predictions_filename(horizon: str) -> str:
    return f"external_predictions_{horizon}.csv" if horizon == "full" else f"external_predictions_{horizon}h.csv"


def internal_predictions_filename(horizon: str) -> str:
    """2026-08-16追加: internal_test_predictions_{horizon}.csv（列: case_id, label,
    ai_probability）のファイル名パターン。external_predictions_*.csvと同じ命名規則
    （fullのみ接尾辞'h'なし）。"""
    return f"internal_test_predictions_{horizon}.csv" if horizon == "full" else f"internal_test_predictions_{horizon}h.csv"


def add_case_id_keys(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    strict, loose, cleaned = [], [], []
    for v in df[id_col]:
        sk, lk, cl = normalize_case_id(v)
        strict.append(sk)
        loose.append(lk)
        cleaned.append(cl)
    df = df.copy()
    df["_strict_key"] = strict
    df["_loose_key"] = loose
    df["_cleaned_id"] = cleaned
    return df


def merge_predictions_with_labels(
    preds: pd.DataFrame, labels: pd.DataFrame, output_dir: Path, horizon: str
) -> pd.DataFrame:
    preds = add_case_id_keys(preds, "case_id")
    # labelsは既に case_id_strict_key / case_id_loose_key を持つ想定（build_true_pal_labels.pyの出力）
    lab = labels.rename(columns={
        "case_id_strict_key": "_strict_key",
        "case_id_loose_key": "_loose_key",
    })

    # Stage 1: strict key exact match
    lab_strict = lab.dropna(subset=["_strict_key"]).drop_duplicates("_strict_key", keep=False)
    merged_strict = preds.merge(lab_strict, on="_strict_key", suffixes=("", "_label"), how="inner")
    matched_ids = set(merged_strict["case_id"])

    # Stage 2: loose key match（strictで拾えなかった残りのみ）
    remaining_preds = preds[~preds["case_id"].isin(matched_ids)]
    lab_loose = lab.dropna(subset=["_loose_key"])
    loose_counts = lab_loose["_loose_key"].value_counts()
    lab_loose_unique = lab_loose[lab_loose["_loose_key"].isin(loose_counts[loose_counts == 1].index)]
    merged_loose = remaining_preds.merge(lab_loose_unique, on="_loose_key", suffixes=("", "_label"), how="inner")

    merged = pd.concat([merged_strict, merged_loose], ignore_index=True, sort=False)
    matched_ids_final = set(merged["case_id"])

    unmatched_preds = preds[~preds["case_id"].isin(matched_ids_final)]
    if len(unmatched_preds):
        unmatched_path = output_dir / f"unmatched_ids_{horizon}.csv"
        unmatched_preds[["case_id", "_strict_key", "_loose_key", "_cleaned_id"]].to_csv(unmatched_path, index=False)
        print(f"[WARN] horizon={horizon}: 突合できなかった外部症例 {len(unmatched_preds)}件 -> {unmatched_path}")

    print(
        f"[INFO] horizon={horizon}: 予測{len(preds)}件中 strict一致{len(merged_strict)}件 + "
        f"loose一致{len(merged_loose)}件 = 突合成功{len(merged)}件"
    )
    return merged


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def bootstrap_auc_ci(y_true: np.ndarray, y_score: np.ndarray, n_boot: int, alpha: float, seed: int) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n == 0 or len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, ys))
    if not aucs:
        return float("nan"), float("nan")
    lo = float(np.percentile(aucs, 100 * (alpha / 2)))
    hi = float(np.percentile(aucs, 100 * (1 - alpha / 2)))
    return lo, hi


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


def youden_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y_true, y_score)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


def plot_roc(y_true: np.ndarray, ai_score: np.ndarray, rule_score: np.ndarray, horizon: str, out_path: Path) -> None:
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    for score, label in [(ai_score, "AI model"), (rule_score, "Rule (POD air leak)")]:
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, score)
        auc = roc_auc_score(y_true, score)
        ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_xlabel("1 - specificity")
    ax.set_ylabel("sensitivity")
    ax.set_title(f"External validation ROC — horizon={horizon}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    labels_path = Path(args.labels_csv).expanduser()
    predictions_dir = Path(args.predictions_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    internal_metrics = DEFAULT_INTERNAL_METRICS
    if args.internal_metrics_json:
        with open(Path(args.internal_metrics_json).expanduser()) as f:
            internal_metrics = json.load(f)
        print(f"[INFO] internal_metrics_jsonで上書き: {args.internal_metrics_json}")
    else:
        print("[INFO] train/internal test AUCは2026-08-15ログ由来の既定値を使用（--internal_metrics_jsonで上書き可）")

    labels = pd.read_csv(labels_path)
    if "case_id_strict_key" not in labels.columns:
        raise RuntimeError(
            "labels_csvに case_id_strict_key 列がありません。"
            "diamond_build_true_pal_labels.py の出力(true_pal_labels.csv)を指定してください。"
        )

    summary_rows = []
    for horizon in args.horizons:
        pred_path = predictions_dir / predictions_filename(horizon)
        if not pred_path.exists():
            print(f"[WARN] horizon={horizon}: 予測ファイルが見つかりません -> {pred_path}（スキップ）")
            continue
        preds = pd.read_csv(pred_path)
        merged = merge_predictions_with_labels(preds, labels, output_dir, horizon)
        merged_scored = merged.dropna(subset=["true_pal"])
        merged_scored = merged_scored[merged_scored["true_pal"].isin([0, 1])].copy()
        merged_scored["true_pal"] = merged_scored["true_pal"].astype(int)

        merged_out_path = output_dir / f"external_merged_{horizon}.csv"
        merged.to_csv(merged_out_path, index=False)

        n_matched = len(merged)
        n_scored = len(merged_scored)
        n_pos = int(merged_scored["true_pal"].sum())
        print(f"[INFO] horizon={horizon}: 真のPALラベルで評価可能な症例={n_scored}（陽性{n_pos}）")

        y_true = merged_scored["true_pal"].to_numpy()
        ai_score = merged_scored["ai_probability"].to_numpy(dtype=float)

        # ⚠️ 2026-08-16修正: 実データのexternal_predictions_{horizon}.csvには
        # case_id,ai_probabilityの2列しかなく、pod_mean_airleak/pod_gt100
        # (ルールベース比較用)は存在しないことが判明。これらの列が無ければ
        # クラッシュせずルールベース比較だけスキップする（AI側のAUC等は算出継続）。
        has_rule_cols = "pod_mean_airleak" in merged_scored.columns
        if has_rule_cols:
            rule_score = merged_scored["pod_mean_airleak"].to_numpy(dtype=float)
            rule_pred = merged_scored["pod_gt100"].fillna(0).astype(int).to_numpy() if "pod_gt100" in merged_scored else None
        else:
            rule_score = None
            rule_pred = None
            print(f"[INFO] horizon={horizon}: pod_mean_airleak列が無いためルールベース比較はスキップします")

        ai_auc = safe_auc(y_true, ai_score)
        ai_ci_lo, ai_ci_hi = bootstrap_auc_ci(y_true, ai_score, args.bootstrap_n, args.alpha, args.seed)
        if has_rule_cols:
            rule_auc = safe_auc(y_true, rule_score)
            rule_ci_lo, rule_ci_hi = bootstrap_auc_ci(y_true, rule_score, args.bootstrap_n, args.alpha, args.seed)
        else:
            rule_auc = rule_ci_lo = rule_ci_hi = float("nan")

        ai_thr = youden_threshold(y_true, ai_score) if len(np.unique(y_true)) >= 2 else float("nan")
        ai_pred = (ai_score >= ai_thr).astype(int) if not np.isnan(ai_thr) else np.zeros_like(y_true)
        ai_cm = confusion_metrics(y_true, ai_pred)

        rule_cm = confusion_metrics(y_true, rule_pred) if rule_pred is not None else {}

        if has_rule_cols:
            plot_roc(y_true, ai_score, rule_score, horizon, output_dir / f"roc_external_{horizon}.png")
        elif HAS_MPL and len(np.unique(y_true)) >= 2:
            fig, ax = plt.subplots(figsize=(5, 5))
            fpr, tpr, _ = roc_curve(y_true, ai_score)
            ax.plot(fpr, tpr, label=f"AI model (AUC={ai_auc:.3f})")
            ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
            ax.set_xlabel("1 - specificity")
            ax.set_ylabel("sensitivity")
            ax.set_title(f"External validation ROC — horizon={horizon}")
            ax.legend(loc="lower right")
            fig.tight_layout()
            fig.savefig(output_dir / f"roc_external_{horizon}.png", dpi=150)
            plt.close(fig)

        # ⚠️ 2026-08-16追加: internal_test_predictions_{horizon}.csv（case_id,label,
        # ai_probability）が指定されていれば、内部test(10%)側もAUC・95%CI・Youden閾値・
        # 感度/特異度/PPV/NPVを実際の予測値から直接計算する（--internal_metrics_jsonの
        # AUC固定値だけでは感度/特異度が分からないため）。IDの突合は不要（既に1症例1行）。
        internal_row_extra: Dict[str, float] = {}
        if args.internal_predictions_dir:
            int_pred_path = Path(args.internal_predictions_dir).expanduser() / internal_predictions_filename(horizon)
            if int_pred_path.exists():
                int_df = pd.read_csv(int_pred_path)
                int_df = int_df.dropna(subset=["label", "ai_probability"])
                int_y = int_df["label"].astype(int).to_numpy()
                int_score = int_df["ai_probability"].to_numpy(dtype=float)
                int_auc = safe_auc(int_y, int_score)
                int_ci_lo, int_ci_hi = bootstrap_auc_ci(int_y, int_score, args.bootstrap_n, args.alpha, args.seed)
                int_thr = youden_threshold(int_y, int_score) if len(np.unique(int_y)) >= 2 else float("nan")
                int_pred = (int_score >= int_thr).astype(int) if not np.isnan(int_thr) else np.zeros_like(int_y)
                int_cm = confusion_metrics(int_y, int_pred)
                internal_row_extra = {
                    "internal_test_auc_recomputed": int_auc,
                    "internal_test_auc_ci_lo": int_ci_lo,
                    "internal_test_auc_ci_hi": int_ci_hi,
                    "internal_test_youden_threshold": int_thr,
                    **{f"internal_{k}": v for k, v in int_cm.items()},
                }
                print(
                    f"[INFO] horizon={horizon}: internal test再計算 AUC={int_auc:.3f} "
                    f"(95%CI {int_ci_lo:.3f}-{int_ci_hi:.3f}, n={len(int_y)}, 陽性{int(int_y.sum())}), "
                    f"感度={int_cm['sensitivity']:.3f}, 特異度={int_cm['specificity']:.3f}"
                )

                # ⚠️ 2026-08-19追加: 循環参照問題への対応(2026-08-17監査で指摘、handoffメモ3節#4)。
                # 上のai_cm(既存のexternal_youden_threshold)は「評価対象集団(外部)自身」から
                # 閾値を選んでおり、感度/特異度が過度に楽観的になりうる。
                # ここでは単施設(DIAMOND)の内部test(held-out 10%)側だけで決めたYouden閾値
                # (int_thr、外部データは一切参照していない)を"固定閾値"として外部コホートの
                # ai_scoreにそのまま適用し、外部側の感度/特異度/PPV/NPVを再計算する。
                # 本来の意味での「学習側で閾値を決め、externalには適用のみ」に対応する指標。
                if not np.isnan(int_thr):
                    ai_pred_fixed_thr = (ai_score >= int_thr).astype(int)
                    ai_cm_fixed_thr = confusion_metrics(y_true, ai_pred_fixed_thr)
                    internal_row_extra["external_threshold_source"] = "internal_test(single-center, DIAMOND held-out 10%)固定閾値を適用(循環参照なし)"
                    internal_row_extra.update({f"ai_fixedthr_{k}": v for k, v in ai_cm_fixed_thr.items()})
                    print(
                        f"[結果] horizon={horizon}: 単施設由来の固定閾値(={int_thr:.4f})を外部コホートに適用 "
                        f"-> 感度={ai_cm_fixed_thr['sensitivity']:.3f}, 特異度={ai_cm_fixed_thr['specificity']:.3f}, "
                        f"PPV={ai_cm_fixed_thr['PPV']:.3f}, NPV={ai_cm_fixed_thr['NPV']:.3f} "
                        f"(参考: 外部自己閾値={ai_thr:.4f}での感度={ai_cm['sensitivity']:.3f}/特異度={ai_cm['specificity']:.3f})"
                    )
            else:
                print(f"[WARN] horizon={horizon}: internal_test_predictions_dirに{int_pred_path.name}が見つかりません")

        im = internal_metrics.get(horizon, {})
        row = {
            "horizon": horizon,
            "train_auc_insample": im.get("train_auc_insample"),
            "internal_test_auc": im.get("internal_test_auc"),
            "internal_test_n": im.get("internal_test_n"),
            "internal_test_n_pos": im.get("internal_test_n_pos"),
            **internal_row_extra,
            "external_n_predicted": len(preds),
            "external_n_matched": n_matched,
            "external_n_scored": n_scored,
            "external_n_positive": n_pos,
            "external_auc": ai_auc,
            "external_auc_ci_lo": ai_ci_lo,
            "external_auc_ci_hi": ai_ci_hi,
            "external_youden_threshold": ai_thr,
            **{f"ai_{k}": v for k, v in ai_cm.items()},
            "rule_auc_continuous": rule_auc,
            "rule_auc_ci_lo": rule_ci_lo,
            "rule_auc_ci_hi": rule_ci_hi,
            **{f"rule_{k}": v for k, v in rule_cm.items()},
        }
        summary_rows.append(row)

        print(
            f"[結果] horizon={horizon}: external AUC={ai_auc:.3f} "
            f"(95%CI {ai_ci_lo:.3f}-{ai_ci_hi:.3f}, n={n_scored}, 陽性{n_pos}) / "
            f"rule AUC(連続値)={rule_auc:.3f}"
        )

    if not summary_rows:
        print("[ERROR] 1件もスコアリングできませんでした。ファイルパス・IDフォーマットを確認してください。")
        return 1

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "auc_summary_all_horizons.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[OK] 3AUC統合表を保存: {summary_path}")
    print(summary_df[["horizon", "train_auc_insample", "internal_test_auc", "external_auc"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

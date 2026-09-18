#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diamond_check_restriction_of_range.py

目的:
  「開発コホートでPOD1流量のAUROCが0.895と高いのに、外部コホートでは0.594しかない」
  という現象が、study-protocol-v1.md 3.6節に記載した仮説
  （restriction of range: 外部コホートは親RCTの二次登録基準
  [POD1エアリーク100〜1000 mL/min]自体でPOD1流量の分散が絞り込まれた集団のため、
  同じ軸での判別力が失われる）で本当に説明できるかを、実データで直接検証する。

  検証方法:
    diamond_roc_pod1_flow_devcohort.py が保存した症例別CSV
    (--input_dir実行時に自動保存される dev_pod1_flow_case_status.csv、
    case_id, status, label, pod_mean_airleak列を含む)を読み込み、
    ①開発コホート全体でのAUROC(既知の0.895の再確認)
    ②POD1流量を外部コホートの登録基準と同じ100〜1000 mL/minに人為的に絞り込んだ
      部分集団だけでAUROCを再計算(restriction of rangeが原因なら、外部の0.594に
      近い値まで下がるはず)
    ③絞り込み前後の分布(中央値・IQR・group別)を記述統計で示す
    を行う。

⚠️ 出力は集計値のみ。症例単位の値はターミナルに出力しない。

使い方（例）:
  python diamond_check_restriction_of_range.py \
      --case_status_csv ~/Documents/DIAMOND/DIAMOND/split_models_external/scored_ge5/dev_pod1_flow_case_status.csv \
      --low 100 --high 1000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--case_status_csv", required=True,
        help="diamond_roc_pod1_flow_devcohort.py が保存した dev_pod{N}_flow_case_status.csv",
    )
    p.add_argument("--low", type=float, default=100.0, help="外部コホート登録基準の下限(mL/min)")
    p.add_argument("--high", type=float, default=1000.0, help="外部コホート登録基準の上限(mL/min)")
    p.add_argument(
        "--external_auc_ref", type=float, default=0.594,
        help="比較対象の外部コホート実測AUC(diamond_roc_alflow_external.py、ge5修正後の値)",
    )
    p.add_argument("--bootstrap_n", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def bootstrap_auc_ci(y_true: np.ndarray, y_score: np.ndarray, n_boot: int, seed: int) -> tuple:
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


def describe_group(label: str, s: pd.Series) -> str:
    s = s.dropna()
    if len(s) == 0:
        return f"  {label}: n=0"
    return (
        f"  {label}: n={len(s)}, 中央値={s.median():.1f}, IQR=({s.quantile(0.25):.1f}-{s.quantile(0.75):.1f}), "
        f"範囲=({s.min():.1f}-{s.max():.1f})"
    )


def compute_auc(
    df: pd.DataFrame, tag: str, with_ci: bool = False,
    n_boot: int = 2000, seed: int = 42, external_ref: float = None,
) -> None:
    y = df["label"].astype(int).to_numpy()
    x = df["pod_mean_airleak"].to_numpy(dtype=float)
    n = len(y)
    n_pos = int(y.sum())
    if n == 0 or len(np.unique(y)) < 2:
        print(f"[{tag}] n={n}, 陽性={n_pos} -> AUC計算不可(陽性/陰性が揃っていない)")
        return
    auc = float(roc_auc_score(y, x))
    if with_ci:
        ci_lo, ci_hi = bootstrap_auc_ci(y, x, n_boot, seed)
        line = f"[{tag}] n={n} (陽性{n_pos}, 陰性{n - n_pos}), AUC={auc:.3f} (95%CI bootstrap: {ci_lo:.3f}-{ci_hi:.3f})"
        print(line)
        if external_ref is not None:
            in_ci = ci_lo <= external_ref <= ci_hi
            verdict = "CI内（サンプルが小さいだけで矛盾しない）" if in_ci else "CI外（restriction of rangeだけでは説明しきれない可能性）"
            print(f"    -> 外部実測値{external_ref:.3f}は{verdict}")
    else:
        print(f"[{tag}] n={n} (陽性{n_pos}, 陰性{n - n_pos}), AUC={auc:.3f}")


def main() -> int:
    args = parse_args()
    path = Path(args.case_status_csv).expanduser()
    df = pd.read_csv(path)
    df = df[(df["status"] == "ok") & df["pod_mean_airleak"].notna() & df["label"].notna()].copy()
    df["label"] = df["label"].astype(int)
    print(f"[INFO] 読込: {path} (有効症例 n={len(df)})")

    print("\n===== ①開発コホート全体（絞り込みなし、既知の0.895の再確認） =====")
    compute_auc(df, "全体")
    print(describe_group("totalnoleak(陰性)", df.loc[df["label"] == 0, "pod_mean_airleak"]))
    print(describe_group("totalleak(陽性)", df.loc[df["label"] == 1, "pod_mean_airleak"]))

    print(f"\n===== ②外部コホートの登録基準と同じ範囲({args.low:.0f}〜{args.high:.0f} mL/min)に絞り込み =====")
    restricted = df[(df["pod_mean_airleak"] >= args.low) & (df["pod_mean_airleak"] <= args.high)].copy()
    n_before = len(df)
    n_after = len(restricted)
    print(f"[INFO] 絞り込み: {n_before}例 -> {n_after}例 ({n_after / n_before:.1%}が範囲内)"
          f"  ※一般集団の中でこの流量帯に該当するのがどれだけ稀かを示す数字でもある")
    compute_auc(
        restricted, f"{args.low:.0f}-{args.high:.0f}mL/minに絞り込み後", with_ci=True,
        n_boot=args.bootstrap_n, seed=args.seed, external_ref=args.external_auc_ref,
    )
    print(describe_group("totalnoleak(陰性)", restricted.loc[restricted["label"] == 0, "pod_mean_airleak"]))
    print(describe_group("totalleak(陽性)", restricted.loc[restricted["label"] == 1, "pod_mean_airleak"]))

    print(f"\n===== ③参考: 範囲の下限のみ({args.low:.0f} mL/min以上、上限なし)で絞り込み =====")
    low_only = df[df["pod_mean_airleak"] >= args.low].copy()
    compute_auc(
        low_only, f"{args.low:.0f}mL/min以上のみ", with_ci=True,
        n_boot=args.bootstrap_n, seed=args.seed, external_ref=args.external_auc_ref,
    )

    print(
        "\n[結論の見方] ②のAUC点推定が①(0.895)から下がっていること自体が方向性の裏付けになる。"
        "点推定が外部実測値まで届かなくても、95%CIが外部実測値を含んでいれば「サンプルが"
        "小さいだけで矛盾しない」と解釈できる（②③はn=20前後のため区間が広くなりやすい）。"
        "また[INFO]の絞り込み後の該当率(%)自体が、一般集団の中でRCT登録基準に該当する症例が"
        "どれだけ稀か（＝restriction of rangeの強さ）を直接示す数字である点にも注目。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

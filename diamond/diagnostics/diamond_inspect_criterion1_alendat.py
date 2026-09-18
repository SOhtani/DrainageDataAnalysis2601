#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diamond_inspect_criterion1_alendat.py

目的:
  true_pal陽性率(51.0%, 98/192)が親試験報告値(63.8%)より低い件について、
  Criterion2(早期侵襲的処置)の疑いは晴れた(2026-08-17、真因は検証print文のバグと判明)ため、
  残る有力候補であるCriterion1(術後5日超の持続気漏)側の問題を診断する。

  具体的には:
    (a) alendat(エアーリーク消失日)の欠測率。欠測している場合、現行ロジックでは
        「dur_alend_days = NaN → criterion1 = NaN>5 = NaN → fillna(False)」という形で
        **暗黙に陰性扱い**になる(ただしinvasive_any(いつでも侵襲的処置歴あり)があれば
        undetermined判定は免れる)。alendatの欠測が多く、かつその大半が「本当は不明なだけ」
        だとすると、真は陽性のはずの症例が暗黙に陰性へ算入され、陽性率を過小評価している
        可能性がある。
    (b) dur_alend_days(alendat-surgedat日数)の分布。負の値(データ入力エラー疑い)・
        ちょうど5日(境界値)・極端に長い値などの異常パターンがないか。
    (c) Criterion1単独/Criterion2単独/両方/どちらでもない、の内訳(患者数)。
    (d) 「alendat欠測 かつ Criterion2でも陽性にならない」患者数＝現行ロジックで
        暗黙に陰性扱いされている、真の判定が実は不明な症例の候補数。
    (e) 感度分析: alendat欠測(かつcriterion2非該当)を「陰性」ではなく「判定不能」として
        分母から除外した場合の陽性率を再計算し、現行の51.0%と比較する。

  入力は diamond_build_true_pal_labels.py (2026-08-17修正版)の出力
  true_pal_labels.csv。出力は全て集計値のみ(件数・割合・日数分布バケット)。
  患者ID・生の日付は一切出力しない（プライバシー制約）。

使い方（例）:
  python diamond_inspect_criterion1_alendat.py \
      --labels_csv ~/Documents/DIAMOND/DIAMOND/split_models_external/true_pal_labels.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--labels_csv", required=True, help="diamond_build_true_pal_labels.py(修正版)の出力")
    return p.parse_args()


def pct(n: int, d: int) -> str:
    if d == 0:
        return "n/a"
    return f"{n}/{d} ({n / d * 100:.1f}%)"


def main() -> int:
    args = parse_args()
    path = Path(args.labels_csv).expanduser()
    df = pd.read_csv(path)
    n_total = len(df)
    print(f"[INFO] 読込: {path}, 総症例数={n_total}")

    required = {"surgedat", "alendat", "dur_alend_days", "criterion1_over5days",
                "criterion2_early_intervention", "true_pal", "invasive_any_anytime"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        print(f"[WARN] 想定より古い(または列名が異なる)true_pal_labels.csvの可能性: 不足列={missing_cols}")
        print("[WARN] diamond_build_true_pal_labels.py の2026-08-17修正版で再生成してから実行してください。")

    surgedat_missing = df["surgedat"].isna() if "surgedat" in df.columns else pd.Series(False, index=df.index)
    alendat_missing = df["alendat"].isna() if "alendat" in df.columns else pd.Series(False, index=df.index)

    print("\n===== 診断(a): surgedat/alendat 欠測率 =====")
    print(f"  surgedat欠測: {pct(int(surgedat_missing.sum()), n_total)}")
    print(f"  alendat欠測: {pct(int(alendat_missing.sum()), n_total)}")

    if "invasive_any_anytime" in df.columns:
        invasive_any = df["invasive_any_anytime"].astype(bool)
        rescued = alendat_missing & invasive_any  # alendat欠測だがinvasive記録ありでundetermined扱いを免れた
        truly_undetermined_candidates = alendat_missing & ~invasive_any  # 現行ロジックでundetermined(NaN)になっているはず
        print("\n===== 診断(a-2): alendat欠測症例の内訳 =====")
        print(f"  alendat欠測 かつ 侵襲的処置歴あり(rescued, undetermined回避): {int(rescued.sum())}例")
        print(f"  alendat欠測 かつ 侵襲的処置歴なし(undetermined=NaN扱いのはず): {int(truly_undetermined_candidates.sum())}例")
        if "true_pal" in df.columns:
            rescued_true_pal_counts = df.loc[rescued, "true_pal"].value_counts(dropna=False).to_dict()
            print(f"    ↑rescued症例のtrue_pal内訳(0/1/NaN): {rescued_true_pal_counts}")
            print("    ※rescued症例でtrue_pal=0(陰性)になっているものは、実際には持続気漏の")
            print("     真の消失日が不明なだけなのに、Criterion2非該当だったため暗黙に陰性扱い")
            print("     された「隠れ判定不能」症例の候補。")

    if "dur_alend_days" in df.columns:
        print("\n===== 診断(b): dur_alend_days(alendat-surgedat) 日数分布（バケット） =====")

        def bucket(d):
            if pd.isna(d):
                return "欠損(alendat無し)"
            if d < 0:
                return "負の値(データ異常疑い)"
            if d == 0:
                return "0日"
            if d <= 4:
                return "1-4日"
            if d == 5:
                return "5日(境界、criterion1では陰性)"
            if d <= 10:
                return "6-10日"
            if d <= 20:
                return "11-20日"
            if d <= 30:
                return "21-30日"
            return "31日以上"

        bucket_counts = df["dur_alend_days"].apply(bucket).value_counts().to_dict()
        for k in ["欠損(alendat無し)", "負の値(データ異常疑い)", "0日", "1-4日",
                  "5日(境界、criterion1では陰性)", "6-10日", "11-20日", "21-30日", "31日以上"]:
            if k in bucket_counts:
                print(f"  {k}: {bucket_counts[k]}例")

    if {"criterion1_over5days", "criterion2_early_intervention"} <= set(df.columns):
        print("\n===== 診断(c): Criterion1/Criterion2 患者単位の寄与内訳 =====")
        c1 = df["criterion1_over5days"].fillna(False).astype(bool)
        c2 = df["criterion2_early_intervention"].fillna(False).astype(bool)
        both = c1 & c2
        c1_only = c1 & ~c2
        c2_only = c2 & ~c1
        neither = ~c1 & ~c2
        print(f"  Criterion1のみ陽性: {int(c1_only.sum())}例")
        print(f"  Criterion2のみ陽性: {int(c2_only.sum())}例")
        print(f"  両方陽性: {int(both.sum())}例")
        print(f"  どちらも陰性(true_pal=0 or undetermined): {int(neither.sum())}例")
        print(f"  Criterion1陽性合計(単独+重複): {int(c1.sum())}例")
        print(f"  Criterion2陽性合計(単独+重複): {int(c2.sum())}例")

    if "true_pal" in df.columns and "alendat" in df.columns and "criterion2_early_intervention" in df.columns:
        print("\n===== 診断(e): 感度分析(alendat欠測かつCriterion2非該当を「判定不能」として除外) =====")
        c2 = df["criterion2_early_intervention"].fillna(False).astype(bool)
        current_determined = df["true_pal"].notna()
        current_positive = (df["true_pal"] == 1)
        n_det_now = int(current_determined.sum())
        n_pos_now = int(current_positive.sum())
        rate_now = n_pos_now / n_det_now if n_det_now else float("nan")
        print(f"  【現行ロジック】判定可能={n_det_now}, 陽性={n_pos_now} ({rate_now:.1%})")

        # 感度分析: alendat欠測 かつ Criterion2非該当 の症例は「判定不能」として分母から除外
        alt_undetermined = current_determined & alendat_missing & ~c2
        alt_determined = current_determined & ~alt_undetermined
        alt_positive = alt_determined & current_positive
        n_det_alt = int(alt_determined.sum())
        n_pos_alt = int(alt_positive.sum())
        rate_alt = n_pos_alt / n_det_alt if n_det_alt else float("nan")
        print(f"  【感度分析: alendat欠測(Criterion2非該当)を判定不能扱いに変更】"
              f"判定可能={n_det_alt}(-{n_det_now - n_det_alt}), 陽性={n_pos_alt} ({rate_alt:.1%})")
        print("  ※この感度分析で陽性率が63.8%に近づく場合、alendat欠測の暗黙陰性扱いが")
        print("   陽性率過小評価の主因である可能性が高い。近づかない場合は、コホート定義")
        print("   (192 vs 199)側、またはalendat自体の記録精度(消失日の判定基準のブレ)側を疑う。")

    print("\n[OK] 診断完了。上記の集計結果を確認してください（患者ID・生の日付は一切出力していません）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

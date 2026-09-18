#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diamond_build_true_pal_labels.py

目的:
  順天堂多施設Thopaz RCT-2のREDCap生データ(縦持ちexport)から、
  親試験(Takamochi et al., Ann Thorac Surg 2025)の一次評価項目定義に準拠した
  「真のPALラベル」を1症例1行で算出する。

  PAL = (1) 術後5日目以降も持続する気漏 [ (alendat - surgedat).days >= 5 ]
        OR
        (2) 術後4日目までの侵襲的処置
            [ (chandat - surgedat).days <= 4  AND  modifi checkbox のいずれかが
              「胸腔ドレーン追加挿入」「胸膜癒着術」「再手術」]

  ⚠️ 2026-08-19確定: Criterion1の境界は`>5`ではなく`>=5`が正しい(--criterion1_boundary=ge5、既定)。
  実データで両方試した結果、ge5の方が親論文Table2実測値(Group A 63/93=67.7%, Group B 64/106=60.4%)に
  大幅に近づいた(ge5: Group A 62/91=68.1%, Group B 59/101=58.4%, 全体121/192=63.0% ※gt5では
  Group A 50/91=54.9%, Group B 48/101=47.5%, 全体98/192=51.0%と乖離が大きかった)。詳細は
  --criterion1_boundary のhelp文字列参照。

  2026-08-09のwiki記録: この定義でREDCap抽出値(症例数203, 陽性率64.5%,
  侵襲的処置内訳=追加ドレーン2例/癒着術5例/再手術1例)が親試験報告値
  (症例数199, 陽性率63.8%, 同2例/5例/1例)とほぼ完全一致したことを確認済み。
  本スクリプトはその再現・恒久化。

⚠️ REDCapのcheckbox変数(modifi___X)のX番号は、REDCapコードブック
  （`data/attachments/2026/08/29639eda2d81458f.pdf` 27頁、フィールド[modifi]）で
  1=トパーズ設定圧変更 / 2=従来法(壁吸引)に変更 / 3=胸腔ドレーンを追加挿入 /
  4=胸膜癒着術 / 5=再手術 / 9=その他 と確認済み（2026-08-16）。
  既定値は modifi___3 / modifi___4 / modifi___5。

  ⚠️ 「胸腔ドレナージ法変更」フォーム(modifi/chandat等が属するインストゥルメント)は
  コードブック上「変更日が異なるものはページを追加して入力」＝**繰り返し可能な
  インストゥルメント**であることが判明（2026-08-16）。1患者が複数回の変更記録を
  持ちうるため、患者単位に「最初の非欠損値」だけを見ると、1回目の記録では
  該当処置なし(0)でも2回目以降の記録で実施されていた場合を見逃す（またはその逆で
  多重カウントする）恐れがある。本スクリプトはmodifi___3/4/5とchandatについては
  **行レベル(変更記録ごと)で判定し、患者単位にOR集約**する（`compute_invasive_criterion`
  関数）。他の静的フィールド(surgedat/facilities/age2等)は従来通り最初の非欠損値で
  問題ない（患者ごとに1回しか記録されない値のため）。

  ⚠️ 2026-08-17修正: 検証用print文のバグを修正。true_pal自体は元々
  「modifi___3/4/5チェック済み **かつ** chandat-surgedat<=4日」でゲートした
  `_row_criterion2`を正しく使っていたが、実行時に表示される「侵襲的処置の内訳」
  検証メッセージは誤ってゲート前の`_any_<key>`(入院期間中いつでも実施されたか)を
  表示していた。この結果、胸膜癒着術が95例と表示され、親試験報告値(5例)の19倍という
  「Criterion2ラベルが根本的に信頼できない」という誤診断を招いた
  (`diamond_inspect_modifi_criterion2.py`でchantim/chanres/pleumeth/日数分布を
  精査した結果、95件の胸膜癒着術記録は100%が「エアーリークが理由」・100%が使用薬剤
  記録ありで、記録自体は正当。ただし95件中93件が術後5-10日/11日以降に発生しており、
  POD4以内という早期基準を満たすのはわずか2件のみだった＝gatingは元から正しく機能
  していたが、検証printだけが違う変数を見ていた)。
  本修正で、per-key(drain/pleurodesis/reop)のゲート済みフラグを新たに計算・出力し、
  検証printもゲート済み版を表示するよう変更（未ゲート版は参考情報として別途表示・
  出力列`invasive_*_anytime`として保持）。

  ⚠️ 2026-08-17追記: 親論文本文(Journal Pre-proof、ユーザー提供)のENDPOINTS節で、
  一次エンドポイントPALの定義が文言レベルで確定した:
    "PAL was defined as either a persistent air leak after POD 5 or the need
     for invasive procedure, including additional chest tube insertion,
     pleurodesis, or reoperation, due to an air leak until POD 4."
  chanres(変更理由: 1=エアーリーク/2=エアーリーク以外)による絞り込みは、
  診断の結果、192例コホートの侵襲的処置記録が100%「エアーリークが理由」だったため
  実質的な絞り込み効果は無かった（が、将来別コホートで再現する際の安全のため、
  --require_air_leak_reasonで任意に有効化できるようにしてある）。

  REDCapは縦持ち(1症例に複数行、redcap_event_nameでイベントが分かれる)
  構造を想定。subjid単位で集約し、各列の最初の非欠損値を採用する。

  ⚠️ 重要な前提（2026-08-16、実データQCで判明。一度誤った仮説で修正し、後に訂正、
  diamond_table1_external.pyと同一の対応）:
    素朴に subjid でgroupbyすると症例数が想定(申告182例、親試験目標最大199例)から
    大きく乖離する(実データで2354)。最初は「8イベントのうち1イベント（行数が
    全体subjidユニーク総数と完全一致）＝除外例も含む登録/スクリーニングログ」と判断し
    丸ごと除外する修正を入れたが、そのイベントこそがsurgedat(手術日)を含む全登録患者
    共通の基本情報を持つ主要イベントであり、丸ごと除外すると真の研究コホートの
    surgedatまで消えてしまう（実際に判定可能症例数=0という不具合が発生）ことが
    `diamond_inspect_redcap_key_columns.py`で判明した。
    正しい絞り込み条件は、全イベントの情報をsubjid単位に素直に集約した上で、
    **割付群(rand)が記録されている症例のみに限定する**こと
    （rand非欠損=210例、既知の182例・199例と矛盾なく整合する包含関係）。
    （--include_unrandomized で絞り込みを無効化可能、
    --exclude_events で特定イベントを手動除外することも可能＝汎用オプションとして維持）。

使い方（例）:
  python diamond_build_true_pal_labels.py \
      --redcap_csv "/Volumes/Extreme Pro/Red Cap data/ThopazRCT2_raw.csv" \
      --output_dir ~/Documents/DIAMOND/DIAMOND/split_models_external
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--redcap_csv", required=True, help="REDCap生データ(縦持ちexport CSV)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--subjid_col", default="subjid")
    p.add_argument("--event_col", default="redcap_event_name")
    p.add_argument("--surgedat_col", default="surgedat")
    p.add_argument("--alendat_col", default="alendat")
    p.add_argument("--chandat_col", default="chandat")
    p.add_argument("--modifi_drain_col", default="modifi___3", help="胸腔ドレーン追加挿入")
    p.add_argument("--modifi_pleurodesis_col", default="modifi___4", help="胸膜癒着術")
    p.add_argument("--modifi_reop_col", default="modifi___5", help="再手術")
    p.add_argument(
        "--chanres_col", default="chanres",
        help="変更理由(1=エアーリーク/2=エアーリーク以外)。2026-08-17、親論文本文の一次エンドポイント"
             "定義'due to an air leak until POD 4'に対応する列と確認。既定でCriterion2は"
             "chanres=1(エアーリークが理由)の記録のみを対象とする",
    )
    p.add_argument(
        "--allow_non_air_leak_reason", action="store_true",
        help="chanres=2(エアーリーク以外が理由)の侵襲的処置記録もCriterion2に含める(非推奨、"
             "QC目的のみ。2026-08-17時点の192例コホートではchanres=1が100%だったため実害なし)",
    )
    p.add_argument("--rand_col", default="rand", help="割付群列。この列が非欠損の症例のみを真のランダム化コホートとして対象とする")
    p.add_argument(
        "--criterion1_boundary", choices=["gt5", "ge5"], default="ge5",
        help="Criterion1(持続気漏)の判定境界。"
             "2026-08-19、実データで両方試して確定: ge5=dur_alend_days>=5(既定)が正しい境界。"
             "Group A 62/91(68.1%) vs 親論文実測63/93(67.7%)、"
             "Group B 59/101(58.4%) vs 親論文実測64/106(60.4%)、"
             "全体121/192(63.0%) vs 親論文63.8%と、いずれも大幅に一致度が改善したため確定。"
             "gt5(旧既定、'POD5超'の素直な読みだが実データと不一致=51.0%)は比較検証用に残置。",
    )
    p.add_argument("--facilities_col", default="facilities")
    p.add_argument(
        "--include_unrandomized", action="store_true",
        help="randが欠損の症例(スクリーニングのみで終わった患者等)も集約に含める(非推奨、QC目的のみ)",
    )
    p.add_argument(
        "--dsdecod_col", default="dsdecod",
        help="「プロトコル治療完了報告」フォームの完了/中止フィールド(1=完了,2=中止)。"
             "randがちょうど210件と親試験報告値199件より多かったため2026-08-16に追加。"
             "既定でdsdecod=1(完了)のみに絞り込む",
    )
    p.add_argument(
        "--include_discontinued", action="store_true",
        help="プロトコル治療が中止(dsdecod=2)となった症例も集約に含める(非推奨、QC目的のみ)",
    )
    p.add_argument(
        "--exclude_events", default=None,
        help="手動で特定イベントを集約前に除外する(カンマ区切り、汎用オプション。通常は不要)",
    )
    p.add_argument("--encoding", default="utf-8", help="読めない場合は cp932 等を試す")
    return p.parse_args()


def normalize_colname(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def resolve_col(df: pd.DataFrame, wanted: str) -> Optional[str]:
    """列名の完全一致 → 正規化一致 の順で解決する。見つからなければNone。"""
    if wanted in df.columns:
        return wanted
    norm_wanted = normalize_colname(wanted)
    for c in df.columns:
        if normalize_colname(c) == norm_wanted:
            return c
    return None


def normalize_case_id(raw_id: object) -> tuple[Optional[str], Optional[str], str]:
    """"施設番号-症例番号[接尾辞]" 形式のIDを正規化する。

    戻り値: (strict_key, loose_key, cleaned_raw)
      strict_key: "facility-caseno" + 接尾辞アルファベット（あれば）
      loose_key : "facility-caseno" のみ（接尾辞・注記を無視した緩い一致用）
      いずれも先頭ゼロは除去する（"35-01" と "35-1" を同一視するため）。
    パースできない場合は (None, None, cleaned_raw) を返す。
    """
    s = str(raw_id).strip()
    # 全角/半角の括弧内の注記（例:「（リーク無し）」）を除去
    cleaned = re.sub(r"[（(].*?[）)]", "", s)
    # 丸数字などの記号を除去
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


def read_redcap_csv(path: Path, encoding: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding=encoding, low_memory=False)
    except UnicodeDecodeError:
        print(f"[WARN] encoding={encoding} で読めなかったため cp932 を試します")
        return pd.read_csv(path, encoding="cp932", low_memory=False)


def aggregate_redcap_long_to_wide(df: pd.DataFrame, subjid_col: str) -> pd.DataFrame:
    """縦持ち(症例×イベント)を subjid 単位に集約し、各列の最初の非欠損値を採る。"""
    def first_non_null(s: pd.Series):
        s2 = s.dropna()
        return s2.iloc[0] if len(s2) else np.nan

    wide = df.groupby(subjid_col, dropna=False).agg(first_non_null)
    wide = wide.reset_index()
    return wide


def is_checked(val: object) -> bool:
    """REDCap checkbox の"チェック済み"表現ゆれ(1/'1'/'Checked'/True等)を吸収。"""
    if pd.isna(val):
        return False
    s = str(val).strip().lower()
    return s in {"1", "1.0", "true", "checked", "yes", "y"}


def compute_invasive_criterion(
    df_long: pd.DataFrame,
    subjid_col: str,
    surgedat_by_patient: dict,
    chandat_col: Optional[str],
    modifi_cols: dict,
    chanres_col: Optional[str] = None,
    require_air_leak_reason: bool = True,
) -> pd.DataFrame:
    """繰り返し可能な「胸腔ドレナージ法変更」フォームを行レベルで正しく評価する。

    各行(=1回の変更記録)ごとに (1) その行のchandat - 患者のsurgedat <= 4日か、
    (2) その行でmodifi___3/4/5のいずれかがチェックされているか、
    (3) require_air_leak_reason=Trueかつchanres_colが指定されている場合、
        その行のchanres(変更理由)がエアーリーク(=1)か、
    を判定し、「該当する行が1つでもあれば陽性」という形で患者単位にOR集約する。
    (3)は2026-08-17、親論文本文の一次エンドポイント定義
    "due to an air leak until POD 4" に対応する形で追加。
    単純に患者ごとの「最初の非欠損値」を見る集約では、複数回の変更記録のうち
    どれか1回だけで処置が行われたケースを正しく拾えない（見逃す、または
    無関係の記録を拾って誤カウントする）ため、この専用関数で対応する。

    戻り値: subjid列 + 各modifi_colsキーごとの "_any_<key>" 列（bool、いつでも実施） +
      "_criterion2_<key>" 列（bool、<=4日以内 かつ [有効なら]エアーリーク理由） +
      "_row_criterion2" 列（bool、いずれかのkeyで上記が成立したか）
      を持つ、subjid単位に集約したDataFrame。
    """
    df = df_long.copy()
    df["_surgedat"] = df[subjid_col].map(surgedat_by_patient)
    if chandat_col and chandat_col in df.columns:
        df["_chandat_dt"] = pd.to_datetime(df[chandat_col], errors="coerce")
    else:
        df["_chandat_dt"] = pd.NaT
    df["_dur_chandat_days"] = (df["_chandat_dt"] - df["_surgedat"]).dt.days

    if require_air_leak_reason and chanres_col and chanres_col in df.columns:
        chanres_num = pd.to_numeric(df[chanres_col], errors="coerce")
        air_leak_reason = chanres_num == 1
    else:
        # chanres列が無い、またはrequire_air_leak_reason=Falseの場合は理由を問わない(常にTrue)
        air_leak_reason = pd.Series(True, index=df.index)

    row_invasive_any = pd.Series(False, index=df.index)
    flag_cols = {}
    for key, col in modifi_cols.items():
        if col and col in df.columns:
            flag = df[col].apply(is_checked)
        else:
            flag = pd.Series(False, index=df.index)
        flag_cols[f"_any_{key}"] = flag
        row_invasive_any = row_invasive_any | flag

    result = pd.DataFrame({subjid_col: df[subjid_col]})
    gated = (df["_dur_chandat_days"] <= 4) & air_leak_reason
    for name, flag in flag_cols.items():
        result[name] = flag
        # ⚠️ 2026-08-17追加: 各成分(drain/pleurodesis/reop)ごとの「<=4日以内 かつ
        # エアーリークが理由」ゲート済みフラグ。従来はtrue_pal計算に使う結合版
        # (_row_criterion2)しか無く、検証用print文が誤って未ゲートの_any_<key>
        # (いつでも可)をそのまま「侵襲的処置の内訳」として表示していたため、
        # 親試験報告値(2/5/1例)との比較が的外れになっていた
        # （胸膜癒着術が実データで95例と算出され、あたかもtrue_palラベルが19倍過大な
        # Criterion2を含むかのように見えたが、true_pal自体は元々日数ゲート済みの
        # criterion2を使っており影響を受けていなかった。ただし検証がずっとできて
        # いなかった点は実害。診断の結果、95件は100%エアーリークが理由・100%薬剤記録
        # ありで記録自体は正当、95件中93件がPOD5以降=期間外だっただけと判明）。
        result[f"_criterion2_{name[len('_any_'):]}"] = flag & gated
    result["_row_criterion2"] = row_invasive_any & gated

    agg = result.groupby(subjid_col, dropna=False).any().reset_index()
    return agg


def main() -> int:
    args = parse_args()
    redcap_path = Path(args.redcap_csv).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    df_raw = read_redcap_csv(redcap_path, args.encoding)
    print(f"[INFO] REDCap生データ読込: {len(df_raw)}行, {len(df_raw.columns)}列")

    subjid_col = resolve_col(df_raw, args.subjid_col)
    if subjid_col is None:
        raise RuntimeError(
            f"subjid列が見つかりません（--subjid_col='{args.subjid_col}'）。"
            f"実際の列名候補: {[c for c in df_raw.columns if 'subj' in normalize_colname(c)]}"
        )

    n_unique_all = df_raw[subjid_col].dropna().astype(str).str.strip().nunique()
    print(f"[INFO] 全イベント合算のsubjidユニーク数(スクリーニングのみで終わった患者を含む): {n_unique_all}")

    df_filtered = df_raw
    if args.exclude_events:
        event_col = resolve_col(df_raw, args.event_col)
        excluded = [e.strip() for e in args.exclude_events.split(",") if e.strip()]
        if event_col:
            df_filtered = df_raw[~df_raw[event_col].isin(excluded)]
            print(f"[INFO] 手動指定によりイベントを除外: {len(excluded)}件 -> 除外後行数={len(df_filtered)}")
        else:
            print("[WARN] --exclude_eventsが指定されましたがevent列が見つからず適用できません")

    # 全イベントの情報をsubjid単位に集約する（「最初の非欠損値」採用のため、
    # ある列がどのイベントにあっても正しく1患者1行に統合される。surgedat等の
    # 基本情報は「登録ログ」的な主要イベントにのみ存在することが判明したため、
    # イベントを除外せず全体を集約するのが正しい）。
    wide = aggregate_redcap_long_to_wide(df_filtered, subjid_col)
    print(f"[INFO] subjid単位に集約(全イベント): {len(wide)}症例")

    # 真のランダム化コホートに限定: rand(割付群)が記録されている症例のみを対象とする。
    rand_col_for_filter = resolve_col(wide, args.rand_col)
    if rand_col_for_filter is not None and not args.include_unrandomized:
        n_before = len(wide)
        wide = wide[wide[rand_col_for_filter].notna()].reset_index(drop=True)
        print(
            f"[INFO] 割付群(rand)が記録されている症例のみに限定(=真のランダム化コホート): "
            f"{n_before} -> {len(wide)}"
        )
    elif rand_col_for_filter is None:
        print(
            f"[WARN] 割付群列（--rand_col='{args.rand_col}'）が見つからず、"
            "ランダム化コホートへの絞り込みができません。全症例を対象とします。"
        )
    else:
        print("[INFO] --include_unrandomizedが指定されたため、rand非欠損によるコホート限定はスキップします")

    # 2026-08-16追加: rand非欠損=210件は親試験の報告値(199例)より多い。コードブック確認の結果、
    # 「プロトコル治療完了報告」フォーム(dsdecod: 1=完了/2=中止)がランダム化された210例全員に
    # 存在することが判明。無作為化後の中止例(有害事象・同意撤回等)を除いた「完了」例のみが
    # 最終的な解析対象コホートに対応すると考えられるため、既定でdsdecod=1に絞る。
    dsdecod_col = resolve_col(wide, args.dsdecod_col)
    if dsdecod_col is not None and not args.include_discontinued:
        n_before = len(wide)
        dsdecod_num = pd.to_numeric(wide[dsdecod_col], errors="coerce")
        n_discontinued = int((dsdecod_num == 2).sum())
        n_missing_dsdecod = int(dsdecod_num.isna().sum())
        wide = wide[dsdecod_num == 1].reset_index(drop=True)
        print(
            f"[INFO] プロトコル治療完了(dsdecod=1)の症例のみに限定: {n_before} -> {len(wide)} "
            f"(中止={n_discontinued}件, dsdecod欠損={n_missing_dsdecod}件)"
        )
    elif dsdecod_col is None:
        print(
            f"[WARN] プロトコル治療完了報告列（--dsdecod_col='{args.dsdecod_col}'）が見つからず、"
            "完了/中止による絞り込みができません。ランダム化コホート全体を対象とします。"
        )
    else:
        print("[INFO] --include_discontinuedが指定されたため、dsdecodによるコホート限定はスキップします")

    col_map = {
        "surgedat": resolve_col(wide, args.surgedat_col),
        "alendat": resolve_col(wide, args.alendat_col),
        "chandat": resolve_col(wide, args.chandat_col),
        "modifi_drain": resolve_col(wide, args.modifi_drain_col),
        "modifi_pleurodesis": resolve_col(wide, args.modifi_pleurodesis_col),
        "modifi_reop": resolve_col(wide, args.modifi_reop_col),
        "chanres": resolve_col(wide, args.chanres_col),
        "rand": resolve_col(wide, args.rand_col),
        "facilities": resolve_col(wide, args.facilities_col),
    }
    missing = [k for k, v in col_map.items() if v is None and k in ("surgedat", "alendat", "chandat")]
    if missing:
        raise RuntimeError(
            f"必須列が見つかりません: {missing}。"
            f"実際の列名を --surgedat_col 等で指定してください。"
            f"列名一覧(先頭50件): {wide.columns.tolist()[:50]}"
        )
    for k, v in col_map.items():
        status = "OK" if v is not None else "見つからず(該当なしとして扱う)"
        print(f"[INFO] 列解決: {k} -> {v} [{status}]")

    surgedat = pd.to_datetime(wide[col_map["surgedat"]], errors="coerce")
    alendat = pd.to_datetime(wide[col_map["alendat"]], errors="coerce")

    dur_alend_days = (alendat - surgedat).dt.days

    # ⚠️ modifi___3/4/5・chandatは繰り返し可能な「胸腔ドレナージ法変更」フォームに属するため、
    # wideの「最初の非欠損値」ではなく、生データを行レベルで評価してから患者単位にOR集約する
    # (compute_invasive_criterion、詳細はモジュールdocstring参照)。
    surgedat_by_patient = dict(zip(wide[subjid_col], surgedat))
    modifi_cols = {
        "drain": col_map["modifi_drain"],
        "pleurodesis": col_map["modifi_pleurodesis"],
        "reop": col_map["modifi_reop"],
    }
    invasive_agg = compute_invasive_criterion(
        df_filtered, subjid_col, surgedat_by_patient, col_map["chandat"], modifi_cols,
        chanres_col=col_map["chanres"],
        require_air_leak_reason=not args.allow_non_air_leak_reason,
    )
    invasive_agg = invasive_agg.set_index(subjid_col)
    invasive_agg = invasive_agg.reindex(wide[subjid_col]).fillna(False)

    # 未ゲート版(入院期間中いつでも実施されたか。臨床的な参考情報として出力に残す)
    drain_flag = invasive_agg["_any_drain"].reset_index(drop=True)
    pleurodesis_flag = invasive_agg["_any_pleurodesis"].reset_index(drop=True)
    reop_flag = invasive_agg["_any_reop"].reset_index(drop=True)
    invasive_any = drain_flag | pleurodesis_flag | reop_flag
    # ゲート済み版(術後4日目まで、かつエアーリーク理由の処置のみ)。
    # true_pal・親試験報告値(2/5/1例)との比較には必ずこちらを使う。
    drain_criterion2 = invasive_agg["_criterion2_drain"].reset_index(drop=True)
    pleurodesis_criterion2 = invasive_agg["_criterion2_pleurodesis"].reset_index(drop=True)
    reop_criterion2 = invasive_agg["_criterion2_reop"].reset_index(drop=True)
    criterion2 = invasive_agg["_row_criterion2"].reset_index(drop=True)

    if args.criterion1_boundary == "ge5":
        criterion1 = dur_alend_days >= 5
        print("[INFO] Criterion1境界: dur_alend_days>=5 を使用(--criterion1_boundary=ge5、2026-08-19確定・既定)")
    else:
        criterion1 = dur_alend_days > 5
        print("[INFO] Criterion1境界: dur_alend_days>5 を使用(--criterion1_boundary=gt5、旧既定・実データと不一致のため比較検証用のみ)")

    # surgedatが無い症例は判定不能。alendat・chandatどちらも無い(=侵襲的処置もなし)場合はcriterion2側も判定不能。
    undetermined = surgedat.isna() | (alendat.isna() & ~invasive_any)

    true_pal = (criterion1.fillna(False) | criterion2.fillna(False)).astype("Int64")
    true_pal[undetermined] = pd.NA

    out = pd.DataFrame({
        "subjid": wide[subjid_col],
        "surgedat": surgedat,
        "alendat": alendat,
        "dur_alend_days": dur_alend_days,
        # chandat/dur_chandat_daysは複数の変更記録に跨りうるため単一値としては持たない。
        # criterion2_early_intervention(行レベル評価の患者単位OR集約結果)を直接参照すること。
        # 未ゲート版(入院期間中いつでも実施されたか。true_palの計算には使わない、参考列)
        "invasive_drain_added_anytime": drain_flag,
        "invasive_pleurodesis_anytime": pleurodesis_flag,
        "invasive_reop_anytime": reop_flag,
        "invasive_any_anytime": invasive_any,
        # ゲート済み版(術後4日目まで。true_pal計算に使用、親試験2/5/1例との比較対象はこちら)
        "invasive_drain_added_pod4": drain_criterion2,
        "invasive_pleurodesis_pod4": pleurodesis_criterion2,
        "invasive_reop_pod4": reop_criterion2,
        "criterion1_over5days": criterion1,
        "criterion2_early_intervention": criterion2,
        "true_pal": true_pal,
    })
    if col_map["rand"]:
        out["rand"] = wide[col_map["rand"]]
    if col_map["facilities"]:
        out["facilities"] = wide[col_map["facilities"]]

    strict_keys, loose_keys, cleaned_ids = [], [], []
    for v in out["subjid"]:
        sk, lk, cl = normalize_case_id(v)
        strict_keys.append(sk)
        loose_keys.append(lk)
        cleaned_ids.append(cl)
    out["case_id_strict_key"] = strict_keys
    out["case_id_loose_key"] = loose_keys
    out["case_id_cleaned"] = cleaned_ids

    unparsed = out[out["case_id_strict_key"].isna()]
    if len(unparsed):
        print(f"[WARN] subjidをID正規化できなかった症例が{len(unparsed)}件あります（case_id_cleaned列を手動確認）")

    out_path = output_dir / "true_pal_labels.csv"
    out.to_csv(out_path, index=False)

    n_total = len(out)
    n_determined = int(out["true_pal"].notna().sum())
    n_positive = int((out["true_pal"] == 1).sum())
    pos_rate = n_positive / n_determined if n_determined else float("nan")
    print(f"[OK] 保存: {out_path}")
    print(f"[結果] 総症例数={n_total}, 判定可能={n_determined}, 陽性(true_pal=1)={n_positive} ({pos_rate:.1%})")
    print(
        "[検証・修正版] 侵襲的処置の内訳(術後4日目までにゲート済み。"
        "親試験報告値=追加ドレーン2例/癒着術5例/再手術1例と比較してください): "
        f"追加ドレーン挿入={int(drain_criterion2.sum())}例, "
        f"胸膜癒着術={int(pleurodesis_criterion2.sum())}例, "
        f"再手術={int(reop_criterion2.sum())}例"
    )
    print(
        "[参考・未ゲート版] 入院期間中いつでも実施された侵襲的処置(true_pal計算には使わない、"
        "臨床的な参考情報。2026-08-16に「80例」等として誤って親試験値と比較していたのはこちら側): "
        f"追加ドレーン挿入={int(drain_flag.sum())}例, "
        f"胸膜癒着術={int(pleurodesis_flag.sum())}例, "
        f"再手術={int(reop_flag.sum())}例"
    )
    print(
        "[NEXT] 上記の「検証・修正版」(ゲート済み)の症例数が2026-08-09時点の記録"
        "（症例数203, 陽性率64.5%, 2/5/1例）と大きくずれる場合は、"
        "--modifi_drain_col 等の列名割り当てを見直してください。"
    )

    # 2026-08-19追加: 親論文Table2(実測値、ユーザー提供)との突合用に、rand(割付群)別の
    # PAL陽性率も出す。親論文Table2: Group A(n=93) 63例(67.7%)。
    # 本コホートはGroup A/Bともn自体が親論文と若干ずれる(既知の192 vs 199問題)ため、
    # 完全一致はしない前提で、境界を変えた時にどれだけ近づくかの比較に使うこと。
    if "rand" in out.columns:
        rand_num = pd.to_numeric(out["rand"], errors="coerce")
        for rand_val, group_label, paper_ref in [(1, "Group A (-8 cmH2O)", "親論文Table2実測値: 63/93 (67.7%)"), (2, "Group B (-15 cmH2O)", "親論文Table2実測値: 64/106 (60.4%)")]:
            sub = out[rand_num == rand_val]
            sub_determined = int(sub["true_pal"].notna().sum())
            sub_positive = int((sub["true_pal"] == 1).sum())
            sub_rate = sub_positive / sub_determined if sub_determined else float("nan")
            print(
                f"[結果・群別] {group_label}: n={len(sub)}, 判定可能={sub_determined}, "
                f"true_pal陽性={sub_positive} ({sub_rate:.1%})  <- {paper_ref}"
            )
    else:
        print("[WARN] rand列が無いため群別のPAL陽性率は出せません")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

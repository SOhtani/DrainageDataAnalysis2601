#!/usr/bin/env bash
# diamond_table1_dev_pipeline.sh
#
# 目的:
#   Excelを手動で開いてCSVエクスポートする作業を自動化する。
#   xlsx2csv（軽量・ストリーミング方式のxlsx→CSV変換ツール。openpyxlのように
#   ワークブック全体のオブジェクトモデルを組み立てないため、肥大化した
#   xlsxでも高速な傾向がある）を使って「本当の除外なし921例」「臨床因子追加」
#   の2シートをCSVに変換し、そのまま diamond_table1_dev.py まで実行する
#   一気通貫パイプライン。
#
# 前提: python3 + pip が使えること（xlsx2csvは初回のみ自動インストール）。
#   diamond_table1_dev.py が同じディレクトリ(~/Documents/DIAMOND/DIAMOND/)
#   に保存済みであること（thopaz_setup_analysis_scripts_6.sh等で導入済みのはず）。
#
# 使い方:
#   bash diamond_table1_dev_pipeline.sh
#
# うまくいかない場合（xlsx2csvでも遅い/失敗する等）は、従来通りExcelの
# GUIで「名前を付けて保存 > CSV UTF-8」を使う手動手順に戻ってください。

set -euo pipefail

DIAMOND_DIR="$HOME/Documents/DIAMOND/DIAMOND"
CASE_FILE="$HOME/Documents/DIAMOND/DIAMOND case file.xlsx"
OUTPUT_DIR="$DIAMOND_DIR/split_models_external"
BASE_CSV="$DIAMOND_DIR/table1_source_base.csv"
FACTOR_CSV="$DIAMOND_DIR/table1_source_factor.csv"
BASE_SHEET="本当の除外なし921例"
FACTOR_SHEET="臨床因子追加"

if [ ! -f "$CASE_FILE" ]; then
  echo "[ERROR] 見つかりません: $CASE_FILE" >&2
  exit 1
fi

if ! python3 -c "import xlsx2csv" >/dev/null 2>&1; then
  echo "[INFO] xlsx2csv が未インストールのためインストールします"
  python3 -m pip install --user xlsx2csv
fi

echo "[INFO] '$BASE_SHEET' を CSV へ変換中..."
python3 -m xlsx2csv -n "$BASE_SHEET" "$CASE_FILE" "$BASE_CSV"
echo "[OK] -> $BASE_CSV"

echo "[INFO] '$FACTOR_SHEET' を CSV へ変換中..."
python3 -m xlsx2csv -n "$FACTOR_SHEET" "$CASE_FILE" "$FACTOR_CSV"
echo "[OK] -> $FACTOR_CSV"

echo "[INFO] diamond_table1_dev.py を実行します"
python3 "$DIAMOND_DIR/diamond_table1_dev.py" \
    --base_csv "$BASE_CSV" \
    --factor_csv "$FACTOR_CSV" \
    --output_dir "$OUTPUT_DIR"

#!/bin/bash
set -e

# thopaz_setup_diamond_pvalue_threshold_diag.sh
#
# 目的:
#   diamond_table1_external.py / diamond_diagnose_label_vs_scored_cohort.py /
#   diamond_score_external_predictions.py の3本を ~/DIAMOND-local/ に配置する。
#
# 2026-09-28 変更点（重複解消、計算ロジックは無変更）:
#   このスクリプトは元々、上記3本のPythonソース全文をheredocでそのまま埋め込み、
#   ~/DIAMOND-local/ に書き出す形だった。埋め込まれていた3本は、現在リポジトリに
#   正本として存在する下記3ファイルと完全に同一内容であることを確認済み
#   （diffで検証、ヘッダ行のみの差分）:
#     - diamond/table1/diamond_table1_external.py
#     - diamond/diagnostics/diamond_diagnose_label_vs_scored_cohort.py
#     - diamond/validation/diamond_score_external_predictions.py
#   二重管理（正本とheredoc埋め込みコピーがズレて発散するリスク）を避けるため、
#   heredocでの埋め込みをやめ、正本からのコピーに変更した。
#   各スクリプトの中身・計算ロジックは一切変更していない（コピー元を変えただけ）。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

mkdir -p ~/DIAMOND-local

cp "$REPO_ROOT/diamond/table1/diamond_table1_external.py" ~/DIAMOND-local/diamond_table1_external.py
echo "[OK] diamond_table1_external.py を書き込みました"

cp "$REPO_ROOT/diamond/diagnostics/diamond_diagnose_label_vs_scored_cohort.py" ~/DIAMOND-local/diamond_diagnose_label_vs_scored_cohort.py
echo "[OK] diamond_diagnose_label_vs_scored_cohort.py を書き込みました"

cp "$REPO_ROOT/diamond/validation/diamond_score_external_predictions.py" ~/DIAMOND-local/diamond_score_external_predictions.py
echo "[OK] diamond_score_external_predictions.py を書き込みました"

python3 -c "import py_compile; [py_compile.compile(f\"$HOME/DIAMOND-local/{n}\", doraise=True) for n in [\"diamond_table1_external.py\",\"diamond_diagnose_label_vs_scored_cohort.py\",\"diamond_score_external_predictions.py\"]]; print(\"[OK] 3ファイルとも構文チェックOK\")"

# DrainageDataAnalysis2601

Thopaz（メデラ社デジタル胸腔ドレナージ機器）由来データを用いた、術後遷延性肺瘻（PAL, prolonged air leak）予測研究（DIAMONDプロジェクト）の解析コード群。

## このリポジトリに含まれるもの / 含まれないもの

**含まれる**: 「真のPALラベル構築→モデル外部検証→ROC/Table1作成→REDCapデータ診断」までの**解析・検証パイプライン**（Python/shell）。主に順天堂多施設RCT-2データを用いた外部検証（本研究の"C系統"）と、院内Thopaz実機ログ（STOP_AIR）の処理ツール一式。

**含まれない**:
- 患者データそのもの（CSV/REDCapエクスポート/DIAMOND case file等）。院内環境でのみ保持。
- 学習済みモデルのチェックポイント（サイズ大のため非搭載）。
- オリジナルのモデル学習コード（1D-CNN/LSTM/Transformer, 2023年開発版）。こちらは別リポジトリ（開発委託先エンジニアによる`chest_drain`）が対象で、本リポジトリはその出力を評価・外部検証するための**下流の解析コード**という位置づけ。

## ディレクトリ構成

```
diamond/
  labels/       真のPALラベル構築（REDCap生データ→ラベルCSV）
  validation/   凍結モデルの外部検証・スコアリング
  table1/       患者背景表（Table1）作成
  roc/          ROC曲線・AUROC比較の可視化
  diagnostics/  データ診断・ラベル整合性チェック・除外基準の検証
  cohort/       コホート統計比較・交差検証・データ量スイープ
  PLAN_C_RECONCILIATION.md   C系統(外部検証)AUC再計算の経緯・計画メモ
stop_air/       院内Thopaz実機ログ(STOP_AIR)からのケース抽出・DIAMOND凍結モデルでのスコアリング
setup/          解析環境セットアップ・p値閾値診断スクリプト
```

## 実行環境

Python 3.11 (pandas, numpy, scikit-learn, matplotlib想定)。入力データパスは各スクリプト内で明示的に指定する形式（院内サーバー上の実データパスに書き換えて実行する前提）。

## スクリプト索引

このリポジトリの大半は、特定の疑問（「陽性率が親論文と合わない」「AUCが高すぎる」等）が
出るたびにその場で書かれた調査用スクリプトであり、各ファイルの冒頭docstringに
「何を・なぜ調べたか」が経緯付きで記録されている。以下は各スクリプトの役割を
1〜2行に要約した索引（詳細・実行例は各ファイルのdocstringを参照）。

**重複調査について（2026-09-28）**: 全32本のPythonスクリプト＋shellスクリプトの
docstringを確認したが、同一ロジックの`_v1`/`_v2`のような並行バージョンは存在しない
（ファイル名が似ているスクリプト同士は、いずれもdocstring内で「〇〇とは何が違うか」を
明記した上で意図的に役割分担されている）。唯一見つかった重複は
`setup/thopaz_setup_diamond_pvalue_threshold_diag.sh`が3本のPythonスクリプトの
全文を（diamond/配下の正本と完全同一の内容のまま）heredocで二重に埋め込んでいた点で、
これは正本からコピーする方式に統合済み（計算ロジックの変更なし、詳細は同ファイル冒頭コメント参照）。

### diamond/labels/ — 真のPALラベル構築

| スクリプト | 役割 |
|---|---|
| `diamond_build_true_pal_labels.py` | 順天堂多施設Thopaz RCT-2のREDCap生データから、親試験定義（持続気漏≥5日 OR 術後4日目までの気漏起因の侵襲的処置）に準拠した「真のPALラベル」を1症例1行で算出する。C系統（外部検証）の全パイプラインが依存する起点スクリプト。 |

### diamond/validation/ — 凍結モデルの外部検証・スコアリング

| スクリプト | 役割 |
|---|---|
| `diamond_freeze_dev_model.py` | 開発コホート（DIAMOND単施設）全体で1D-CNNを1本学習し、重み・scalerを「凍結モデル」として保存する（交差検証ではなく外部検証用の固定パラメータ作成）。 |
| `diamond_external_validate.py` | 凍結モデルを順天堂Thopazエクセルデータに適用し、ルールベースラベル・真のPALラベル双方でAUROC等を算出する。 |
| `diamond_score_external_predictions.py` | `diamond_build_true_pal_labels.py`の出力と、9:1分割モデル(full/12h/18h/24h)の外部予測を症例IDで突合し、外部検証AUC・95%CI・混同行列・train/internal/external統合表を算出する。C系統の主解析スクリプト。 |
| `diamond_score_external_by_group.py` | `diamond_score_external_predictions.py`が保存したマージ済みCSVを使い、外部検証結果を割付群（Group A/B）別に分けて集計する派生・下流スクリプト。 |

### diamond/table1/ — 患者背景表（Table1）作成

| スクリプト | 役割 |
|---|---|
| `diamond_table1_dev.py` | 単施設(DIAMOND)コホート全体（n=921、背景データ完備例）のTable1を作成する。 |
| `diamond_table1_dev_split.py` | `diamond_table1_dev.py`の母集団を、9:1分割モデル学習時のinternal test(10%)とtrain(90%)に分けてTable1を作成する（`diamond_table1_dev.py`の分割版、重複ではなく別集計単位）。 |
| `diamond_table1_dev_pipeline.sh` | Excelの2シートをxlsx2csvでCSV変換してから`diamond_table1_dev.py`を実行する一気通貫パイプライン（手動Excelエクスポート作業の自動化のみ、集計ロジックは持たない）。 |
| `diamond_table1_external.py` | 外部(順天堂RCT-2)コホートのREDCap生データからTable1を作成する（`--labels_csv`指定でtrue_pal層別も追加）。単施設側の`diamond_table1_dev.py`に対応する外部コホート版。 |

### diamond/roc/ — ROC曲線・AUROC比較の可視化

| スクリプト | 役割 |
|---|---|
| `diamond_plot_roc_by_cohort.py` | internal/externalそれぞれについて、full/12h/18h/24hのROC曲線を1枚にまとめて重ね書きする（コホート軸で2枚）。 |
| `diamond_plot_roc_internal_external.py` | horizonごとに1枚、internalとexternalのROC曲線を重ね書きする（horizon軸で4枚）。`diamond_plot_roc_by_cohort.py`とは軸の切り方が逆の対（重複ではなく意図的な相補ペア、両ファイルのdocstringに明記）。 |
| `diamond_roc_pod1_flow_devcohort.py` | 単施設コホートで、POD1時点の平均流量というAIモデルを介さない単一スコアのROC/AUCを算出する（ルールベース参考線・開発コホート版）。 |
| `diamond_roc_alflow_external.py` | 外部コホートで、REDCap `alflow`（POD1流量に相当する1点測定）のROC/AUCを算出する。`diamond_roc_pod1_flow_devcohort.py`の外部コホート対になる版（概念・window定義を揃えてある）。 |

### diamond/diagnostics/ — データ診断・ラベル整合性チェック・除外基準の検証

各スクリプトは特定の疑問1件ずつに対応する一回性の調査スクリプト（時系列で読むと調査の経緯が追える）。

| スクリプト | 役割 |
|---|---|
| `diamond_inspect_dev_background_candidates.py` | 単施設Table1の元データがどのExcelファイル/シートに入っているかを、シート名・列名・行数のみで一覧確認する（最初の探索ステップ）。 |
| `diamond_inspect_dev_candidate_values.py` | 上記で絞り込んだ候補シートについて、値レベル（欠測数・度数・平均等の集計値のみ、患者レベルの生値は出さない）で内容を確認する。 |
| `diamond_inspect_redcap_subjid.py` | 外部Table1で対象症例数が想定(182〜199例)より大幅に多い(2354)問題の原因切り分け。subjid表記ゆれの疑いを集計値のみで検証する。 |
| `diamond_inspect_redcap_event_overlap.py` | 上記で判明した「最大イベント=より大きな母集団」仮説を、イベント別subjidユニーク数の重なりで検証する。 |
| `diamond_inspect_redcap_key_columns.py` | 外部ラベル構築が「判定可能=0」になった原因切り分け。主要列（surgedat等）がどのイベントに存在し、日付パースが成功しているかを確認する。 |
| `diamond_inspect_modifi_criterion2.py` | Criterion2（早期侵襲的処置）の胸膜癒着術件数が親論文報告値の16倍という乖離の原因を、REDCapコードブックの`chanres`/`chantim`列と照合して診断する。 |
| `diamond_inspect_criterion1_alendat.py` | true_pal陽性率が親試験報告値より低い件について、Criterion1（持続気漏）側のalendat欠測・日数分布・感度分析を診断する。 |
| `diamond_diagnose_label_vs_scored_cohort.py` | ラベルCSV(Table1/陽性率の母数)とAI予測スコアリング後のコホート(AUC評価の母数)が同一集団・同一ラベル定義かを突合確認する。 |
| `diamond_leak_check.py` | Astra外部レビュー提案の情報リーク検査。horizon以降のデータを変えてもhorizon内の入力・予測スコアが不変であることを合成データで検証する。 |
| `diamond_check_restriction_of_range.py` | 開発コホートと外部コホートでPOD1流量ルールのAUROCが大きく異なる現象が、外部コホートの登録基準によるrestriction of rangeで説明できるかを検証する。 |
| `diamond_check_persistent_low_leak_exclusion.py` | 単施設コホートのPOD1ルールAUCが高すぎる件について、「陰性群の大半がそもそも気漏を経験していない症例」という仮説を、除外前後のAUC比較で検証する。 |

### diamond/cohort/ — コホート統計比較・交差検証・データ量スイープ

| スクリプト | 役割 |
|---|---|
| `diamond_step1_duration_feasibility.py` | 複合アウトカム(PAL)の前段階として、ドレーン留置期間という単純な連続変数がair leakトレンドの要約統計量から予測できるか、軽量な相関チェックで確認する（Step1実行可能性検証）。 |
| `diamond_dev_cohort_cv.py` | 開発コホートの内部検証を、1回きりの9:1分割ではなくstratified k分割交差検証で行う（`diamond_freeze_dev_model.py`と同一の前処理ロジックを使用、より頑健な内部検証値を得るための別スクリプト）。 |
| `diamond_data_amount_sweep.py` | 学習データ量とAUCの関係を、単施設held-out testと外部コホート双方について新規に測定する（`diamond_freeze_dev_model.py`と同一パイプラインを土台にした新規実験、過去の学習曲線の再現ではない点に注意）。 |
| `diamond_compare_cohorts_stats.py` | 単施設コホートと外部コホートを、Table1出力（`--dump_patient_csv`）を使って患者単位で統計的検定（Mann-Whitney U／カイ二乗・Fisher）により直接比較する。 |
| `diamond_simulate_return_to_room_cohort.py` | 新規前向き観察研究の組み入れ基準（帰室時リーク20-2000 mL/min）に合わせた学習コホートを、DIAMOND生データから事後的に再構成した場合の規模・陽性率を試算する。 |

### stop_air/ — 院内Thopaz実機ログ(STOP_AIR)の処理ツール

一連のパイプライン（上から順に実行）。

| スクリプト | 役割 |
|---|---|
| `thop_log_to_csv.py` | Thopaz実機の生ログ(.log、独自バイナリ形式)をthopeasyアプリ無しでCSVに変換する。フォーマットのリバースエンジニアリング結果そのもの（他スクリプトから共有される基盤ロジック）。 |
| `thop_identify_target.py` | 同一デバイスの使い回しにより1ログに複数患者のセッションが混在する問題に対し、フォルダ名日付とログStart Dateの突合で対象患者(TARGET)のログを識別する。 |
| `thop_extract_target.py` | TARGET判定されたログのみをCSV化し、症例ごとのマニフェスト(target_manifest.csv)を出力する。 |
| `thop_diagnose_zero_rows.py` | TARGET症例のうちデータ行数0件だったケースについて、同フォルダ内の他ログにrescue候補が無いか再スキャンする診断スクリプト。 |
| `thop_inspect_truncation.py` | データ行数0件の原因が、パーサーの早期打ち切りバグによるものでないかを検証する診断スクリプト（`thop_diagnose_zero_rows.py`とは異なる仮説を検証）。 |
| `thop_case_summary.py` | データ行数>0の症例について、同一フォルダの複数セグメントを結合し、ドレナージ期間ベースのproxy PALラベル（DIAMOND B1/B2と同定義）と基本統計量を1症例1行で算出する。 |
| `thop_score_diamond_model.py` | STOP_AIR症例をDIAMOND凍結モデル(24h)でスコアリングし、proxyラベルに対する暫定AUROCを算出する（`diamond_external_validate.py`と同一の前処理を踏襲）。 |

### setup/ — 環境セットアップ

| スクリプト | 役割 |
|---|---|
| `thopaz_setup_diamond_pvalue_threshold_diag.sh` | `diamond_table1_external.py`・`diamond_diagnose_label_vs_scored_cohort.py`・`diamond_score_external_predictions.py`の3本を`~/DIAMOND-local/`に配置するデプロイスクリプト。2026-09-28以降は各ファイルの正本（diamond/配下）からコピーする方式（以前はheredocでソース全文を二重に埋め込んでいた重複を解消）。 |

## 関連

- 背景・経緯: 院内wiki「Thopaz/デジタル胸腔ドレナージ × PAL予測」B/C系統
- オリジナルのモデル学習コード（1D-CNN/LSTM/Transformer）は別途、開発委託先エンジニアによる非公開リポジトリで管理。

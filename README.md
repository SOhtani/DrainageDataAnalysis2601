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

## 関連

- 背景・経緯: 院内wiki「Thopaz/デジタル胸腔ドレナージ × PAL予測」B/C系統
- オリジナルのモデル学習コード（1D-CNN/LSTM/Transformer）は別途、開発委託先エンジニアによる非公開リポジトリで管理。

# EyeTrackingSystem

Tobiiアイトラッカーを用いて、画像刺激提示中の視線データを取得・保存する実験用システムです。

---

# 概要

本システムは以下を実行します：

- Tobii Eye Trackerの接続
- 画像刺激の順番提示（実験刺激）
- 視線データ（視線座標・瞳孔径）のリアルタイム取得
- CSV形式でのデータ保存

---

# 必要環境

- Windows 10 / 11（推奨）, Linux (Ubuntu 24.04.4 LTS)
- Python 3.10
  - `tobii-research` がPython 3.11 以降に対応していないため、Python 3.11 や 3.12 などの環境では `pip install` 時にエラーになります。必ず 3.10 の環境を用意してください。
- Tobii Eye Tracker（対応デバイス）
- OpenCV
- screeninfo
- tobii-research SDK

---

# セットアップ

```bash
# Python 3.10 を指定して仮想環境を作成
python3.10 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 使い方

1. 画像ファイルを `images/` フォルダに配置する
2. main.pyで画像ファイルの表示する時間や順番を決定
3. ディスプレイに接続
4. tobiiに接続
5. main.pyを実行する

```bash
python main.py
```

本プログラムではモニターに接続することを前提としており、映像はモニターに表示される。
デフォルトでは視線計測のデータは`date/gaze_pupil_data.csv` として出力します。

## プロジェクト構成

```
EyeTrackingSystem/
├── images/         # 入力画像フォルダ（任意）
├── config.json     # imageの画像ファイルとmain.pyでの表示名の紐づけスクリプト
├── main.py         # 動画生成スクリプト
├── display.py      # モニターに表示するスクリプト
├── tobii.py         # tobii接続管理スクリプト
└── data/           # 生成された動画フォルダ
```

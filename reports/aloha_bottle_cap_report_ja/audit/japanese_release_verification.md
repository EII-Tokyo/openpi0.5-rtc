# 日本語版リリース検証

検証日：2026-07-31

## 成果物

- PDF：`aloha_bottle_cap_report_ja.pdf`
- ページ数：51
- 本文参照図：19
- 表：30
- 生成した科学図：18
- 実写真：1
- 教示データの実フレームを組み合わせた図：2

## 自動検証

次のコマンドを報告ディレクトリで実行した。

```bash
/home/eii/project/openpi0.5-rtc-reward-learning/.venv/bin/python scripts/verify_japanese_report.py
/home/eii/project/openpi0.5-rtc-reward-learning/.venv/bin/python -m pytest -q ../../tests/report/test_bilingual_report_contract.py
```

結果：

- 報告検証：PASS
- 回帰テスト：7件すべて合格
- 空白ページ：0
- 分割可能な長表：0
- Noto Serif CJK JP / Noto Sans CJK JP：PDFへ埋め込み済み
- 主要事実の照合：9項目
- 致命的な LaTeX エラー、未定義参照、未定義引用、Overfull box：0

## 目視検証

全51ページを縮小一覧で確認し、次を原寸で再確認した。

- 表紙：左列の項目名と右列の内容が行ごとに整列している。
- 図1.1：上部俯瞰カメラの指示線が、上部支持架台中央の実カメラへ向いている。
- 図4.1：最後の二つの工程枠の間に明確な余白があり、枠と矢印が重ならない。
- 計画表：P0とP1は1枚の横向きページに収め、P2とP3はそれぞれ見出しと表を同一ページに配置した。
- 付録：付録Aの見出しだけが単独ページを占めず、A.1とA.2を同じページから開始した。
- すべての横向き表は1ページ内に収まり、ページ境界で分割されていない。

## 境界

Poppler は一部の埋め込み CJK フォントに型不一致の警告を表示するが、`pdffonts` では対象フォントが埋め込み済みかつ Unicode 対応として確認でき、PDFは正常に開ける。報告中の現場処理速度と連続運転時間は、逐次計数表や完全映像ではなく、現場展示時の観測に基づく概算として明記している。

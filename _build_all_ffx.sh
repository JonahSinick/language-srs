#!/bin/bash
sections=(
  "01_ビサイド"
  "02_連絡船リキ号"
  "03_キーリカ"
  "04_連絡船ウィンノ号"
  "05_ルカ"
  "06_ミヘンセッツ街道"
  "07_キノコ岩街道"
  "08_ジョゼ街道"
  "09_幻光河"
  "10_グアドサラム"
  "11_雷平原"
  "12_マカラーニャの森"
  "13_マカラーニャ寺院"
  "14_サヌビア砂漠とホーム"
  "15_聖ベベル宮"
  "16_浄罪の路"
  "17_ナギ平原"
  "18_ガガゼト山"
  "19_ザナルカンド遺跡"
  "20_シン"
  "21_エンディング"
)

total=${#sections[@]}
for i in "${!sections[@]}"; do
  section="${sections[$i]}"
  num=$((i + 1))
  echo ""
  echo "=========================================="
  echo "[$num/$total] Building: $section"
  echo "=========================================="
  python3 build_ffx.py "$section" 2>&1
  if [ $? -ne 0 ]; then
    echo "ERROR building $section"
  fi
done

echo ""
echo "=========================================="
echo "ALL SECTIONS COMPLETE"
echo "=========================================="

# Print summary
for section in "${sections[@]}"; do
  if [ -f "ffx/$section/segments.json" ]; then
    count=$(python3 -c "import json; print(len(json.load(open('ffx/$section/segments.json'))))")
    echo "  $section: $count segments"
  else
    echo "  $section: MISSING"
  fi
done

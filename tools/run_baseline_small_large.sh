#!/usr/bin/env bash
set -e

cd /home/linuxu/Desktop/Research_Project/yolov7

python test.py \
  --weights runs/train/seadronessee_baseline/weights/best.pt \
  --data data/seadronessee.yaml \
  --img 640 \
  --conf 0.25 \
  --iou 0.3 \
  --task val \
  --name exp_baseline_small_large_conf025 \
  --project runs/test \
  --save-txt --save-conf --exist-ok

python scripts/04_eval_small_large_baseline.py

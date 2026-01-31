# Experiment 01 — Baseline YOLOv7

## Goal
Establish a reference YOLOv7 detector on SeaDronesSee V2 (Object Detection), using standard fine-tuning from pretrained weights without architectural changes.

## Inputs
- Dataset: SeaDronesSee V2 (not included in this repo)
- Dataset YAML: `data/seadronessee.yaml`
- Pretrained weights: `weights/yolov7.pt` (download separately)

## Training
Run YOLOv7 training with the baseline configuration:

```bash
python3 train.py \
  --workers 4 \
  --device 0 \
  --batch-size 8 \
  --data data/seadronessee.yaml \
  --img 640 640 \
  --cfg cfg/training/yolov7.yaml \
  --weights weights/yolov7.pt \
  --name seadronessee_baseline \
  --hyp data/hyp.scratch.p5.yaml
Expected outputs
YOLOv7 will write outputs under:

runs/train/seadronessee_baseline/

weights/best.pt

weights/last.pt

results.txt

training curves (if enabled by YOLOv7)

Evaluation (Test)
Evaluate the trained model on the validation set:

python3 test.py \
  --weights runs/train/seadronessee_baseline/weights/best.pt \
  --data data/seadronessee.yaml \
  --img 640 \
  --device 0
Expected outputs
YOLOv7 will report detection metrics to stdout and may create an output folder under runs/test/ depending on YOLOv7 version/config.

Notes
This baseline is used as the reference point for later experiments.

The dataset and weights are not included in this repository due to size/licensing constraints.

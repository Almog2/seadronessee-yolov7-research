# Experiment 04 — Binary Small/Large CNN Concept (Baseline Run)

This experiment evaluates the idea of grouping object classes into **two size-based groups**:
- **Small**: swimmer, life_saving_appliances
- **Large**: boat, jetski, buoy

Instead of evaluating 5-class classification, we map both GT and predictions into {Small, Large}
and compute a 2×2 confusion matrix and group-level Precision/Recall/F1.

This section documents the **Baseline YOLOv7 run output** evaluated in a binary manner.

---

## 1) Class grouping

Original class order (YOLO YAML):
`["swimmer", "boat", "jetski", "life_saving_appliances", "buoy"]`

Group mapping:
- Small: swimmer, life_saving_appliances
- Large: boat, jetski, buoy

Implementation:
- group_id = 0 for Small
- group_id = 1 for Large

---

## 2) Step A — Produce YOLO predictions (Baseline weights)

Command:

```bash
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
Predictions directory used by the evaluator:
/home/linuxu/Desktop/Research_Project/yolov7/runs/test/exp_baseline_small_large/labels

3) Step B — Evaluate binary Small/Large performance
Script:

scripts/04_eval_small_large_baseline.py

Outputs:

small_large_results.txt written into:
/home/linuxu/Desktop/Research_Project/yolov7/small_large_eval_baseline

Evaluation method:

For each validation image:

read GT boxes (from data/labels/val)

read predicted boxes (from YOLO saved labels)

convert boxes to absolute xyxy

greedy IoU matching (threshold IoU >= 0.30)

update 2×2 confusion matrix for groups (Small/Large)

compute TP/FP/FN per group → Precision/Recall/F1

4) Results (Baseline binary evaluation)
Confusion matrix (rows=true, cols=pred):

  Small  Large
Small 6003 27
Large 13 2868

Group metrics:

Small:

Precision = 0.9045

Recall = 0.9189

F1 = 0.9116

Large:

Precision = 0.9327

Recall = 0.9276

F1 = 0.9301

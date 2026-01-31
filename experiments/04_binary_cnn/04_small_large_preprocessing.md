# Experiment 04 — Binary Small/Large Evaluation (Preprocessing Run)

This experiment evaluates a **binary grouping** of the original 5 SeaDronesSee classes into:
- **Small**: swimmer, life_saving_appliances
- **Large**: boat, jetski, buoy

We apply the binary evaluation on the **preprocessing YOLOv7 run** predictions.

---

## 1) Class grouping

Original class order (YOLO YAML):
`["swimmer", "boat", "jetski", "life_saving_appliances", "buoy"]`

Group mapping:
- **Small (0)**: swimmer, life_saving_appliances
- **Large (1)**: boat, jetski, buoy

---

## 2) Inputs

Validation data (ground-truth):
- Images: `/home/linuxu/Desktop/Research_Project/yolov7/data/images/val`
- Labels: `/home/linuxu/Desktop/Research_Project/yolov7/data/labels/val`

YOLO predictions (preprocessing run):
- Prediction labels directory:
  `/home/linuxu/Desktop/Research_Project/yolov7/runs/test/seadronessee_tiled_test/labels`

> Note: the evaluator reads YOLO prediction TXT files from the predictions directory and matches them to GT by filename stem.

---

## 3) Method

We convert both GT boxes and predicted boxes to absolute pixel coordinates and perform **greedy IoU matching**:
- IoU threshold: `0.30`
- Each prediction can match at most one GT box and vice versa.
- After matching:
  - Matched pairs contribute to the **Small/Large confusion matrix**
  - Unmatched predictions are counted as **FP** (by predicted group)
  - Unmatched GT boxes are counted as **FN** (by true group)

Outputs:
- `small_large_results.txt` containing:
  - Confusion matrix (Small/Large)
  - TP / FP / FN and precision/recall/F1 per group

---

## 4) How to run

From the YOLOv7 root:

```bash
python scripts/04_eval_small_large_preprocess.py
Outputs will be saved to:

/home/linuxu/Desktop/Research_Project/yolov7/small_large_eval_preprocess
Key file:

small_large_results.txt

5) Results
Experiment 4 — Binary Small vs Large (Matched-Only Evaluation, IoU=0.30)
======================================================================

Definition:
- Small group: {swimmer, life_saving_appliances}
- Large group: {boat, jetski, buoy}
Protocol:
- We evaluate only predictions that can be matched to a GT box at IoU >= 0.30 (matched-only).
- Unmatched predictions are excluded from the matched-only classification metrics.

------------------------------------------------------------
Preprocessing outputs (small/large evaluation)
------------------------------------------------------------
Matched-only protocol summary:
- Total YOLO predictions processed: 8471
- Matched at IoU >= 0.30: 7719
- No-match (excluded from matched-only evaluation): 752

Matched-only performance:
- Pipeline accuracy (matched-only): 0.7610
- Weighted F1 (matched-only): 0.7669

Group-level metrics (matched-only):
- Small:  Precision=0.9324  Recall=0.6856  F1=0.7902
- Large:  Precision=0.6011  Recall=0.9050  F1=0.7223

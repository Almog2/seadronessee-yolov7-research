# Experiment 04 — Binary Small vs Large Analysis

## Goal
Analyze detection and classification performance from an operational perspective by grouping object classes into **Small** and **Large** categories.

This experiment aims to:
- Emphasize the importance of **small-object detection** (e.g., swimmers),
- Reduce sensitivity to fine-grained class confusion,
- Provide a clearer view of how preprocessing and the CNN stage affect size-dependent behavior.

---

## Binary Group Definition
Original SeaDronesSee classes are grouped as follows:

- **Small objects**
  - swimmer
  - life_saving_appliances

- **Large objects**
  - boat
  - jetski
  - buoy

All evaluations are performed under a **matched-only protocol**:
- A prediction is evaluated only if it can be matched to a ground-truth box with **IoU ≥ 0.30**.

---

## Evaluation Scenarios
Binary evaluation is performed for two YOLO-based pipelines:

1. **Baseline YOLOv7**
   - Trained on the original dataset (Experiment 01).

2. **YOLOv7 with Preprocessing**
   - Trained on the preprocessed dataset with tiling and filtering (Experiment 02).

This allows a direct comparison of how preprocessing affects small vs. large object performance.

---

## Step 1 — Generate YOLO Predictions
Run YOLOv7 evaluation with binary-oriented thresholds:

```bash
python3 test.py \
  --weights runs/train/seadronessee_baseline/weights/best.pt \
  --data data/seadronessee.yaml \
  --img 640 \
  --conf 0.25 \
  --iou 0.30 \
  --task val \
  --name exp_baseline_small_large \
  --project runs/test \
  --save-txt \
  --save-conf \
  --exist-ok
For the preprocessing model, use the corresponding weights and dataset YAML.

Step 2 — Binary Evaluation Script
Evaluate predictions using the small/large grouping:

python3 eval_small_large.py
Evaluation details
Predictions and GT boxes are matched greedily using IoU ≥ 0.30.

Each GT and prediction is mapped to Small or Large.

Confusion matrix and group-level metrics are computed.

Metrics Reported
For each group (Small / Large), the following metrics are reported:

True Positives (TP)

False Positives (FP)

False Negatives (FN)

Precision

Recall

F1-score

In addition, a 2×2 confusion matrix (rows = true group, columns = predicted group) is produced.

Outputs
Evaluation outputs are saved under:

small_large_eval_*/
├── small_large_results.txt
The results file includes:

Dataset and matching statistics,

Binary confusion matrix,

Group-level precision, recall, and F1-scores.

Notes
This binary analysis abstracts away fine-grained class errors and focuses on size-dependent behavior.

Improvements in the Small group are particularly important for maritime search-and-rescue scenarios.

Results from this experiment support the main conclusion of the paper:
improving proposal quality via preprocessing has a larger impact than post-detection classification refinement.

Dataset files, trained weights, and YOLO run outputs are not included in this repository.


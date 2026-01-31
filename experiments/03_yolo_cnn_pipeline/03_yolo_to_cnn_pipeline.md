# Experiment 03 — YOLOv7 → CNN Verification Stage (Multi-Class)

In this experiment we add a lightweight CNN stage after YOLOv7.
The goal is to **refine the predicted class** for each YOLO detection, especially under small-object ambiguity.

Pipeline (evaluated on validation images):
1. Run YOLOv7 on the validation set and save predictions (`--save-txt --save-conf`).
2. Build a CNN dataset by cropping predicted boxes from the original validation images.
3. Assign each crop a label using the best-matching GT box (IoU-based matching).
4. Train a small CNN (MobileNetV2) on the cropped dataset (5 classes).
5. Evaluate the YOLO→CNN pipeline: for each YOLO prediction, crop the box and let CNN output the refined class,
   then compare to GT using IoU matching.

---

## 1) Inputs

We use:
- Validation images: `/home/linuxu/Desktop/Research_Project/yolov7/data/images/val`
- Validation GT labels: `/home/linuxu/Desktop/Research_Project/yolov7/data/labels/val`
- YOLO predictions (from Experiment 02 test run):  
  `/home/linuxu/Desktop/Research_Project/yolov7/runs/test/seadronessee_tiled_test/labels`

Class order (must match YOLO YAML):
`["swimmer", "boat", "jetski", "life_saving_appliances", "buoy"]`

---

## 2) Step A — Build CNN Dataset from YOLO Predictions

Script:
- `tools/build_cnn_dataset_from_yolo_test.py`

Output:
- Cropped images organized by class folder:
  `/home/linuxu/Desktop/Research_Project/yolov7/cnn_dataset_from_test/images/<class_name>/...`
- Metadata CSV:
  `/home/linuxu/Desktop/Research_Project/yolov7/cnn_dataset_from_test/meta.csv`

Labeling rule:
- Each YOLO predicted box is matched to the best GT box (highest IoU).
- A crop is kept only if IoU >= 0.30.
- Crops are padded by 10% for context.
- Very small crops are discarded.

Run output (our execution):
- Crops saved: 7971
- Skipped (no GT match): 473
- Skipped (too small): 27
- Per-class counts:
  - swimmer: 5167
  - boat: 2053
  - jetski: 310
  - life_saving_appliances: 117
  - buoy: 324

---

## 3) Step B — Train CNN (MobileNetV2)

Script:
- `scripts/03_cnn_train.py`

Notes:
- We use MobileNetV2 (lightweight) with a 5-class classifier head.
- Crops are resized to 128×128.
- WeightedRandomSampler is used to mitigate class imbalance.
- Best model checkpoint is saved to:
  `/home/linuxu/Desktop/Research_Project/yolov7/cnn_runs/best_cnn.pt`

---

## 4) Step C — Evaluate CNN Alone (Held-out Split)

Script:
- `scripts/03_cnn_eval.py`

Outputs:
- `cnn_results.txt`
- `cnn_confusion_matrix.csv`
- `cnn_confusion_matrix.png`

---

## 5) Step D — Evaluate Full YOLO→CNN Pipeline (Matched Predictions Only)

Script:
- `scripts/03_pipeline_eval_yolo_plus_cnn.py`

Evaluation logic:
- For each YOLO predicted box:
  - Crop from the original validation image (with padding),
  - CNN predicts a refined class,
  - Match the predicted box to GT using IoU>=0.30,
  - Update confusion matrix using (GT class, CNN refined class).

Outputs:
- `pipeline_results.txt`
- `pipeline_confusion_matrix.csv`
- `pipeline_confusion_matrix.png`

---

## Repository note

We do not store:
- the dataset crops
- YOLO runs outputs
- CNN checkpoints

These are excluded due to size constraints.

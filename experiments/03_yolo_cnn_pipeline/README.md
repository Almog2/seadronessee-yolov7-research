# Experiment 03 — YOLO → CNN Verification Pipeline

## Goal
Evaluate a two-stage detection pipeline in which a lightweight CNN is used to refine YOLOv7 classification decisions on detected objects.

This experiment isolates the **classification refinement capability** of a CNN when operating on crops extracted from YOLO detections, without attempting to recover missed detections.

---

## Pipeline Description
The pipeline consists of the following stages:

1. **YOLOv7 Detection**  
   Use the trained YOLOv7 model from Experiment 02 (preprocessing run) to generate detections on the validation set.

2. **Crop Extraction & Dataset Construction**  
   Extract image crops from YOLO detections that can be matched to ground-truth boxes (IoU ≥ 0.30).  
   Each crop is labeled using the matched ground-truth class.

3. **CNN Training**  
   Train a lightweight CNN (MobileNetV2) on the extracted crops for multi-class classification.

4. **Pipeline Evaluation**  
   Replace YOLO’s class prediction with the CNN’s prediction for matched detections and evaluate the end-to-end pipeline.

---

## Step 1 — Generate YOLO Detections
Run YOLOv7 evaluation from Experiment 02 to produce prediction label files:

```bash
python3 test.py \
  --data data/seadronessee_tiled_train.yaml \
  --img 640 \
  --weights runs/train/seadronessee_tiled_train/weights/best.pt \
  --task val \
  --device 0 \
  --name seadronessee_tiled_test \
  --save-txt \
  --save-conf
YOLO predictions will be saved under:

runs/test/seadronessee_tiled_test/labels/
Step 2 — Build CNN Dataset from YOLO Outputs
Construct a CNN dataset from matched YOLO detections:

python3 tools/build_cnn_dataset_from_yolo_test.py
Dataset construction details
Crops are extracted from validation images.

A detection is kept if it matches a ground-truth box with IoU ≥ 0.30.

Crops are padded to include local context.

Extremely small crops are filtered out.

Class imbalance is preserved and later handled during CNN training.

Output
cnn_dataset_from_test/
├── images/
│   ├── swimmer/
│   ├── boat/
│   ├── jetski/
│   ├── life_saving_appliances/
│   └── buoy/
└── meta.csv
Step 3 — CNN Training
Train a lightweight CNN classifier on the extracted crops:

python3 train_cnn.py
CNN details
Architecture: MobileNetV2

Input size: 128 × 128

Loss: Cross-Entropy

Optimizer: Adam

Class imbalance handled using a WeightedRandomSampler

Train/validation split: 80 / 20

The best model is saved as:

cnn_runs/best_cnn.pt
Step 4 — CNN Evaluation (Standalone)
Evaluate CNN classification performance on a held-out split:

python3 eval_cnn.py
Outputs
cnn_results.txt

cnn_confusion_matrix.csv

cnn_confusion_matrix.png

Step 5 — End-to-End YOLO + CNN Evaluation
Evaluate the full pipeline by replacing YOLO class predictions with CNN predictions for matched detections:

python3 eval_yolo_cnn_pipeline.py
Evaluation protocol
Only detections matched to GT boxes (IoU ≥ 0.30) are evaluated.

Metrics reflect classification refinement, not detection recall recovery.

Outputs
yolo_cnn_eval/
├── pipeline_results.txt
├── pipeline_confusion_matrix.csv
└── pipeline_confusion_matrix.png
Notes
The CNN operates strictly on YOLO detections; missed detections cannot be recovered.

Improvements observed here reflect classification refinement, not proposal generation.

This experiment motivates the size-based binary analysis in Experiment 04.

Dataset files and trained weights are not included in this repository.

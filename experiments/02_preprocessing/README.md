# Experiment 02 — YOLOv7 with Data-Centric Preprocessing

## Goal
Evaluate the effect of data-centric preprocessing on small-object detection performance without modifying the YOLOv7 architecture.

The main objective is to improve the effective visibility and scale of small targets (e.g., swimmers) by transforming the training data rather than changing the detector.

---

## Preprocessing Strategy
The preprocessing pipeline is applied **only to the training set** and consists of:

1. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**  
   Enhances local contrast to improve object visibility over water surfaces.

2. **HSV Color Jitter (train only)**  
   Introduces photometric variation to improve robustness to illumination changes.

3. **SAHI-like Tiling**  
   Large images are split into overlapping tiles (640×640) to increase the relative size of small objects.

4. **Bounding-box filtering**  
   Very small or low-visibility boxes are removed to reduce label noise.

This preprocessing produces a new training dataset while keeping the validation set unchanged.

---

## Dataset Preparation
Run the preprocessing script to generate the tiled training dataset:

```bash
python3 tools/preprocessing_tiling.py
Output
A new dataset directory is created (example):

data_preprocessed_full/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── seadronessee_tiled_train.yaml
The YAML file points YOLOv7 to the preprocessed training images and the original validation set.

Training
Train YOLOv7 using the preprocessed dataset:

python3 train.py \
  --workers 4 \
  --device 0 \
  --batch-size 4 \
  --epochs 15 \
  --img 640 640 \
  --data data/seadronessee_tiled_train.yaml \
  --cfg cfg/training/yolov7.yaml \
  --weights weights/yolov7.pt \
  --name seadronessee_tiled_train \
  --hyp data/hyp.scratch.p5.yaml
Expected outputs
runs/train/seadronessee_tiled_train/

weights/best.pt

results.txt

training curves

Evaluation (Test)
Evaluate the trained model on the validation set:

python3 test.py \
  --data data/seadronessee_tiled_train.yaml \
  --img 640 \
  --weights runs/train/seadronessee_tiled_train/weights/best.pt \
  --task val \
  --device 0 \
  --name seadronessee_tiled_test \
  --conf-thres 0.25 \
  --iou-thres 0.5 \
  --save-conf \
  --save-txt
Notes
No architectural changes were made to YOLOv7.

Improvements (or regressions) observed in this experiment are attributed solely to data preprocessing.

Outputs from this experiment are used as input for the CNN verification stage in Experiment 03.

Dataset files and trained weights are not included in this repository.


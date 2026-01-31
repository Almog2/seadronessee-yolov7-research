# Experiment 02 — Preprocessing for Small-Object Detection (CLAHE + HSV Jitter + Tiling)

This experiment evaluates whether **data-centric preprocessing** improves YOLOv7 performance on small objects in SeaDronesSee.

Compared to the baseline (Experiment 01), we apply:
- **CLAHE** contrast enhancement (train + val)
- **HSV jitter** augmentation (train only)
- **SAHI-like tiling** into 640×640 overlapping tiles (train + val)
- Filtering of tiny bounding boxes in tiles to reduce noisy labels

After preprocessing, YOLOv7 is trained and evaluated on the resulting dataset.

---

## 1) Preprocessing Script

The preprocessing pipeline is implemented in:

- `tools/preprocess_dataset_full.py`

It reads the original dataset from:

- `/home/linuxu/Desktop/Research_Project/yolov7/data`

and writes a new preprocessed dataset to:

- `/home/linuxu/Desktop/Research_Project/yolov7/data_preprocessed_full`

The output directory follows standard YOLO layout:

data_preprocessed_full/
├── images/
│ ├── train/
│ └── val/
└── labels/
├── train/
└── val/


---

## 2) Tiling-Only Variant (Train Only)

A separate tiling-only script was used to tile the training split into:

- `data/images/train_tiled`
- `data/labels/train_tiled`

Script location:

- `tools/tile_train_only.py`

This variant keeps only tiles with at least one label, reducing dataset size.

---

## 3) Training (YOLOv7)

Training is executed from the YOLOv7 root directory using the tiled dataset YAML:

- `data/seadronessee_tiled_train.yaml`

Runner script:

- `scripts/02_preprocessing_train.py`

---

## 4) Testing (YOLOv7)

Evaluation is executed from the YOLOv7 root directory using the same dataset YAML:

Runner script:

- `scripts/02_preprocessing_test.py`

This run also saves predictions (`--save-txt`, `--save-conf`) for additional analysis.

---

## Outputs

YOLOv7 outputs are stored under:

- `runs/train/seadronessee_tiled_train/`
- `runs/test/seadronessee_tiled_test/` (depending on YOLOv7 version/config)

All outputs are excluded from the repository due to size constraints.

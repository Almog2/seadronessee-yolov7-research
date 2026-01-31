# 00 — Dataset Preparation (COCO → YOLO)

This document describes how the SeaDronesSee dataset was prepared for training and evaluation with YOLOv7.

The preparation process includes:
- Dataset structure and class definition
- Conversion from COCO annotation format to YOLO format
- Dataset configuration using a YOLO-compatible YAML file

All steps are documented to ensure reproducibility.

---

## 1) Dataset Overview

All experiments in this project are conducted on the **SeaDronesSee** maritime UAV dataset.

The dataset contains five object categories:

- swimmer  
- boat  
- jetski  
- buoy  
- life-saving appliances  

Ground-truth annotations are available for the training and validation splits.  
Labels for the official test split are not publicly available and are therefore not used for quantitative evaluation.

---

## Dataset Acquisition

The SeaDronesSee dataset used in this project corresponds to **Version 2 (V2)**.

The dataset can be downloaded from the official project website:
https://seadronessee.cs.uni-tuebingen.de

Due to its size and licensing terms, the dataset is not included in this repository.
Users are expected to download the dataset manually and place it in the directory structure described in this document.


## 2) Target Dataset Structure (YOLO Format)

For compatibility with YOLOv7, the dataset is organized using the standard YOLO directory layout.

Expected structure under the YOLOv7 directory:

data/
├── images/
│ ├── train/
│ └── val/
├── labels/
│ ├── train/
│ └── val/
└── annotations/
├── instances_train.json
└── instances_val.json


- `images/` contains the raw UAV images
- `labels/` contains YOLO-format label files generated from COCO annotations
- `annotations/` contains the original COCO JSON files

---

## 3) COCO → YOLO Conversion

The original SeaDronesSee annotations are provided in COCO format.
To train YOLOv7, these annotations were converted into YOLO label files using a custom conversion script.

### Conversion Script

The conversion utility is located at:

tools/coco_to_yolo.py


The script performs the following operations:
- Reads COCO annotations from a JSON file
- Converts bounding boxes from COCO format  
  `(top-left x, top-left y, width, height)`
  to YOLO format  
  `(class_id, x_center, y_center, width, height)` normalized by image size
- Writes one `.txt` label file per image
- Filters out categories that are not used in the experiments

---

## 4) Class Definition and Filtering

During conversion, only the following five classes are retained:

[swimmer, boat, jetski, life_saving_appliances, buoy]


All other categories (if present) are excluded.

The class order defined during conversion is preserved consistently across:
- YOLO training
- Evaluation
- Post-detection CNN experiments

---

## 5) Running the Conversion Script

The conversion script is executed separately for the training and validation splits by updating the configuration section inside `tools/coco_to_yolo.py`.

### Training Split
- COCO JSON: `data/annotations/instances_train.json`
- Images directory: `data/images/train`
- Output labels: `data/labels/train`

### Validation Split
- COCO JSON: `data/annotations/instances_val.json`
- Images directory: `data/images/val`
- Output labels: `data/labels/val`

After running the script, each image has a corresponding YOLO label file with the same base name.

---

## 6) Dataset Configuration File (seadronessee.yaml)

YOLOv7 uses a dataset configuration file to locate images and define classes.

The dataset configuration used in this project is:

configs/seadronessee.yaml


Example content:

```yaml
train: data/images/train
val: data/images/val

nc: 5
names: [swimmer, boat, jetski, life_saving_appliances, buoy]
This file is referenced directly during training and evaluation.

7) Notes on Reproducibility
The dataset itself is not included in this repository due to size and licensing constraints.

Absolute paths may be used locally on the training server, but relative paths are preferred for documentation and portability.

All experiments consistently use the same class definitions and validation split to ensure fair comparison across runs.

Result
After completing the dataset preparation:

All images are paired with YOLO-format label files

The dataset structure is compatible with YOLOv7

The project is ready for baseline training and subsequent experiments



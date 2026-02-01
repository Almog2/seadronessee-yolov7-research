# seadronessee-yolov7-research  
## Detecting Missing People at Sea Using AI  
### YOLOv7 with Data-Centric Preprocessing and CNN-Based Verification

This repository accompanies an academic seminar project in **Computational Learning** at **Afeka College of Engineering (2026)**.

We study deep-learning-based detection of **swimmers and rescue-related objects** in maritime UAV imagery, focusing on **small-object detection** under real sea conditions (low contrast, reflections, occlusions, and strong class imbalance).

---

## Abstract
Detecting swimmers and rescue-related objects in maritime UAV imagery is challenging due to small apparent object size, low contrast against water surfaces, specular reflections, and severe class imbalance.  
In this project, we evaluate a staged detection pipeline on the **SeaDronesSee (Object Detection V2)** dataset. The pipeline consists of:

1. **YOLOv7** object detector (baseline fine-tuning),
2. **Data-centric preprocessing** designed to improve the effective visibility/scale of small targets (without changing YOLO architecture),
3. A lightweight **CNN verification stage (MobileNetV2)** trained on crops extracted from YOLO detections and evaluated under a **matched-only** protocol (**IoU ≥ 0.30**) to isolate classification refinement.

We analyze (i) how preprocessing affects detector behavior, (ii) how a verification CNN refines classification on matched detections, and (iii) how performance differs between **small vs. large** object groups.  
Overall, our findings indicate that improving upstream proposal quality is the dominant factor for end-to-end performance, while the CNN stage primarily functions as a **classification refinement mechanism** rather than a recovery solution for missed detections.

---

## Pipeline Overview
The research follows a staged experimental design:

### 1) Baseline Detection (Experiment 1)
Fine-tune YOLOv7 using standard pretrained weights and a standard training configuration.

### 2) Preprocessing for Small Objects (Experiment 2)
A data-centric intervention applied to the dataset preparation/training pipeline to improve small-object visibility.  
In our implementation this includes **selective tiling** (SAHI-like) and filtering rules to increase the effective scale of small targets without changing the detector architecture.

### 3) Two-Stage YOLO → CNN Pipeline (Experiment 3)
Train a lightweight CNN verifier (**MobileNetV2**) on crops extracted from YOLO detections.
Evaluation is performed under a **matched-only** protocol (**IoU ≥ 0.30**) to isolate classification refinement quality.

### 4) Binary Small vs Large Analysis (Experiment 4)
Group classes into:
- **Small:** swimmer, life_saving_appliances  
- **Large:** boat, jetski, buoy  

This analysis emphasizes operational priorities (e.g., swimmers) and highlights size-dependent behavior.

---

## Experiments Summary
Each experiment is documented step-by-step under the `experiments/` directory.

- **Experiment 1 – Baseline YOLOv7**  
  Establishes reference detection performance on SeaDronesSee V2.

- **Experiment 2 – YOLOv7 with Preprocessing**  
  Evaluates how data-centric preprocessing (tiling/filtering) affects small-object detection and confusion patterns.

- **Experiment 3 – Multi-Class CNN Verification**  
  Adds a post-detection CNN to refine YOLO classification decisions on matched detections.

- **Experiment 4 – Binary Size-Based Evaluation**  
  End-to-end evaluation using a small vs. large grouping to better reflect maritime SAR priorities.

---

## Repository Structure
seadronessee-yolov7-research/
├── experiments/ # Experiment run instructions + notes
│ ├── 00_setup/
│ ├── 01_baseline/
│ ├── 02_preprocessing/
│ ├── 03_yolo_cnn_pipeline/
│ └── 04_small_large/
├── tools/ # Utility scripts (e.g., COCO → YOLO, dataset builders)
├── configs/ # YAML/config files used in experiments
├── docs/ # Paper / figures / supplementary docs
└── README.md


> Note: This repository does **not** include the dataset, large run outputs, or model weights.

---

## Dataset
We use **SeaDronesSee – Object Detection V2** (Maritime Computer Vision / MaCVi initiative).

- Website: https://seadronessee.cs.uni-tuebingen.de  
- Version used: **Object Detection V2**
- Classes used in this project (YOLO order):  
  `swimmer, boat, jetski, life_saving_appliances, buoy`

Due to dataset size and licensing constraints, the dataset is **not included** in this repository.

### Dataset Directory Layout (Expected)
After downloading SeaDronesSee V2, place it under your YOLOv7 working directory as:

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


For details on converting annotations and preparing the dataset, see:  
`experiments/00_setup/00_dataset_preparation.md`

---

## Reproducibility & How to Run
This repository is organized as a reproducible research pipeline:

1. **Environment setup** and dataset preparation:  
   `experiments/00_setup/`

2. Each experiment has its own folder with:
   - executable commands (train/test),
   - required config references,
   - expected outputs and notes.

Please refer to the relevant experiment folder for detailed instructions.

### What is intentionally not included
To keep the repository lightweight and compliant:
- **dataset files** (images/labels/annotations),
- **trained weights** (`*.pt`),
- **YOLO run outputs** (`runs/` directories).

---

## Citation
If you use this repository for academic purposes, please cite the associated seminar paper (see `[Seminar Paper.docx](https://github.com/user-attachments/files/24991638/Seminar.Paper.doc)
cs/`).

---

## Authors
- **Almog Dinur**  
- **Yosi Alemu**  

Seminar in Computational Learning  
Afeka College of Engineering — 2026

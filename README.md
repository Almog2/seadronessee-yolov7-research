# seadronessee-yolov7-research
# Detecting Missing People at Sea Using AI  
### YOLOv7 with Preprocessing and CNN-Based Verification

This repository accompanies an academic seminar project in **Computational Learning** at **Afeka College of Engineering**.

The project investigates the use of deep learning for detecting missing people and rescue-related objects in maritime UAV imagery, with a focus on **small-object detection under challenging sea conditions**.

---

## Abstract

Detecting swimmers and rescue-related objects in maritime UAV imagery is challenging due to small apparent object size, low contrast against water surfaces, specular reflections, and strong class imbalance.  
In this project, we evaluate a **multi-stage detection pipeline** on the SeaDronesSee dataset. The pipeline consists of a YOLOv7-based detector, optional data-centric preprocessing aimed at improving small-object visibility, and a post-detection CNN verification stage operating on cropped detections.

We analyze how preprocessing affects detector behavior, how a verification CNN refines classification decisions, and how performance differs between small and large object groups. Our findings show that **improving upstream proposal quality is the dominant factor** for overall pipeline performance, while the CNN stage primarily acts as a refinement mechanism rather than a recovery solution for missed detections.

---

## Pipeline Overview

The research follows a staged experimental design:

1. **Baseline Detection**  
   Fine-tuning YOLOv7 using standard pretrained weights and default training configuration.

2. **Preprocessing for Small Objects**  
   A data-centric intervention applied only to the training set, using selective tiling and filtering to increase the effective scale of small targets without changing the detector architecture.

3. **Two-Stage YOLO → CNN Pipeline**  
   A lightweight CNN verifier (MobileNetV2) is trained on crops extracted from YOLO detections and evaluated under a matched-only protocol (IoU ≥ 0.30) to isolate classification refinement quality.

4. **Binary Small vs Large Analysis**  
   Classes are grouped into small (swimmer, life-saving appliances) and large (boat, jetski, buoy) to analyze size-dependent behavior and sensitivity to small targets.

---

## Experiments Summary

- **Experiment 1 – Baseline YOLOv7**  
  Establishes a reference detector performance on the SeaDronesSee dataset.

- **Experiment 2 – YOLOv7 with Preprocessing**  
  Evaluates how selective tiling and filtering affect small-object detection and background confusion.

- **Experiment 3 – Multi-Class CNN Verification**  
  Adds a post-detection CNN to refine YOLO classification decisions on matched detections.

- **Experiment 4 – Binary Size-Based Evaluation**  
  Performs an end-to-end evaluation using small vs large object grouping to better reflect operational priorities in maritime search and rescue.

Each experiment is documented step-by-step under the `experiments/` directory.

---

## Repository Structure

seadronessee-yolov7-research/
├── experiments/ # Experiment notebooks and run instructions
│ ├── 00_setup/
│ ├── 01_baseline/
│ ├── 02_preprocessing/
│ ├── 03_yolo_cnn_pipeline/
│ └── 04_binary_cnn/
├── tools/ # Utility scripts (e.g., COCO → YOLO conversion)
├── configs/ # Dataset and experiment configuration files
├── external/ # YOLOv7 submodule
├── docs/ # Supplementary documentation
└── README.md


---

## Dataset

This project uses the **SeaDronesSee** maritime UAV dataset (Version 2).

The dataset is publicly available as part of the Maritime Computer Vision (MaCVi) initiative and is commonly used for benchmarking maritime object detection methods.

- Dataset website: https://seadronessee.cs.uni-tuebingen.de
- Version used: **Object Detection V2**

Due to dataset size and licensing constraints, the dataset is **not included** in this repository.

### Dataset Setup

After downloading SeaDronesSee V2, the dataset should be placed inside the YOLOv7 directory using the following structure:

external/yolov7/data/
├── images/
│ ├── train/
│ └── val/
├── labels/
│ ├── train/
│ └── val/
└── annotations/
├── instances_train.json
└── instances_val.json


Refer to `experiments/00_setup/00_dataset_preparation.md` for details on annotation conversion and preprocessing.


## How to Run

This repository is organized as a **reproducible research pipeline**.

- Environment setup and dataset preparation are documented under `experiments/00_setup/`
- Each experiment has its own folder with execution instructions and command-line examples
- YOLOv7 is included as a Git submodule under `external/yolov7`

Please refer to the relevant experiment folder for detailed instructions.

---

## Authors

- **Almog Dinur**  
- **Yosi Alemu**

Seminar in Computational Learning  
Afeka College of Engineering  
2026

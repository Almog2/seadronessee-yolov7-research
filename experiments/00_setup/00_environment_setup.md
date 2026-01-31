# 00 — Environment Setup (YOLOv7) — Linux Only

This document describes how to set up the YOLOv7 training environment for this seminar project on a Linux machine.

It includes:
- Fetching the YOLOv7 submodule
- Installing dependencies
- Verifying YOLOv7 imports correctly
- Downloading pretrained YOLOv7 weights

> Note: YOLOv7 is tracked in this repository as a submodule under: `external/yolov7`.

---

## 1) Fetch YOLOv7 Submodule

From the repository root:

```bash
git submodule update --init --recursive
After this step, YOLOv7 should exist under:

external/yolov7/

2) Install Dependencies
From the repository root:

cd external/yolov7
python3 -m pip install -r requirements.txt
3) Verify Imports (Sanity Check)
This verifies that YOLOv7 modules can be imported successfully.

cd external/yolov7
python3 -c "import models, utils; print('yolov7 imports OK')"
Expected output:

yolov7 imports OK

4) Download Pretrained Weights (yolov7.pt)
YOLOv7 training in our experiments starts from standard pretrained weights.

cd external/yolov7
mkdir -p weights
curl -L -o weights/yolov7.pt https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
After download, the file should exist at:

external/yolov7/weights/yolov7.pt

5) Optional Checks
Check Python version
python3 --version
Check GPU availability (if relevant)
nvidia-smi
Result
After completing this setup, you should have:

YOLOv7 code available under external/yolov7/

All dependencies installed

Pretrained weights downloaded

A working environment ready for dataset preparation and training experiments


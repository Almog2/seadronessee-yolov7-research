import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torchvision import models, transforms
import matplotlib.pyplot as plt

VAL_IMAGES_DIR = "/home/linuxu/Desktop/Research_Project/yolov7/data/images/val"
VAL_LABELS_DIR = "/home/linuxu/Desktop/Research_Project/yolov7/data/labels/val"

YOLO_PRED_LABELS_DIR = "/home/linuxu/Desktop/Research_Project/yolov7/runs/test/seadronessee_tiled_test/labels"
CNN_WEIGHTS = "/home/linuxu/Desktop/Research_Project/yolov7/cnn_runs/best_cnn.pt"

OUT_DIR = "/home/linuxu/Desktop/Research_Project/yolov7/yolo_cnn_eval"
os.makedirs(OUT_DIR, exist_ok=True)

CLASS_NAMES = ["swimmer", "boat", "jetski", "life_saving_appliances", "buoy"]
NUM_CLASSES = len(CLASS_NAMES)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 128
IOU_MATCH_THRESH = 0.30
PAD_RATIO = 0.10


def read_yolo_txt(path):
    """
    YOLO prediction format: cls cx cy w h conf
    """
    out = []
    if not os.path.isfile(path):
        return out
    with open(path, "r") as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 5:
                continue
            cls = int(float(p[0]))
            cx, cy, w, h = map(float, p[1:5])
            conf = float(p[5]) if len(p) >= 6 else None
            out.append((cls, cx, cy, w, h, conf))
    return out


def yolo_to_xyxy(cx, cy, w, h, W, H):
    x1 = (cx - w / 2) * W
    y1 = (cy - h / 2) * H
    x2 = (cx + w / 2) * W
    y2 = (cy + h / 2) * H
    return x1, y1, x2, y2


def clip_xyxy(x1, y1, x2, y2, W, H):
    x1 = max(0, min(int(round(x1)), W - 1))
    y1 = max(0, min(int(round(y1)), H - 1))
    x2 = max(0, min(int(round(x2)), W - 1))
    y2 = max(0, min(int(round(y2)), H - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def area(b):
    x1, y1, x2, y2 = b
    return max(0, x2 - x1) * max(0, y2 - y1)


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def pad_box(b, W, H, ratio):
    x1, y1, x2, y2 = b
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * ratio), int(bh * ratio)
    return clip_xyxy(x1 - px, y1 - py, x2 + px, y2 + py, W, H)


def find_image(stem):
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        p = os.path.join(VAL_IMAGES_DIR, stem + ext)
        if os.path.isfile(p):
            return p
    return None


cnn = models.mobilenet_v2(weights=None)
cnn.classifier[1] = nn.Linear(cnn.last_channel, NUM_CLASSES)
cnn.load_state_dict(torch.load(CNN_WEIGHTS, map_location=DEVICE))
cnn.to(DEVICE)
cnn.eval()

tfm = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

pred_files = sorted(glob.glob(os.path.join(YOLO_PRED_LABELS_DIR, "*.txt")))
if not pred_files:
    raise RuntimeError(f"No YOLO prediction txt files found in: {YOLO_PRED_LABELS_DIR}")

cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)

total_preds = 0
matched_preds = 0
skipped_no_gt = 0
skipped_no_match = 0

for idx, pred_txt in enumerate(pred_files, start=1):
    stem = Path(pred_txt).stem
    img_path = find_image(stem)
    if img_path is None:
        continue

    img = cv2.imread(img_path)
    if img is None:
        continue
    H, W = img.shape[:2]

    gt_txt = os.path.join(VAL_LABELS_DIR, stem + ".txt")
    gts = read_yolo_txt(gt_txt)
    gt_boxes = []
    for gt_cls, cx, cy, bw, bh, _ in gts:
        b = clip_xyxy(*yolo_to_xyxy(cx, cy, bw, bh, W, H), W, H)
        if b is not None:
            gt_boxes.append((gt_cls, b))

    if not gt_boxes:
        skipped_no_gt += 1
        continue

    preds = read_yolo_txt(pred_txt)
    if not preds:
        continue

    for pred_cls, cx, cy, bw, bh, conf in preds:
        total_preds += 1
        pb = clip_xyxy(*yolo_to_xyxy(cx, cy, bw, bh, W, H), W, H)
        if pb is None:
            continue

        pb = pad_box(pb, W, H, PAD_RATIO)
        if pb is None:
            continue
        x1, y1, x2, y2 = pb
        crop = img[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            continue

        x = tfm(crop).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = cnn(x)
            refined_cls = int(torch.argmax(logits, dim=1).item())

        best_iou = 0.0
        best_gt_cls = None
        for gt_cls, gb in gt_boxes:
            s = iou(pb, gb)
            if s > best_iou:
                best_iou = s
                best_gt_cls = gt_cls

        if best_iou >= IOU_MATCH_THRESH and best_gt_cls is not None:
            cm[best_gt_cls, refined_cls] += 1
            matched_preds += 1
        else:
            skipped_no_match += 1

    if idx % 200 == 0:
        print(f"Progress: {idx}/{len(pred_files)} files processed...")

support = cm.sum(axis=1)
prec = np.zeros(NUM_CLASSES, dtype=float)
rec = np.zeros(NUM_CLASSES, dtype=float)
f1 = np.zeros(NUM_CLASSES, dtype=float)

for c in range(NUM_CLASSES):
    tp = cm[c, c]
    fp = cm[:, c].sum() - tp
    fn = cm[c, :].sum() - tp
    prec[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec[c] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1[c] = (2 * prec[c] * rec[c] / (prec[c] + rec[c])) if (prec[c] + rec[c]) > 0 else 0.0

acc = np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0.0
macro_prec = prec.mean()
macro_rec = rec.mean()
macro_f1 = f1.mean()
weighted_prec = (prec * support).sum() / max(1, support.sum())
weighted_rec = (rec * support).sum() / max(1, support.sum())
weighted_f1 = (f1 * support).sum() / max(1, support.sum())

report_path = os.path.join(OUT_DIR, "pipeline_results.txt")
with open(report_path, "w") as f:
    f.write(f"DEVICE: {DEVICE}\n")
    f.write(f"Total YOLO predictions processed: {total_preds}\n")
    f.write(f"Matched predictions (IoU>={IOU_MATCH_THRESH}): {matched_preds}\n")
    f.write(f"Skipped images with no GT: {skipped_no_gt}\n")
    f.write(f"Skipped predictions with no GT match: {skipped_no_match}\n\n")
    f.write(f"Pipeline Accuracy (matched-only): {acc:.4f}\n\n")
    f.write("Per-class metrics:\n")
    f.write("class,precision,recall,f1,support\n")
    for i, name in enumerate(CLASS_NAMES):
        f.write(f"{name},{prec[i]:.4f},{rec[i]:.4f},{f1[i]:.4f},{support[i]}\n")
    f.write("\n")
    f.write(f"Macro Precision: {macro_prec:.4f}\n")
    f.write(f"Macro Recall: {macro_rec:.4f}\n")
    f.write(f"Macro F1: {macro_f1:.4f}\n\n")
    f.write(f"Weighted Precision: {weighted_prec:.4f}\n")
    f.write(f"Weighted Recall: {weighted_rec:.4f}\n")
    f.write(f"Weighted F1: {weighted_f1:.4f}\n")

np.savetxt(os.path.join(OUT_DIR, "pipeline_confusion_matrix.csv"), cm, fmt="%d", delimiter=",")

plt.figure(figsize=(10, 8))
plt.imshow(cm)
plt.xticks(range(NUM_CLASSES), CLASS_NAMES, rotation=45, ha="right")
plt.yticks(range(NUM_CLASSES), CLASS_NAMES)
plt.title("YOLO + CNN Pipeline Confusion Matrix (Matched Predictions)")
plt.xlabel("Predicted (CNN refined)")
plt.ylabel("True (GT)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "pipeline_confusion_matrix.png"), dpi=200)

print("DONE. Saved to:", OUT_DIR)
print("Key files:")
print(" - pipeline_results.txt")
print(" - pipeline_confusion_matrix.csv")
print(" - pipeline_confusion_matrix.png")

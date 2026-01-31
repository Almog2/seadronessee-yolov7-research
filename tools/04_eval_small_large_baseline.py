import os
import glob
from pathlib import Path

import cv2
import numpy as np

VAL_IMAGES_DIR = "/home/linuxu/Desktop/Research_Project/yolov7/data/images/val"
VAL_LABELS_DIR = "/home/linuxu/Desktop/Research_Project/yolov7/data/labels/val"

PRED_LABELS_DIR = "/home/linuxu/Desktop/Research_Project/yolov7/runs/test/exp_baseline_small_large/labels"

OUT_DIR = "/home/linuxu/Desktop/Research_Project/yolov7/small_large_eval_baseline"
os.makedirs(OUT_DIR, exist_ok=True)

CLASS_NAMES = ["swimmer", "boat", "jetski", "life_saving_appliances", "buoy"]
SMALL_CLASSES = {"swimmer", "life_saving_appliances"}

IOU_THRESH = 0.30


def read_yolo_txt(path):
    """
    Supports both GT (cls cx cy w h) and pred (cls cx cy w h conf)
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


def find_image(stem):
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        p = os.path.join(VAL_IMAGES_DIR, stem + ext)
        if os.path.isfile(p):
            return p
    return None


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


def group_from_class_id(cls_id):
    name = CLASS_NAMES[cls_id]
    return 0 if name in SMALL_CLASSES else 1  # 0=Small, 1=Large


def main():
    pred_files = sorted(glob.glob(os.path.join(PRED_LABELS_DIR, "*.txt")))
    if not pred_files:
        raise RuntimeError(f"No prediction txt files found in: {PRED_LABELS_DIR}")

    cm = np.zeros((2, 2), dtype=int)

    tp = np.zeros(2, dtype=int)
    fp = np.zeros(2, dtype=int)
    fn = np.zeros(2, dtype=int)

    images_used = 0
    total_preds = 0
    total_gts = 0
    skipped_no_img = 0
    skipped_no_gt = 0

    for pf in pred_files:
        stem = Path(pf).stem
        img_path = find_image(stem)
        if img_path is None:
            skipped_no_img += 1
            continue

        img = cv2.imread(img_path)
        if img is None:
            skipped_no_img += 1
            continue
        H, W = img.shape[:2]

        gt_path = os.path.join(VAL_LABELS_DIR, stem + ".txt")
        gt_raw = read_yolo_txt(gt_path)
        if not gt_raw:
            skipped_no_gt += 1
            continue

        pred_raw = read_yolo_txt(pf)
        if not pred_raw:
            for gt_cls, cx, cy, w, h, _ in gt_raw:
                g = group_from_class_id(gt_cls)
                fn[g] += 1
            images_used += 1
            continue

        gts = []
        for gt_cls, cx, cy, bw, bh, _ in gt_raw:
            b = clip_xyxy(*yolo_to_xyxy(cx, cy, bw, bh, W, H), W, H)
            if b is None:
                continue
            gts.append((b, group_from_class_id(gt_cls)))

        preds = []
        for pred_cls, cx, cy, bw, bh, conf in pred_raw:
            b = clip_xyxy(*yolo_to_xyxy(cx, cy, bw, bh, W, H), W, H)
            if b is None:
                continue
            preds.append((b, group_from_class_id(pred_cls)))

        if not gts:
            skipped_no_gt += 1
            continue

        images_used += 1
        total_preds += len(preds)
        total_gts += len(gts)

        if len(preds) == 0:
            for _, g_true in gts:
                fn[g_true] += 1
            continue

        iou_mat = np.zeros((len(preds), len(gts)), dtype=float)
        for i, (pb, _) in enumerate(preds):
            for j, (gb, _) in enumerate(gts):
                iou_mat[i, j] = iou(pb, gb)

        matched_pred = set()
        matched_gt = set()

        while True:
            i, j = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
            best = iou_mat[i, j]
            if best < IOU_THRESH:
                break
            if i in matched_pred or j in matched_gt:
                iou_mat[i, j] = -1.0
                continue

            matched_pred.add(i)
            matched_gt.add(j)

            g_pred = preds[i][1]
            g_true = gts[j][1]
            cm[g_true, g_pred] += 1

            if g_pred == g_true:
                tp[g_true] += 1
            else:
                fp[g_pred] += 1
                fn[g_true] += 1

            iou_mat[i, j] = -1.0

        for i, (_, g_pred) in enumerate(preds):
            if i not in matched_pred:
                fp[g_pred] += 1

        for j, (_, g_true) in enumerate(gts):
            if j not in matched_gt:
                fn[g_true] += 1

    prec = np.zeros(2, dtype=float)
    rec = np.zeros(2, dtype=float)
    f1 = np.zeros(2, dtype=float)

    for g in [0, 1]:
        prec[g] = tp[g] / (tp[g] + fp[g]) if (tp[g] + fp[g]) > 0 else 0.0
        rec[g] = tp[g] / (tp[g] + fn[g]) if (tp[g] + fn[g]) > 0 else 0.0
        f1[g] = (2 * prec[g] * rec[g] / (prec[g] + rec[g])) if (prec[g] + rec[g]) > 0 else 0.0

    labels = ["Small", "Large"]

    report_path = os.path.join(OUT_DIR, "small_large_results.txt")
    with open(report_path, "w") as f:
        f.write(f"IOU_THRESH: {IOU_THRESH}\n")
        f.write(f"Images used: {images_used}\n")
        f.write(f"Total GT boxes: {total_gts}\n")
        f.write(f"Total Pred boxes: {total_preds}\n")
        f.write(f"Skipped (no image): {skipped_no_img}\n")
        f.write(f"Skipped (no GT): {skipped_no_gt}\n\n")

        f.write("Confusion matrix (rows=true, cols=pred):\n")
        f.write("      Small  Large\n")
        f.write(f"Small {cm[0,0]:6d} {cm[0,1]:6d}\n")
        f.write(f"Large {cm[1,0]:6d} {cm[1,1]:6d}\n\n")

        f.write("Group metrics:\n")
        f.write("group,tp,fp,fn,precision,recall,f1\n")
        for g in [0, 1]:
            f.write(f"{labels[g]},{tp[g]},{fp[g]},{fn[g]},{prec[g]:.4f},{rec[g]:.4f},{f1[g]:.4f}\n")

    print("DONE. Saved to:", OUT_DIR)
    print("Key file: small_large_results.txt")
    print("Confusion matrix (rows=true, cols=pred):")
    print("      Small  Large")
    print(f"Small {cm[0,0]:6d} {cm[0,1]:6d}")
    print(f"Large {cm[1,0]:6d} {cm[1,1]:6d}")
    print("Group metrics:")
    for g in [0, 1]:
        print(f"{labels[g]}: TP={tp[g]} FP={fp[g]} FN={fn[g]}  P={prec[g]:.4f} R={rec[g]:.4f} F1={f1[g]:.4f}")


if __name__ == "__main__":
    main()

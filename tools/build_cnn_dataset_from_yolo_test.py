import os
import glob
import cv2
from pathlib import Path

# =========================
# EDIT THESE PATHS
# =========================

VAL_IMAGES_DIR = "/home/linuxu/Desktop/Research_Project/yolov7/data/images/val"
VAL_LABELS_DIR = "/home/linuxu/Desktop/Research_Project/yolov7/data/labels/val"

YOLO_TEST_LABELS_DIR = "/home/linuxu/Desktop/Research_Project/yolov7/runs/test/seadronessee_tiled_test/labels"

OUT_ROOT = "/home/linuxu/Desktop/Research_Project/yolov7/cnn_dataset_from_test"
OUT_IMAGES_DIR = os.path.join(OUT_ROOT, "images")
OUT_META_CSV = os.path.join(OUT_ROOT, "meta.csv")

CLASS_NAMES = ["swimmer", "boat", "jetski", "life_saving_appliances", "buoy"]
NUM_CLASSES = len(CLASS_NAMES)

IOU_MATCH_THRESH = 0.30
MIN_CROP_SIZE = 20
PAD_RATIO = 0.10
MAX_PER_CLASS = 20000  # set None to disable


def read_yolo_labels(path):
    """
    YOLO txt format: cls cx cy w h [conf]
    returns list of tuples (cls, cx, cy, w, h, conf_or_None)
    """
    out = []
    if not os.path.isfile(path):
        return out
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            cx, cy, w, h = map(float, parts[1:5])
            conf = float(parts[5]) if len(parts) >= 6 else None
            out.append((cls, cx, cy, w, h, conf))
    return out


def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return x1, y1, x2, y2


def clip_xyxy(x1, y1, x2, y2, img_w, img_h):
    x1 = max(0, min(int(round(x1)), img_w - 1))
    y1 = max(0, min(int(round(y1)), img_h - 1))
    x2 = max(0, min(int(round(x2)), img_w - 1))
    y2 = max(0, min(int(round(y2)), img_h - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def area_xyxy(b):
    x1, y1, x2, y2 = b
    return max(0, x2 - x1) * max(0, y2 - y1)


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = area_xyxy(a) + area_xyxy(b) - inter
    return inter / union if union > 0 else 0.0


def pad_box(box, img_w, img_h, pad_ratio):
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    px = int(round(bw * pad_ratio))
    py = int(round(bh * pad_ratio))
    nx1 = max(0, x1 - px)
    ny1 = max(0, y1 - py)
    nx2 = min(img_w - 1, x2 + px)
    ny2 = min(img_h - 1, y2 + py)
    if nx2 <= nx1 or ny2 <= ny1:
        return None
    return nx1, ny1, nx2, ny2


def find_image_file(stem):
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        p = os.path.join(VAL_IMAGES_DIR, stem + ext)
        if os.path.isfile(p):
            return p
    return None


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def main():
    ensure_dir(OUT_ROOT)
    ensure_dir(OUT_IMAGES_DIR)

    per_class_count = {i: 0 for i in range(NUM_CLASSES)}
    for i, name in enumerate(CLASS_NAMES):
        ensure_dir(os.path.join(OUT_IMAGES_DIR, name))

    pred_files = sorted(glob.glob(os.path.join(YOLO_TEST_LABELS_DIR, "*.txt")))
    if not pred_files:
        raise FileNotFoundError(f"No prediction txt files found in: {YOLO_TEST_LABELS_DIR}")

    rows = []
    saved = 0
    skipped_no_match = 0
    skipped_small = 0

    for pred_txt in pred_files:
        stem = Path(pred_txt).stem
        img_path = find_image_file(stem)
        if img_path is None:
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        preds = read_yolo_labels(pred_txt)

        gt_txt = os.path.join(VAL_LABELS_DIR, stem + ".txt")
        gts = read_yolo_labels(gt_txt)

        gt_boxes = []
        for (gt_cls, cx, cy, bw, bh, _) in gts:
            gx1, gy1, gx2, gy2 = yolo_to_xyxy(cx, cy, bw, bh, w, h)
            gbox = clip_xyxy(gx1, gy1, gx2, gy2, w, h)
            if gbox is not None and area_xyxy(gbox) > 0:
                gt_boxes.append((gt_cls, gbox))

        if not gt_boxes:
            continue

        for (pred_cls, cx, cy, bw, bh, conf) in preds:
            px1, py1, px2, py2 = yolo_to_xyxy(cx, cy, bw, bh, w, h)
            pbox = clip_xyxy(px1, py1, px2, py2, w, h)
            if pbox is None:
                continue

            best_iou = 0.0
            best_gt_cls = None
            for (gt_cls, gbox) in gt_boxes:
                score = iou(pbox, gbox)
                if score > best_iou:
                    best_iou = score
                    best_gt_cls = gt_cls

            if best_iou < IOU_MATCH_THRESH or best_gt_cls is None:
                skipped_no_match += 1
                continue

            if MAX_PER_CLASS is not None and per_class_count[best_gt_cls] >= MAX_PER_CLASS:
                continue

            padded = pad_box(pbox, w, h, PAD_RATIO)
            if padded is None:
                continue
            x1, y1, x2, y2 = padded
            if (x2 - x1) < MIN_CROP_SIZE or (y2 - y1) < MIN_CROP_SIZE:
                skipped_small += 1
                continue

            crop = img[y1:y2, x1:x2]
            if crop is None or crop.size == 0:
                continue

            class_name = CLASS_NAMES[best_gt_cls]
            out_name = f"{stem}_c{saved:07d}.jpg"
            out_path = os.path.join(OUT_IMAGES_DIR, class_name, out_name)
            cv2.imwrite(out_path, crop)

            rows.append([
                out_path,
                str(best_gt_cls),
                class_name,
                stem,
                f"{conf:.4f}" if conf is not None else "",
                f"{best_iou:.4f}",
            ])

            per_class_count[best_gt_cls] += 1
            saved += 1

    with open(OUT_META_CSV, "w") as f:
        f.write("crop_path,label_id,label_name,source_image,conf,iou_to_gt\n")
        for r in rows:
            f.write(",".join(r) + "\n")

    print("Done.")
    print(f"Crops saved: {saved}")
    print(f"Skipped (no GT match): {skipped_no_match}")
    print(f"Skipped (too small): {skipped_small}")
    print("Per-class counts:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {per_class_count[i]}")
    print(f"Meta CSV: {OUT_META_CSV}")
    print(f"Dataset root: {OUT_ROOT}")


if __name__ == "__main__":
    main()

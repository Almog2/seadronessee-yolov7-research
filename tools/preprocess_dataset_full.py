import cv2
import random
import shutil
from pathlib import Path

# =========================
# USER CONFIG (YOUR PATHS)
# =========================
SRC_DATASET_ROOT = r"/home/linuxu/Desktop/Research_Project/yolov7/data"
DST_DATASET_ROOT = r"/home/linuxu/Desktop/Research_Project/yolov7/data_preprocessed_full"

SPLITS = ["train", "val"]

# Output image format
OUT_IMG_EXT = ".jpg"

# --- CLAHE settings ---
APPLY_CLAHE = True
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# --- HSV Jitter settings (TRAIN only) ---
APPLY_HSV_JITTER = True
HSV_JITTER_SPLITS = ["train"]
H_GAIN = 0.015
S_GAIN = 0.7
V_GAIN = 0.4

# --- Tiling (SAHI-like) settings ---
APPLY_TILING = True
TILE_SIZE = 640
TILE_OVERLAP = 0.20
MIN_BBOX_AREA_PX = 10
MIN_BBOX_W_PX = 2
MIN_BBOX_H_PX = 2

RNG_SEED = 42


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def list_images(images_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted([p for p in images_dir.iterdir() if p.suffix.lower() in exts])


def read_yolo_labels(label_path: Path):
    if not label_path.exists():
        return []
    rows = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            cls = int(float(parts[0]))
            cx, cy, w, h = map(float, parts[1:])
            rows.append((cls, cx, cy, w, h))
    return rows


def yolo_to_xyxy(lbl, img_w, img_h):
    cls, cx, cy, w, h = lbl
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return cls, x1, y1, x2, y2


def xyxy_to_yolo(cls, x1, y1, x2, y2, img_w, img_h):
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    cx = x1 + w / 2
    cy = y1 + h / 2
    return (cls, cx / img_w, cy / img_h, w / img_w, h / img_h)


def clip_box(x1, y1, x2, y2, clip_w, clip_h):
    x1c = max(0.0, min(float(clip_w), x1))
    y1c = max(0.0, min(float(clip_h), y1))
    x2c = max(0.0, min(float(clip_w), x2))
    y2c = max(0.0, min(float(clip_h), y2))
    return x1c, y1c, x2c, y2c


def apply_clahe_bgr(img_bgr, clip_limit, tile_grid_size):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


def apply_hsv_jitter_bgr(img_bgr, hgain, sgain, vgain, rng: random.Random):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype("float32")

    dh = rng.uniform(-1, 1) * hgain * 180
    ds = 1 + rng.uniform(-1, 1) * sgain
    dv = 1 + rng.uniform(-1, 1) * vgain

    hsv[..., 0] = (hsv[..., 0] + dh) % 180
    hsv[..., 1] = (hsv[..., 1] * ds).clip(0, 255)
    hsv[..., 2] = (hsv[..., 2] * dv).clip(0, 255)

    return cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2BGR)


def compute_tile_starts(full_len, tile_len, overlap_ratio):
    if tile_len >= full_len:
        return [0]
    stride = int(tile_len * (1.0 - overlap_ratio))
    stride = max(1, stride)
    starts = list(range(0, full_len - tile_len + 1, stride))
    last = full_len - tile_len
    if starts[-1] != last:
        starts.append(last)
    return starts


def make_tiles_and_labels(img_bgr, labels_yolo, tile_size, overlap_ratio):
    H, W = img_bgr.shape[:2]
    labels_xyxy = [yolo_to_xyxy(l, W, H) for l in labels_yolo]

    xs = compute_tile_starts(W, tile_size, overlap_ratio)
    ys = compute_tile_starts(H, tile_size, overlap_ratio)

    tiles = []
    for ty in ys:
        for tx in xs:
            tile = img_bgr[ty:ty + tile_size, tx:tx + tile_size]
            th, tw = tile.shape[:2]

            tile_labels = []
            for cls, x1, y1, x2b, y2b in labels_xyxy:
                sx1, sy1 = x1 - tx, y1 - ty
                sx2, sy2 = x2b - tx, y2b - ty

                cx1, cy1, cx2, cy2 = clip_box(sx1, sy1, sx2, sy2, tw, th)

                bw = cx2 - cx1
                bh = cy2 - cy1
                area = bw * bh
                if bw < MIN_BBOX_W_PX or bh < MIN_BBOX_H_PX or area < MIN_BBOX_AREA_PX:
                    continue

                tile_labels.append(xyxy_to_yolo(cls, cx1, cy1, cx2, cy2, tw, th))

            tiles.append((tile, tile_labels, (tx, ty, tw, th)))
    return tiles


def write_yolo_labels(label_path: Path, labels_yolo):
    ensure_dir(label_path.parent)
    with open(label_path, "w", encoding="utf-8") as f:
        for cls, cx, cy, w, h in labels_yolo:
            f.write(f"{int(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def copy_yaml_files(src_root: Path, dst_root: Path):
    for p in src_root.glob("*.yaml"):
        shutil.copy2(p, dst_root / p.name)


def process_split(split: str, rng: random.Random):
    src_images_dir = Path(SRC_DATASET_ROOT) / "images" / split
    src_labels_dir = Path(SRC_DATASET_ROOT) / "labels" / split

    dst_images_dir = Path(DST_DATASET_ROOT) / "images" / split
    dst_labels_dir = Path(DST_DATASET_ROOT) / "labels" / split

    ensure_dir(dst_images_dir)
    ensure_dir(dst_labels_dir)

    images = list_images(src_images_dir)
    if not images:
        raise FileNotFoundError(f"No images found in: {src_images_dir}")

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        label_path = src_labels_dir / (img_path.stem + ".txt")
        labels = read_yolo_labels(label_path)

        # 1) CLAHE
        if APPLY_CLAHE:
            img = apply_clahe_bgr(img, CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE)

        # 2) HSV jitter (train only)
        if APPLY_HSV_JITTER and split in HSV_JITTER_SPLITS:
            img = apply_hsv_jitter_bgr(img, H_GAIN, S_GAIN, V_GAIN, rng)

        # 3) SAHI-like tiling
        if APPLY_TILING:
            tiles = make_tiles_and_labels(img, labels, TILE_SIZE, TILE_OVERLAP)
            for idx, (tile_img, tile_labels, (tx, ty, tw, th)) in enumerate(tiles):
                out_name = f"{img_path.stem}_tile_{ty}_{tx}_{idx}{OUT_IMG_EXT}"
                out_img_path = dst_images_dir / out_name
                out_lbl_path = dst_labels_dir / (Path(out_name).stem + ".txt")

                cv2.imwrite(str(out_img_path), tile_img)
                write_yolo_labels(out_lbl_path, tile_labels)
        else:
            out_img_path = dst_images_dir / (img_path.stem + OUT_IMG_EXT)
            out_lbl_path = dst_labels_dir / (img_path.stem + ".txt")
            cv2.imwrite(str(out_img_path), img)
            write_yolo_labels(out_lbl_path, labels)


def main():
    rng = random.Random(RNG_SEED)

    dst_root = Path(DST_DATASET_ROOT)
    ensure_dir(dst_root / "images")
    ensure_dir(dst_root / "labels")

    copy_yaml_files(Path(SRC_DATASET_ROOT), dst_root)

    for split in SPLITS:
        process_split(split, rng)

    print("Done.")
    print(f"Preprocessed dataset saved to: {DST_DATASET_ROOT}")
    print("Next step: create/update a YAML to point YOLO to this new dataset root.")


if __name__ == "__main__":
    main()

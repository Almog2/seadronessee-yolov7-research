import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt

DATASET_ROOT = "/home/linuxu/Desktop/Research_Project/yolov7/cnn_dataset_from_test/images"
MODEL_WEIGHTS = "/home/linuxu/Desktop/Research_Project/yolov7/cnn_runs/best_cnn.pt"
OUT_DIR = "/home/linuxu/Desktop/Research_Project/yolov7/cnn_eval"

CLASS_NAMES = ["swimmer", "boat", "jetski", "life_saving_appliances", "buoy"]
NUM_CLASSES = len(CLASS_NAMES)

IMG_SIZE = 128
BATCH_SIZE = 64
SEED = 42
VAL_SPLIT = 0.20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(SEED)
torch.manual_seed(SEED)

tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

ds = datasets.ImageFolder(DATASET_ROOT, transform=tfms)

folder_to_idx = ds.class_to_idx
desired = {name: i for i, name in enumerate(CLASS_NAMES)}
for name in CLASS_NAMES:
    if name not in folder_to_idx:
        raise RuntimeError(f"Missing class folder: {name} in {DATASET_ROOT}")

remapped_samples = []
for path, _ in ds.samples:
    folder = os.path.basename(os.path.dirname(path))
    remapped_samples.append((path, desired[folder]))
ds.samples = remapped_samples
ds.targets = [y for _, y in remapped_samples]

n_total = len(ds)
n_val = int(n_total * VAL_SPLIT)
n_train = n_total - n_val
train_ds, val_ds = random_split(ds, [n_train, n_val])

val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"Dataset total={n_total} | eval split={n_val} | DEVICE={DEVICE}")

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
model.to(DEVICE)

state = torch.load(MODEL_WEIGHTS, map_location=DEVICE)
model.load_state_dict(state)
model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for x, y in val_loader:
        x = x.to(DEVICE)
        logits = model(x)
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        y_pred.extend(pred.tolist())
        y_true.extend(y.numpy().tolist())

y_true = np.array(y_true, dtype=int)
y_pred = np.array(y_pred, dtype=int)

cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
for t, p in zip(y_true, y_pred):
    cm[t, p] += 1

prec = np.zeros(NUM_CLASSES, dtype=float)
rec = np.zeros(NUM_CLASSES, dtype=float)
f1 = np.zeros(NUM_CLASSES, dtype=float)
support = cm.sum(axis=1)

for c in range(NUM_CLASSES):
    tp = cm[c, c]
    fp = cm[:, c].sum() - tp
    fn = cm[c, :].sum() - tp

    prec[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec[c] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1[c] = (2 * prec[c] * rec[c] / (prec[c] + rec[c])) if (prec[c] + rec[c]) > 0 else 0.0

acc = (np.trace(cm) / np.sum(cm)) if np.sum(cm) > 0 else 0.0
macro_prec = prec.mean()
macro_rec = rec.mean()
macro_f1 = f1.mean()

weighted_prec = (prec * support).sum() / max(1, support.sum())
weighted_rec = (rec * support).sum() / max(1, support.sum())
weighted_f1 = (f1 * support).sum() / max(1, support.sum())

np.savetxt(os.path.join(OUT_DIR, "cnn_confusion_matrix.csv"), cm, fmt="%d", delimiter=",")

with open(os.path.join(OUT_DIR, "cnn_results.txt"), "w") as f:
    f.write(f"DEVICE: {DEVICE}\n")
    f.write(f"Total eval samples: {len(y_true)}\n")
    f.write(f"Accuracy: {acc:.4f}\n\n")
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

plt.figure(figsize=(10, 8))
plt.imshow(cm)
plt.xticks(range(NUM_CLASSES), CLASS_NAMES, rotation=45, ha="right")
plt.yticks(range(NUM_CLASSES), CLASS_NAMES)
plt.title("CNN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "cnn_confusion_matrix.png"), dpi=200)

print("DONE. Saved to:", OUT_DIR)
print("Key files:")
print(" - cnn_results.txt")
print(" - cnn_confusion_matrix.csv")
print(" - cnn_confusion_matrix.png")

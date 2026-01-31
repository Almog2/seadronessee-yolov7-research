import os
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import datasets, transforms, models

DATASET_ROOT = "/home/linuxu/Desktop/Research_Project/yolov7/cnn_dataset_from_test/images"
OUT_DIR = "/home/linuxu/Desktop/Research_Project/yolov7/cnn_runs"
MODEL_PATH = os.path.join(OUT_DIR, "best_cnn.pt")

CLASS_NAMES = ["swimmer", "boat", "jetski", "life_saving_appliances", "buoy"]
NUM_CLASSES = len(CLASS_NAMES)

SEED = 42
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3
IMG_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_SPLIT = 0.8

random.seed(SEED)
torch.manual_seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

train_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
    transforms.ToTensor(),
])

val_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

full_dataset = datasets.ImageFolder(DATASET_ROOT, transform=train_tfms)

folder_to_idx = full_dataset.class_to_idx
desired = {name: i for i, name in enumerate(CLASS_NAMES)}
for name in CLASS_NAMES:
    if name not in folder_to_idx:
        raise RuntimeError(f"Missing class folder: {name} in {DATASET_ROOT}")

remapped_samples = []
for path, _ in full_dataset.samples:
    folder_name = Path(path).parent.name
    remapped_samples.append((path, desired[folder_name]))
full_dataset.samples = remapped_samples
full_dataset.targets = [y for _, y in remapped_samples]

n_total = len(full_dataset)
n_train = int(n_total * TRAIN_SPLIT)
n_val = n_total - n_train
train_ds, val_ds = random_split(full_dataset, [n_train, n_val])

val_ds.dataset = datasets.ImageFolder(DATASET_ROOT, transform=val_tfms)
val_ds.dataset.samples = remapped_samples
val_ds.dataset.targets = [y for _, y in remapped_samples]

train_indices = train_ds.indices
train_targets = [full_dataset.targets[i] for i in train_indices]

class_counts = [0] * NUM_CLASSES
for t in train_targets:
    class_counts[t] += 1

class_weights = [1.0 / max(1, c) for c in class_counts]
sample_weights = [class_weights[t] for t in train_targets]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print("Dataset sizes:", n_train, n_val)
print("Train class counts:", class_counts)

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_val_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in train_loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    train_loss = total_loss / max(1, total)
    train_acc = correct / max(1, total)

    model.eval()
    v_correct = 0
    v_total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            v_correct += (preds == y).sum().item()
            v_total += y.size(0)

    val_acc = v_correct / max(1, v_total)

    print(f"Epoch {epoch}/{EPOCHS} | train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_PATH)

print("Training complete.")
print("Best val_acc:", best_val_acc)
print("Saved best model to:", MODEL_PATH)

"""
Task 2 — Dog Breed Identification
Model: ResNet18 with Transfer Learning & Fine-tuning
Dataset: Stanford Dogs Dataset (120 breeds)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import os
import urllib.request
import tarfile
from PIL import Image
from pathlib import Path

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
BATCH_SIZE   = 32
EPOCHS       = 15
LR_HEAD      = 0.001   # learning rate for new FC layer
LR_BACKBONE  = 0.0001  # lower LR for pretrained layers
NUM_CLASSES  = 120
IMG_SIZE     = 224     # ResNet expects 224×224
VAL_SPLIT    = 0.2
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# ─────────────────────────────────────────────
# 1. Download Stanford Dogs Dataset
# ─────────────────────────────────────────────
DATA_DIR   = Path("./stanford_dogs")
IMAGES_DIR = DATA_DIR / "Images"

def download_stanford_dogs():
    DATA_DIR.mkdir(exist_ok=True)
    url   = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
    tfile = DATA_DIR / "images.tar"

    if not IMAGES_DIR.exists():
        print("Downloading Stanford Dogs Dataset (~750 MB)...")
        urllib.request.urlretrieve(url, tfile)
        print("Extracting...")
        with tarfile.open(tfile) as t:
            t.extractall(DATA_DIR)
        tfile.unlink()
        print("Dataset ready!")
    else:
        print("Dataset already downloaded.")

download_stanford_dogs()

# ─────────────────────────────────────────────
# 2. Custom Dataset Class
# ─────────────────────────────────────────────
class StanfordDogsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir  = Path(root_dir)
        self.transform = transform
        self.samples   = []
        self.classes   = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        for cls in self.classes:
            cls_dir = self.root_dir / cls
            for img_path in cls_dir.glob("*.jpg"):
                self.samples.append((img_path, self.class_to_idx[cls]))

        print(f"Found {len(self.samples)} images across {len(self.classes)} breeds.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# ─────────────────────────────────────────────
# 3. Transforms & DataLoaders
# ─────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],   # ImageNet mean
                         [0.229, 0.224, 0.225]),   # ImageNet std
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

full_dataset = StanfordDogsDataset(IMAGES_DIR, transform=train_transform)

val_size   = int(VAL_SPLIT * len(full_dataset))
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Apply val transform to validation split
val_dataset.dataset = StanfordDogsDataset(IMAGES_DIR, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print(f"Training samples  : {train_size:,}")
print(f"Validation samples: {val_size:,}")

# ─────────────────────────────────────────────
# 4. ResNet18 with Transfer Learning
# ─────────────────────────────────────────────
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Freeze all backbone layers first
for param in model.parameters():
    param.requires_grad = False

# Replace final FC layer for 120 dog breeds
in_features    = model.fc.in_features
model.fc       = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, NUM_CLASSES)
)

model = model.to(DEVICE)

total_params    = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters    : {total_params:,}")
print(f"Trainable parameters: {trainable_params:,} (FC head only)")

# ─────────────────────────────────────────────
# 5. Loss, Optimizer & Scheduler
# ─────────────────────────────────────────────
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Only train the FC head initially
optimizer = optim.Adam(model.fc.parameters(), lr=LR_HEAD, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# ─────────────────────────────────────────────
# 6. Training & Evaluation Functions
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted  = outputs.max(1)
        correct       += predicted.eq(labels).sum().item()
        total         += labels.size(0)

    return running_loss / total, 100.0 * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted  = outputs.max(1)
            correct       += predicted.eq(labels).sum().item()
            total         += labels.size(0)

    return running_loss / total, 100.0 * correct / total

# ─────────────────────────────────────────────
# 7. Phase 1 — Train FC Head Only (5 epochs)
# ─────────────────────────────────────────────
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
best_val_acc = 0.0

PHASE1_EPOCHS = 5
print("\n── Phase 1: Training FC Head ────────────────────────────")
print(f"{'Epoch':>6} {'Train Loss':>11} {'Train Acc':>10} {'Val Loss':>9} {'Val Acc':>8}")
print("─" * 55)

for epoch in range(1, PHASE1_EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
    val_loss,   val_acc   = evaluate(model, val_loader, criterion, DEVICE)
    scheduler.step()

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(f"{epoch:>6} {train_loss:>11.4f} {train_acc:>9.2f}% {val_loss:>9.4f} {val_acc:>7.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_dog_breed_model.pth")

# ─────────────────────────────────────────────
# 8. Phase 2 — Fine-tune Full Network (10 epochs)
# ─────────────────────────────────────────────
# Unfreeze all layers
for param in model.parameters():
    param.requires_grad = True

# Use differential learning rates
optimizer = optim.Adam([
    {"params": model.fc.parameters(),  "lr": LR_HEAD},
    {"params": [p for name, p in model.named_parameters()
                if "fc" not in name],  "lr": LR_BACKBONE},
], weight_decay=1e-4)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - PHASE1_EPOCHS)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n── Phase 2: Fine-tuning Full Network ({trainable_params:,} params) ──")
print(f"{'Epoch':>6} {'Train Loss':>11} {'Train Acc':>10} {'Val Loss':>9} {'Val Acc':>8}")
print("─" * 55)

for epoch in range(PHASE1_EPOCHS + 1, EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
    val_loss,   val_acc   = evaluate(model, val_loader, criterion, DEVICE)
    scheduler.step()

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(f"{epoch:>6} {train_loss:>11.4f} {train_acc:>9.2f}% {val_loss:>9.4f} {val_acc:>7.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_dog_breed_model.pth")

print(f"\n✓ Best validation accuracy : {best_val_acc:.2f}%")
print("✓ Model saved to best_dog_breed_model.pth")

# ─────────────────────────────────────────────
# 9. Training Curves
# ─────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history["train_loss"], label="Train")
ax1.plot(history["val_loss"],   label="Validation")
ax1.axvline(x=PHASE1_EPOCHS - 1, color="gray", linestyle="--", label="Fine-tune start")
ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.plot(history["train_acc"], label="Train")
ax2.plot(history["val_acc"],   label="Validation")
ax2.axvline(x=PHASE1_EPOCHS - 1, color="gray", linestyle="--", label="Fine-tune start")
ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
ax2.legend(); ax2.grid(True, alpha=0.3)

plt.suptitle("Dog Breed Identification — ResNet18 Transfer Learning", fontsize=12)
plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.show()
print("Training curves saved!")

# ─────────────────────────────────────────────
# 10. Sample Predictions
# ─────────────────────────────────────────────
model.load_state_dict(torch.load("best_dog_breed_model.pth", map_location=DEVICE))
model.eval()

classes     = full_dataset.classes
dataiter    = iter(val_loader)
images, labels = next(dataiter)

with torch.no_grad():
    outputs  = model(images.to(DEVICE))
    _, preds = outputs.max(1)

def imshow(img):
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = img.numpy().transpose(1, 2, 0)
    img  = std * img + mean
    return np.clip(img, 0, 1)

fig, axes = plt.subplots(2, 6, figsize=(18, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(imshow(images[i]))
    true_cls = classes[labels[i]].split("-")[-1].replace("_", " ")
    pred_cls = classes[preds[i].cpu()].split("-")[-1].replace("_", " ")
    color    = "green" if labels[i] == preds[i].cpu() else "red"
    ax.set_title(f"T: {true_cls}\nP: {pred_cls}", fontsize=7, color=color)
    ax.axis("off")

plt.suptitle("Dog Breed Predictions (green=correct, red=wrong)", fontsize=11)
plt.tight_layout()
plt.savefig("sample_predictions.png", dpi=150)
plt.show()
print("Sample predictions saved!")

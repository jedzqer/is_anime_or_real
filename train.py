import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import os
from collections import Counter
from torch.optim.lr_scheduler import StepLR

# ---------------------
# ⚙️ 1. Hyperparameter Configuration
# ---------------------
batch_size = 32
lr = 5e-6  # Lower learning rate, more suitable for fine-tuning
epochs = 10
img_size = 224
patience = 3  # Tolerance rounds for EarlyStopping

# ---------------------
# 🎨 2. Data Augmentation and Preprocessing
# ---------------------
train_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(180),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------------------
# 📦 3. Load and Split Dataset into Training/Validation
# ---------------------
dataset = datasets.ImageFolder('dataset', transform=train_transform)
train_len = int(0.8 * len(dataset))
val_len = len(dataset) - train_len
train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

# Manually set the validation set transform
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Class indices
class_names = dataset.classes
print("📂 Classes:", class_names)
print("📊 Class indices:", dataset.class_to_idx)

# ---------------------
# 🧠 4. Load Pretrained Model
# ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
# Create a new pretrained model
model = models.resnet18(pretrained=True)
# Replace the last fully connected layer to adapt to the 2-class task
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)
"""

# Load ResNet18 and replace the last layer with 2-class output
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('resnet18_anime_real.pth', map_location=device))
model.to(device)

# ---------------------
# 🧪 5. Calculate Class Weights
# ---------------------
targets = [label for _, label in dataset.samples]
count = Counter(targets)
print("📈 Class sample counts:", count)

# Note: The class order is consistent with class_to_idx (0=real, 1=anime)
weights = [1.0 / count[i] for i in range(len(count))]
weights = torch.tensor(weights).to(device)

# ---------------------
# 🔧 6. Loss Function, Optimizer, and Learning Rate Scheduler
# ---------------------
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

# ---------------------
# 🚀 7. Training and Validation Loop with EarlyStopping
# ---------------------
best_acc = 0
counter = 0

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Validation phase
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = 100 * correct / total
    avg_train_loss = running_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)

    print(f"📊 Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    # 🧠 EarlyStopping + Save Best Model
    if val_acc > best_acc:
        best_acc = val_acc
        counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
        print("💾 New best model saved!")
    else:
        counter += 1
        if counter >= patience:
            print("🛑 Training stopped early, validation performance no longer improving.")
            break

    scheduler.step()

print("🎉 Training completed, best model saved!")    

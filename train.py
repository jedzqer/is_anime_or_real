import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import os

# ⚙️ 1. Configure hyperparameters
batch_size = 32
lr = 0.0005
epochs = 3
img_size = 224

# 🎨 2. Data augmentation and preprocessing
train_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
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

# 📦 3. Load dataset and split into training/validation sets
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

# 🧠 4. Load pre-trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
# Create a new pre-trained model
model = models.resnet18(pretrained=True)
# Replace the final fully connected layer for 2-class task
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)
"""

# Load ResNet18 and replace the last layer with 2-class output
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('resnet18_anime_real.pth', map_location=device))
model.to(device)

# 🔧 5. Loss function & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# 🚀 6. Training and validation loop
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

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = 100 * correct / total
    print(f"📊 Epoch [{epoch+1}/{epochs}] | Loss: {running_loss:.4f} | Val Accuracy: {val_acc:.2f}%")

# 💾 7. Save the model
torch.save(model.state_dict(), 'resnet18_anime_real.pth')
print("✅ Model saved!")
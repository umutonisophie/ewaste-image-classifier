import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# 1. Configuration & Paths
DATA_DIR = "./MASTER_e_waste_visual_dataset"
BATCH_SIZE = 16  # Larger batch for efficiency
EPOCHS = 15  # Number of full passes over dataset
LEARNING_RATE = 0.0005  # Smaller LR for stability
MODEL_SAVE_PATH = "ewaste_mobilenet_master_model.pth"

# 2. Advanced Image Augmentation (Simulates real-world phone camera conditions)
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),  # Handles different phone angles
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Mimics varied lighting
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. Custom Dataset that combines all device categories
class MasterImageFolder(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Collect non-empty category folders
        self.classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d)) and len(os.listdir(os.path.join(root_dir, d))) > 0
        ])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # Gather all image paths and labels
        self.samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for file_name in os.listdir(cls_dir):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    self.samples.append((os.path.join(cls_dir, file_name), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, label
        except Exception:
            # In case of corrupt images, return a tensor of zeros
            return torch.zeros(3, 224, 224), label

print("⚙️ Scanning and combining all device categories into a unified dataset...")
master_dataset = MasterImageFolder(DATA_DIR, transform=train_transforms)
dataloader = DataLoader(master_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# Save category labels for reference
with open("labels.txt", "w") as f:
    for name in master_dataset.classes:
        f.write(f"{name}\n")

print(f"📊 Dataset stats: {len(master_dataset.classes)} categories locked.")
print(f"📈 Total images for training: {len(master_dataset)}\n")

# 4. Initialize MobileNetV3
print("🧠 Initializing MobileNetV3 architecture...")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Training will run on: {device.type.upper()}")

weights = models.MobileNet_V3_Large_Weights.DEFAULT
model = models.mobilenet_v3_large(weights=weights)

# Replace the classifier head to match number of categories
num_features = model.classifier[0].in_features
model.classifier = nn.Sequential(
    nn.Linear(num_features, 1280),
    nn.Hardswish(),
    nn.Dropout(p=0.3),
    nn.Linear(1280, len(master_dataset.classes))
)
model = model.to(device)

# 5. Define Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 6. Training Loop
print("\n🚀 Starting training on the master dataset...")
print("────────────────────────────────────────────────────────────────")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(master_dataset)
    epoch_acc = running_corrects.double() / len(master_dataset)
    print(f"Epoch {epoch + 1}/{EPOCHS} -> Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc * 100:.2f}%")

print("────────────────────────────────────────────────────────────────")
# 7. Save the trained model weights
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"🎉 Training complete! Master model saved as: '{MODEL_SAVE_PATH}'")
print("📄 Category labels saved in 'labels.txt'")
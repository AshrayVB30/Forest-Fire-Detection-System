import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
data_dir = os.path.join(project_root, "data", "Image_data", "ForestFireDataset", "train")
model_dir = os.path.join(current_dir, "models")
os.makedirs(model_dir, exist_ok=True)

print(f"Data directory: {data_dir}")
print(f"Model directory: {model_dir}")

# Check dataset exists
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Data directory not found: {data_dir}")

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Dataset & Loader
dataset = datasets.ImageFolder(data_dir, transform=transform)
print(f"Dataset classes: {dataset.classes}")
print(f"Dataset size: {len(dataset)}")

dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Model: Transfer Learning (ResNet18)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: fire, nofire
model.to(device)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss, correct = 0.0, 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    avg_loss = running_loss / len(dataloader)
    acc = correct / len(dataset)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Acc: {acc:.4f}")

# Save model
model_path = os.path.join(model_dir, "fire_detection_resnet18.pth")
torch.save(model.state_dict(), model_path)
print(f"✅ Model saved to: {model_path}")

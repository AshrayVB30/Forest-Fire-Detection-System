import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# ---------------- CONFIG ----------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# Dataset paths
train_dir = os.path.join(project_root, "data", "Image_data", "ForestFireDataset", "train")
test_dir = os.path.join(project_root, "data", "Image_data", "ForestFireDataset", "test")

model_dir = os.path.join(current_dir, "models")
os.makedirs(model_dir, exist_ok=True)

print(f"Train dir: {train_dir}")
print(f"Test dir:  {test_dir}")
print(f"Model dir: {model_dir}")

# ---------------- MAIN ----------------
def main():
    # ---------------- DATASET ----------------
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"❌ Train data not found at {train_dir}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Full dataset
    full_dataset = datasets.ImageFolder(train_dir, transform=transform)
    print(f"Classes: {full_dataset.classes}")
    print(f"Total images: {len(full_dataset)}")

    # Train/Validation Split
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)  # 🔑 num_workers=0 on Windows
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # ---------------- MODEL ----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)  # fire / no fire
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # ---------------- TRAINING ----------------
    epochs = 10
    best_acc = 0.0
    model_path = os.path.join(model_dir, "fire_detection_resnet18.pth")

    for epoch in range(epochs):
        # Training
        model.train()
        running_loss, correct = 0.0, 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

        train_acc = correct / train_size
        avg_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(f"✅ Saved best model (Val Acc: {val_acc:.4f})")

    print(f"🎉 Training finished. Best Val Acc: {best_acc:.4f}")
    print(f"📌 Model saved to {model_path}")


if __name__ == "__main__":
    main()

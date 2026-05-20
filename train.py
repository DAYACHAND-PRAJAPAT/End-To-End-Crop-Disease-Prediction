import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
import os

from model import PlantDiseaseResNet18
from utils import (
    train_transform,
    val_transform,
    PLANT_DISEASE_CLASSES
)


def train(
    data_dir="data",
    epochs=5,
    batch_size=16,
    lr=0.001,
    save_path="artifacts/model/plant_disease_model.pth"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    train_ds = datasets.ImageFolder(
        root=os.path.join(data_dir, "train"),
        transform=train_transform
    )

    val_ds = datasets.ImageFolder(
        root=os.path.join(data_dir, "val"),
        transform=val_transform
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = PlantDiseaseResNet18(num_classes=len(PLANT_DISEASE_CLASSES)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0

    for epoch in range(epochs):

        # --- Training phase ---
        model.train()
        correct = 0
        total = 0

        pbar = tqdm(train_loader)
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        print(f"Epoch {epoch+1} Train Accuracy: {train_acc:.2f}%")

        # --- Validation phase ---
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, preds = torch.max(outputs, 1)

                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100 * val_correct / val_total
        print(f"Validation Accuracy: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    'model_state': model.state_dict(),
                    'val_acc': val_acc,
                },
                save_path
            )
            print("Best model saved")


if __name__ == "__main__":
    train()
import torch
import torch.nn as nn
from tqdm import tqdm
import os

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, device='cuda', num_epochs=20, save_path='best_model.pth'):
    model.to(device)
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")

        for x, y in loop:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            loop.set_postfix(loss=loss.item(), acc=100. * correct / total)

        val_acc = evaluate_model(model, val_loader, device)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f" Saved best model with val acc: {val_acc:.2f}%")

        if scheduler:
            scheduler.step()

    print(f"\n Training complete. Best Val Accuracy: {best_val_acc:.2f}%")
    return best_val_acc

def evaluate_model(model, loader, device='cuda'):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    acc = 100. * correct / total
    print(f"🔍 Validation Accuracy: {acc:.2f}%")
    return acc

# Optional CLI entry point
if __name__ == "__main__":
    from model import TinyAudioTransformer
    from dataset import get_dataloaders  # your custom loaders
    from torch.optim import Adam

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, val_loader = get_dataloaders()
    model = TinyAudioTransformer(num_classes=8)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-4)

    train_model(model, train_loader, val_loader, criterion, optimizer, device=device)
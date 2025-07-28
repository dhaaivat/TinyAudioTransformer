import torch
import torch.nn as nn
from tqdm import tqdm
import os

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None,
                device='cuda', num_epochs=20, save_path='best_model.pth',
                patience=5, delta=0.0):
    
    model.to(device)
    best_val_acc = 0.0
    best_model_state = None
    epochs_no_improve = 0

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

        # Check for improvement
        if val_acc > best_val_acc + delta:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            epochs_no_improve = 0
            torch.save(best_model_state, save_path)
            print(f" Saved new best model (val acc: {val_acc:.2f}%)")
        else:
            epochs_no_improve += 1
            print(f" No improvement for {epochs_no_improve} epoch(s)")

        if scheduler:
            scheduler.step()

        # Early stopping check
        if epochs_no_improve >= patience:
            print(f"\n Early stopping at epoch {epoch+1}")
            break

    print(f"\n Training complete. Best Val Accuracy: {best_val_acc:.2f}%")
    return best_val_acc

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

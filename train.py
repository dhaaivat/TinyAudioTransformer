import torch
import torch.nn as nn
from tqdm import tqdm
import os
from audio_augmentations import get_waveform_augmentations  
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None,
                device='cuda', num_epochs=20, save_path='best_model.pth',
                patience=5, delta=0.0, switch_epoch=6):
    
    model.to(device)
    best_val_acc = 0.0
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        # Warm start logic: activate augmentations after `switch_epoch`
        if epoch == switch_epoch:
            print(f"[INFO] Switching to augmented data from epoch {epoch+1} onwards...")
            if hasattr(train_loader.dataset, "augment"):
                train_loader.dataset.augment = True
                train_loader.dataset.waveform_augment = get_waveform_augmentations(
                    sample_rate=train_loader.dataset.sample_rate
                )

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

        if val_acc > best_val_acc + delta:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            epochs_no_improve = 0
            torch.save(best_model_state, save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[Early Stopping] No improvement in {patience} epochs.")
                break
              
        if scheduler:
            scheduler.step()

        # Early stopping check
        if epochs_no_improve >= patience:
            print(f"\n Early stopping at epoch {epoch+1}")
            break

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
  


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, val_loader = get_dataloaders()
    


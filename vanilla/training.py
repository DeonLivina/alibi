import torch
import torch.nn as nn
from tqdm import tqdm
from dataloader import get_dataloaders
from witness_loader import get_all_dataloaders
from model import GWMamba

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

# for just strain
#train_loader, val_loader, test_loader = get_dataloaders(batch_size=8)

# for strain + witness
train_loader, val_loader, test_loader = get_all_dataloaders(batch_size=8)

model     = GWMamba(n_classes=4).to(device) # use n_classes = 4 for strain only and n_classes = 8 for strain and withness
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()


def train_epoch(model, loader, epoch, epochs):
    model.train()
    total_loss = 0.0
    correct    = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1:02d}/{epochs} [Train]", leave=False)

    for x_batch, y_batch in pbar:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        pred = model(x_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct    += (pred.argmax(dim=1) == y_batch).sum().item()

        # update bar with current loss
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    accuracy = correct   / len(loader.dataset)
    return avg_loss, accuracy


def val_epoch(model, loader, epoch, epochs):
    model.eval()
    total_loss = 0.0
    correct    = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1:02d}/{epochs} [Val]  ", leave=False)

    with torch.no_grad():
        for x_batch, y_batch in pbar:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            pred = model(x_batch)
            loss = criterion(pred, y_batch)

            total_loss += loss.item()
            correct    += (pred.argmax(dim=1) == y_batch).sum().item()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    accuracy = correct   / len(loader.dataset)
    return avg_loss, accuracy


# training loop
EPOCHS   = 20
best_val = float('inf')

epoch_bar = tqdm(range(EPOCHS), desc="Training", leave=True)

for epoch in epoch_bar:
    train_loss, train_acc = train_epoch(model, train_loader, epoch, EPOCHS)
    val_loss,   val_acc   = val_epoch(model,   val_loader,   epoch, EPOCHS)

    # update outer bar
    epoch_bar.set_postfix(
        train_loss = f"{train_loss:.4f}",
        train_acc  = f"{train_acc:.4f}",
        val_loss   = f"{val_loss:.4f}",
        val_acc    = f"{val_acc:.4f}"
    )

    # print summary line
    tqdm.write(f"Epoch {epoch+1:02d}/{EPOCHS} | "
               f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
               f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    # save best model
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), "best_model.pt")
        tqdm.write(f"  → saved best model (val_loss: {val_loss:.4f})")
import torch
import torch.nn as nn
import numpy as np
from model import GWMamba
from dataloader import get_all_dataloaders
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

CLASSES = {
    0: "background",
    1: "signal",
    2: "glitch",
    3: "signal_glitch"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path, n_classes=4):
    model = GWMamba(n_classes=n_classes).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"Model loaded from {checkpoint_path}")
    return model


def predict_loader(model, loader):
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)

            logits = model(x_batch)
            probs = torch.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)

            all_preds.append(preds.cpu())
            all_labels.append(y_batch.cpu())
            all_probs.append(probs.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)

    accuracy = (all_preds == all_labels).float().mean().item()
    print(f"Accuracy: {accuracy:.4f}")

    return all_preds, all_labels, all_probs


def plot_confusion_matrix(labels, preds):
    cm = confusion_matrix(labels.numpy(), preds.numpy())

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Reds",
        xticklabels=list(CLASSES.values()),
        yticklabels=list(CLASSES.values())
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix - Strain + Witness")
    plt.tight_layout()
    plt.show()


def evaluate(model, test_loader):

    preds, labels, probs = predict_loader(model, test_loader)

    print("\nClassification Report:")
    print(classification_report(
        labels.numpy(),
        preds.numpy(),
        target_names=list(CLASSES.values())
    ))

    cm = confusion_matrix(labels.numpy(), preds.numpy())

    print("Confusion Matrix:")
    print(cm)

    plot_confusion_matrix(labels, preds)

    return preds, labels, probs


if __name__ == "__main__":

    model = load_model("best_model.pt")

    _, _, test_loader = get_all_dataloaders(batch_size=32)
    preds, labels, probs = evaluate(model, test_loader)

from plistlib import load
import numpy as np
import h5py
from pathlib import Path
from sklearn.utils import shuffle
import torch
from torch.utils.data import Dataset, DataLoader

out_dir = Path(r"C:\Users\deonf\alibi\dataset\out")
classes = {
    "background":     0,
    "signal":         1,
    "glitch":         2,
    "signal_glitch":  3
}

class GWDatasetDualChannel(Dataset):
    def __init__(self, x_data, y_data):
        # x_data: (N, 2, 4096) — strain + witness stacked
        # transpose to (N, 4096, 2) for Mamba
        self.X = torch.tensor(x_data, dtype=torch.float32).permute(0, 2, 1)  # (N, 4096, 2)
        self.y = torch.tensor(y_data, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_strain_witness_data(out_dir=out_dir):
    all_x = []
    all_y = []

    for cls, label in classes.items():
        with h5py.File(out_dir / f"{cls}.h5", "r") as f:
            data = f["data"][:]          # (N, 2, 4096)
            strain  = data[:, 0, :]      # (N, 4096)
            witness = data[:, 1, :]      # (N, 4096)

        # Stack strain and witness
        x = np.stack([strain, witness], axis=1)
        y = np.full(len(x), label)

        all_x.append(x)
        all_y.append(y)

    x_data = np.concatenate(all_x, axis=0)  # (200000, 2, 4096)
    y_data = np.concatenate(all_y, axis=0)  # (200000,)

    x_data, y_data = shuffle(x_data, y_data, random_state=42)
    return x_data, y_data


def get_all_dataloaders(batch_size, out_dir=out_dir):
    x, y = load_strain_witness_data()

    print(f"x shape: {x.shape}")  # (200000, 2, 4096)
    print(f"y shape: {y.shape}")  # (200000,)
   
    dataset = GWDatasetDualChannel(x, y)

    total      = len(dataset)
    train_size = int(0.7 * total)
    val_size   = int(0.1 * total)
    test_size  = total - train_size - val_size

    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

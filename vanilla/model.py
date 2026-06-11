import torch
import torch.nn as nn
from mambapy.mamba import Mamba, MambaConfig


class GWMamba(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()

        self.config = MambaConfig(
            d_model       = 32,   # restored to reasonable size
            d_state       = 8,   # kept small for memory
            n_layers      = 4,
            expand_factor = 2,
            pscan         = True
        )
        self.downsample = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=8, stride=4, padding=2),  # 4096 > 1024  # use 1 if strain only, 2 if strain + witness
            nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=8, stride=4, padding=2), # 1024 > 256
            nn.GELU(),
        )

      
        self.in_proj    = nn.Linear(32, self.config.d_model)       # 32 > 64
        self.mamba      = Mamba(self.config)                       # (B, 256, 64) 
        self.norm       = nn.LayerNorm(self.config.d_model)
        self.classifier = nn.Linear(self.config.d_model, n_classes)

    def forward(self, x):
        # x: (B, 4096, 1)

        # downsample
        x = x.permute(0, 2, 1)       
        x = self.downsample(x)        # (B, 32, 256)
        x = x.permute(0, 2, 1)       # (B, 256, 32)

        # mamba
        x = self.in_proj(x)           # (B, 256, 64)
        x = self.mamba(x)             # (B, 256, 64)

        # classify
        x = x.mean(dim=1)             # (B, 64)
        x = self.norm(x)              # (B, 64)
        x = self.classifier(x)        # (B, 4)
        return x


if __name__ == "__main__":
    model = GWMamba(n_classes=4) 
    x     = torch.randn(8, 4096, 1)
    out   = model(x)
    print(f"Output shape: {out.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

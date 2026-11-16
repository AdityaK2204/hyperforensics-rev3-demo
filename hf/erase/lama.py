"""LaMa object-removal model (Suvorov et al., WACV 2022), TorchScript build from simple-lama-inpainting."""
from __future__ import annotations

import numpy as np
import torch

from ..config import DATA

WEIGHTS = DATA / "erase" / "dl" / "big-lama.pt"


class Lama:
    def __init__(self, device: str | None = None):
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.net = torch.jit.load(str(WEIGHTS), map_location=self.dev).eval()

    @torch.no_grad()
    def __call__(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """rgb (H, W, 3) uint8, mask (H, W) bool -> filled image (H, W, 3) uint8."""
        h, w = mask.shape
        ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
        img = np.pad(rgb, ((0, ph), (0, pw), (0, 0)), mode="reflect")
        m = np.pad(mask.astype(np.float32), ((0, ph), (0, pw)), mode="reflect")
        x = torch.from_numpy(img).permute(2, 0, 1)[None].float().div(255).to(self.dev)
        mk = torch.from_numpy((m > 0.5).astype(np.float32))[None, None].to(self.dev)
        out = self.net(x, mk)[0].permute(1, 2, 0).clamp(0, 1).mul(255).byte().cpu().numpy()
        return out[:h, :w]

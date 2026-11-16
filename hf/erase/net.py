"""Erasure detector: ResNet-34 encoder (ImageNet init) + light FPN decoder.

Output: a per-pixel erasure logit map at 1/4 resolution, and an image-level logit
pooled from the most confident 1% of that map. The network sees photos at a working
scale of about 1024 px on the long side, which is the scale the training data has.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet34_Weights, resnet34

from ..config import DATA

WEIGHTS = DATA / "erase" / "erasenet.pt"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
WORK_SIDE = 1024


def _block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class EraseNet(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        r = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)       # 1/4
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4  # 1/4 .. 1/32
        self.lat = nn.ModuleList([nn.Conv2d(c, 128, 1) for c in (64, 128, 256, 512)])
        self.smooth = nn.ModuleList([_block(128, 128) for _ in range(4)])
        self.head = nn.Sequential(_block(512, 128), nn.Conv2d(128, 1, 1))
        self.cal = nn.Linear(2, 1)

    def forward(self, x):
        x = (x - MEAN.to(x.device)) / STD.to(x.device)
        c1 = self.l1(self.stem(x))
        c2 = self.l2(c1)
        c3 = self.l3(c2)
        c4 = self.l4(c3)
        p = self.lat[3](c4)
        outs = [self.smooth[3](p)]
        for lat, sm, c in ((self.lat[2], self.smooth[2], c3), (self.lat[1], self.smooth[1], c2), (self.lat[0], self.smooth[0], c1)):
            p = F.interpolate(p, size=c.shape[2:], mode="nearest") + lat(c)
            outs.append(sm(p))
        size = outs[-1].shape[2:]
        feat = torch.cat([F.interpolate(o, size=size, mode="bilinear", align_corners=False) for o in outs], dim=1)
        m = self.head(feat)                                        # (B, 1, H/4, W/4)
        flat = m.flatten(1)
        k = max(1, flat.shape[1] // 100)
        top = flat.topk(k, dim=1).values.mean(dim=1, keepdim=True)
        avg = flat.mean(dim=1, keepdim=True)
        return m, self.cal(torch.cat([top, avg], dim=1)).squeeze(1)


_net = None


def available() -> bool:
    return WEIGHTS.exists()


def _load():
    global _net
    if _net is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = EraseNet(pretrained=False)
        ck = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
        net.load_state_dict(ck["state_dict"])
        _net = (net.to(dev).eval(), dev, dict(ck.get("meta", {})))
    return _net


def meta() -> dict:
    return _load()[2]


@torch.no_grad()
def run(rgb: np.ndarray, side: int = WORK_SIDE) -> dict:
    """rgb (H, W, 3) uint8 at any size -> {'logit', 'score', 'map' (h, w) probabilities}."""
    import cv2
    net, dev, _ = _load()
    H, W = rgb.shape[:2]
    s = min(1.0, side / max(H, W))          # reduce large photos to the working scale; never enlarge
    if abs(s - 1) > 1e-3:
        rgb = cv2.resize(rgb, (max(32, int(round(W * s))), max(32, int(round(H * s)))),
                         interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    h, w = rgb.shape[:2]
    ph, pw = (32 - h % 32) % 32, (32 - w % 32) % 32
    x = np.pad(rgb, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    t = torch.from_numpy(x).permute(2, 0, 1)[None].float().div(255).to(dev)
    m, logit = net(t)
    m = torch.sigmoid(m)[0, 0].cpu().numpy()[: (h + 3) // 4, : (w + 3) // 4]
    lg = float(logit.item())
    return dict(logit=lg, score=float(1 / (1 + np.exp(-lg))), map=m.astype(np.float32))

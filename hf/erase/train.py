"""Train the erasure detector on the generated pairs.

Model selection uses only the held-out COCO pairs (val split). The demo photos are never
seen in training or selection.

    python -m hf.erase.train [--epochs N]
"""
from __future__ import annotations

import json
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .make_data import OUT
from .net import WEIGHTS, EraseNet

CROP = 448
AUG_PHONE = False


def _auc(s, y):
    s, y = np.asarray(s, float), np.asarray(y)
    from scipy.stats import rankdata
    r = rankdata(s)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(1, n0 * n1))


class Pairs(Dataset):
    def __init__(self, split: str, train: bool, phone_aug: bool = False):
        idx = [x for x in json.loads((OUT / "index.json").read_text()) if x["split"] == split]
        self.items = [(x["stem"], 0) for x in idx] + [(x["stem"], 1) for x in idx]
        self.train = train
        self.phone_aug = phone_aug

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        stem, y = self.items[i]
        rgb = cv2.cvtColor(cv2.imread(str(OUT / f"{stem}_{'e' if y else 'o'}.jpg")), cv2.COLOR_BGR2RGB)
        m = (cv2.imread(str(OUT / f"{stem}_m.png"), 0) > 127).astype(np.float32) if y else \
            np.zeros(rgb.shape[:2], np.float32)
        if not self.train:
            return rgb, m, y
        rng = np.random.default_rng()
        s = float(np.exp(rng.uniform(np.log(0.7), np.log(1.5))))
        H, W = rgb.shape[:2]
        sz = (max(CROP // 2, int(W * s)), max(CROP // 2, int(H * s)))
        rgb = cv2.resize(rgb, sz, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        m = cv2.resize(m, sz, interpolation=cv2.INTER_LINEAR)
        if self.phone_aug:
            # phone-camera look, applied to both classes so it cannot become a label cue:
            # heavy edge-preserving denoising (smooth skies, walls, floors) on part of the frame,
            # and low-light exposure with lifted shadows
            if rng.random() < 0.5:
                d = int(rng.choice([5, 7, 9]))
                sm = cv2.bilateralFilter(rgb, d, float(rng.uniform(25, 60)), float(rng.uniform(5, 15)))
                if rng.random() < 0.5:
                    h_, w_ = rgb.shape[:2]
                    a = np.zeros((h_, w_), np.float32)
                    cy, cx = rng.uniform(0, h_), rng.uniform(0, w_)
                    ry, rx = rng.uniform(0.25, 0.7) * h_, rng.uniform(0.25, 0.7) * w_
                    yy, xx = np.mgrid[0:h_, 0:w_]
                    a = (((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 < 1).astype(np.float32)
                    a = cv2.GaussianBlur(a, (0, 0), 25)[..., None]
                    rgb = (rgb * (1 - a) + sm * a).astype(np.uint8)
                else:
                    rgb = sm
            if rng.random() < 0.3:
                g = float(rng.uniform(1.3, 2.2))
                f = float(rng.uniform(0.35, 0.7))
                x = (rgb.astype(np.float32) / 255.0) ** g * f
                x += rng.normal(0, 0.01, x.shape).astype(np.float32)
                rgb = (np.clip(x, 0, 1) * 255).astype(np.uint8)
        if rng.random() < 0.3:                      # one more lossy save at a random quality
            q = int(rng.integers(55, 92))
            rgb = cv2.imdecode(cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_UNCHANGED)
        H, W = rgb.shape[:2]
        ph, pw = max(0, CROP - H), max(0, CROP - W)
        if ph or pw:
            rgb = np.pad(rgb, ((0, ph), (0, pw), (0, 0)), mode="reflect")
            m = np.pad(m, ((0, ph), (0, pw)))
        H, W = rgb.shape[:2]
        if y and rng.random() < 0.7:                # mostly crops that contain the fill
            ys, xs = np.nonzero(m > 0.5)
            if len(ys):
                cy, cx = int(np.median(ys)), int(np.median(xs))
                y0 = int(np.clip(cy - rng.integers(CROP // 4, 3 * CROP // 4), 0, H - CROP))
                x0 = int(np.clip(cx - rng.integers(CROP // 4, 3 * CROP // 4), 0, W - CROP))
            else:
                y0, x0 = int(rng.integers(0, H - CROP + 1)), int(rng.integers(0, W - CROP + 1))
        else:
            y0, x0 = int(rng.integers(0, H - CROP + 1)), int(rng.integers(0, W - CROP + 1))
        rgb, m = rgb[y0:y0 + CROP, x0:x0 + CROP], m[y0:y0 + CROP, x0:x0 + CROP]
        if rng.random() < 0.5:
            rgb, m = rgb[:, ::-1], m[:, ::-1]
        lab = float(m.mean() > 0.003)
        return (torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div(255),
                torch.from_numpy(np.ascontiguousarray(m))[None], torch.tensor(lab))


def loss_fn(mlog, ilog, m, y):
    mt = F.interpolate(m, size=mlog.shape[2:], mode="area")
    bce = F.binary_cross_entropy_with_logits(mlog, mt, pos_weight=torch.tensor(3.0, device=m.device))
    p = torch.sigmoid(mlog)
    dice = 1 - (2 * (p * mt).sum((1, 2, 3)) + 1) / (p.sum((1, 2, 3)) + mt.sum((1, 2, 3)) + 1)
    dice = (dice * (y > 0.5)).sum() / (y > 0.5).sum().clamp(min=1)
    return bce + 0.5 * dice + F.binary_cross_entropy_with_logits(ilog, y)


@torch.no_grad()
def evaluate(net, dev, side=640):
    ds = Pairs("val", train=False)
    scores, ys, f1s = [], [], []
    for i in range(len(ds)):
        rgb, m, y = ds[i]
        H, W = rgb.shape[:2]
        s = side / max(H, W)
        r = cv2.resize(rgb, (int(W * s) // 32 * 32, int(H * s) // 32 * 32), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(r).permute(2, 0, 1)[None].float().div(255).to(dev)
        with torch.autocast("cuda", dtype=torch.float16):
            mlog, ilog = net(t)
        scores.append(float(ilog.float().item()))
        ys.append(y)
        if y:
            p = torch.sigmoid(mlog.float())[0, 0].cpu().numpy() > 0.5
            g = cv2.resize(m, (p.shape[1], p.shape[0]), interpolation=cv2.INTER_AREA) > 0.5
            f1s.append(2 * (p & g).sum() / max(1, p.sum() + g.sum()))
    scores, ys = np.array(scores), np.array(ys)
    neg = np.sort(scores[ys == 0])
    thr = float(neg[int(0.95 * len(neg))])          # 5% false alarms on held-out originals
    return dict(auc=_auc(scores, ys), acc=float(((scores > 0) == ys).mean()),
                tpr_at_5fpr=float((scores[ys == 1] > thr).mean()), thr_5fpr=thr, pixel_f1=float(np.mean(f1s)))


def main(epochs: int = 14):
    torch.manual_seed(0)
    dev = torch.device("cuda")
    net = EraseNet(pretrained=True).to(dev)
    dl = DataLoader(Pairs("train", True, AUG_PHONE), batch_size=12, shuffle=True, num_workers=6, drop_last=True,
                    persistent_workers=True)
    enc = [p for n, p in net.named_parameters() if n.split(".")[0] in ("stem", "l1", "l2", "l3", "l4")]
    dec = [p for n, p in net.named_parameters() if n.split(".")[0] not in ("stem", "l1", "l2", "l3", "l4")]
    opt = torch.optim.AdamW([dict(params=enc, lr=1e-4), dict(params=dec, lr=5e-4)], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[1e-4, 5e-4], total_steps=epochs * len(dl), pct_start=0.1)
    scaler = torch.amp.GradScaler()
    best, hist = -1.0, []
    for ep in range(epochs):
        net.train()
        t0, tot = time.time(), 0.0
        for x, m, y in dl:
            x, m, y = x.to(dev, non_blocking=True), m.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                mlog, ilog = net(x)
            loss = loss_fn(mlog.float(), ilog.float(), m, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += float(loss)
        net.eval()
        ev = evaluate(net, dev)
        ev.update(epoch=ep + 1, loss=tot / len(dl), seconds=round(time.time() - t0))
        hist.append(ev)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in ev.items()}), flush=True)
        if ev["auc"] > best:
            best = ev["auc"]
            torch.save(dict(state_dict=net.state_dict(), meta=dict(val=ev, history=hist, phone_aug=AUG_PHONE,
                            n_train=len(dl.dataset), work_side=1024)), WEIGHTS)
    print("best val AUC", round(best, 4))


if __name__ == "__main__":
    import hf.erase.train as me
    if "--out" in sys.argv:
        from pathlib import Path
        WEIGHTS = Path(sys.argv[sys.argv.index("--out") + 1])
        me.WEIGHTS = WEIGHTS
    me.AUG_PHONE = AUG_PHONE = "--phone-aug" in sys.argv
    me.main(int(sys.argv[sys.argv.index("--epochs") + 1]) if "--epochs" in sys.argv else 14)

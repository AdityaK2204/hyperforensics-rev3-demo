"""06 Topology and fusion: one graph network, three edge sets.

Node state   h_k <- (evidence, trust) of cue k, where evidence is the cue's frozen
             log-likelihood ratio and trust is P(cue right | estimated damage).
Messages     pass only along the edge set.
Readout      p(fake | x) = sigmoid( sum_k node_k + b ). node_k is cue k's
             contribution after hearing from its neighbours, which is what the
             interface reports as "the cues that decided it".

The network is identical in every arm; only the edges change:

  A1  dependence-derived   edge(i,k) <=> measured error correlation > theta
  A2  fully connected      every pair of cues (the Rev 1 design)
  A3  independence         no edges: an additive, naive-Bayes-style combiner

Same cues, same features, same budget, same seeds.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .config import CUE_IDS, ECE_BINS, FUSION_SEEDS, K, PIXEL_CUES, R, REF_DIR, RUNG_IDS
from .measure import _clean, auc

ARMS = ["A1", "A2", "A3"]
ARM_LABEL = dict(A1="dependence graph", A2="fully connected", A3="independence")
torch.set_num_threads(2)


class GraphFusion(nn.Module):
    def __init__(self, k: int = K, h: int = 8):
        super().__init__()
        self.inp = nn.Linear(3 + k, h)
        self.msg = nn.Linear(h, h, bias=False)
        self.upd = nn.Linear(h, h)
        self.out = nn.Linear(h, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(1))
        self.register_buffer("eye", torch.eye(k))

    def forward(self, X, A, mask):
        b = X.shape[0]
        h = torch.tanh(self.inp(torch.cat([X, self.eye.unsqueeze(0).expand(b, -1, -1)], dim=2)))
        h = h * mask.unsqueeze(-1)
        m = torch.bmm(A, h)                                   # messages along edges only
        h = torch.tanh(self.upd(h) + self.msg(m)) * mask.unsqueeze(-1)
        node = self.out(h).squeeze(-1) * mask * X[..., 2]       # gated by trust: no trust, no say
        return node.sum(dim=1) + self.bias, node


# ---------------------------------------------------------------------- features
class Frozen:
    """The frozen-at-R0 store: thresholds, evidence calibration, trust, measured dependence."""

    def __init__(self):
        z = np.load(REF_DIR / "frozen.npz")
        self.tau, self.llr, self.trust = z["tau"], z["llr"], z["trust"]
        self.T, self.scale = z["T"], z["scale"]
        self.theta = float(z["theta"])
        rho = np.nan_to_num(z["rho_calib"], nan=0.0)          # (R, K, K)
        adj = np.where(rho > self.theta, rho, 0.0)
        for r in range(R):
            np.fill_diagonal(adj[r], 0.0)
        self.adj = adj.astype(np.float32)

    def decide(self, scores: np.ndarray) -> np.ndarray:
        return np.nan_to_num(scores, nan=-np.inf) >= np.nan_to_num(self.tau, nan=np.inf)[None, :]

    def graphs(self, scores: np.ndarray, post: np.ndarray, probe: np.ndarray | None = None):
        """scores (m, K) with NaN for masked cues, post (m, R) rung posterior, probe (m, K)
        the same cues after one plain re-save (None: assume every decision survives) ->
        node features X (m, K, 3), mask (m, K), adjacency per arm, trust (m, K), evidence (m, K)."""
        mask = (~np.isnan(scores)).astype(np.float32)
        s = np.nan_to_num(scores, nan=0.0)
        ev = np.clip(self.llr[None, :, 0] * s + self.llr[None, :, 1], -3.0, 3.0) * mask
        d = self.decide(scores)
        b = np.ones_like(d) if probe is None else (self.decide(probe) == d)
        kk = np.arange(K)[None, :]
        Tsel = self.T[kk, :, d.astype(int), b.astype(int)]    # (m, K, R)
        tr = np.einsum("mkr,mr->mk", np.clip(2.0 * Tsel - 1.0, 0.0, 1.0), post) * mask
        # Trust gates everything a node may say: evidence and decision enter only
        # multiplied by trust, so a cue that cannot be trusted at this level of damage
        # has no way to move the verdict, whatever its score.
        X = np.stack([ev * tr, d * tr, tr], axis=2).astype(np.float32)
        pair = mask[:, :, None] * mask[:, None, :]
        A1 = np.einsum("mr,rij->mij", post, self.adj).astype(np.float32) * pair
        A2 = (1.0 - np.eye(K, dtype=np.float32))[None] * pair
        A3 = np.zeros_like(A2)
        return X, mask, dict(A1=A1, A2=A2, A3=A3), tr, ev


def _t(a):
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))


def _train(X, A, M, y, seed: int, epochs: int = 350) -> GraphFusion:
    torch.manual_seed(seed)
    net = GraphFusion()
    opt = torch.optim.Adam(net.parameters(), lr=0.02, weight_decay=2e-3)
    lossf = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(net(X, A, M)[0], y)
        loss.backward()
        opt.step()
    return net.eval()


def ece(p: np.ndarray, y: np.ndarray, bins: int = ECE_BINS):
    edges = np.linspace(0, 1, bins + 1)
    out, diagram = 0.0, []
    for i in range(bins):
        sel = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= 1.0)
        if sel.sum() == 0:
            diagram.append(None)
            continue
        out += sel.mean() * abs(p[sel].mean() - y[sel].mean())
        diagram.append([round(float(p[sel].mean()), 3), round(float(y[sel].mean()), 3), int(sel.sum())])
    return float(out), diagram


# ------------------------------------------------------------------- train + eval
def train_and_evaluate(records, S, F, Sp, ref: dict, log=print) -> None:
    hub = ref.pop("_hub")
    frozen = Frozen()
    y = np.array([r["y"] for r in records])
    split = np.array([r["split"] for r in records])
    fam = np.array([r["family"] for r in records])
    tr, te = split == "train", split == "test"
    n = len(records)
    flat = lambda a, m: _t(a[m].reshape((-1,) + a.shape[2:]))  # noqa: E731
    ytr = _t(np.repeat(y[tr], R))

    def graphs(S_, Sp_):
        X_, M_, A_, _, _ = frozen.graphs(S_.reshape(n * R, K), hub.oof.reshape(n * R, R), Sp_.reshape(n * R, K))
        return X_.reshape(n, R, K, 3), M_.reshape(n, R, K), {a: v.reshape(n, R, K, K) for a, v in A_.items()}

    X, M, A = graphs(S, Sp)

    # Training sees every sample twice: once as recorded, once with the sensor cue
    # masked. In the corpus a missing fingerprint record is far commoner among fakes
    # (generator output has no camera), and a network will happily learn "no C4 means
    # fake". Outside the corpus most authentic files have no registered camera either.
    # A third copy has all pixel cues (C1-C6) masked: that is the large-photo case, where
    # those cues are outside their calibration domain and only C8 and C9 can be read.
    S4, Sp4 = S.copy(), Sp.copy()
    S4[:, :, 3], Sp4[:, :, 3] = np.nan, np.nan
    Xm, Mm, Am = graphs(S4, Sp4)
    SP_, SPp = S.copy(), Sp.copy()
    SP_[:, :, PIXEL_CUES], SPp[:, :, PIXEL_CUES] = np.nan, np.nan
    Xq, Mq, Aq = graphs(SP_, SPp)
    cat = lambda *xs: torch.cat([flat(x, tr) for x in xs], dim=0)  # noqa: E731
    ytr3 = torch.cat([ytr, ytr, ytr])

    def train(Xa, Ma, Aa, Xb, Mb, Ab, arm, seed, Xc=Xq, Mc=Mq, Ac=Aq):
        return _train(cat(Xa, Xb, Xc), cat(Aa[arm], Ab[arm], Ac[arm]), cat(Ma, Mb, Mc), ytr3, seed)

    states, P = {}, {}
    for arm in ARMS:
        nets = [train(X, M, A, Xm, Mm, Am, arm, seed) for seed in FUSION_SEEDS]
        states[arm] = [net.state_dict() for net in nets]
        with torch.no_grad():
            P[arm] = np.stack([torch.sigmoid(net(flat(X, te), flat(A[arm], te), flat(M, te))[0]).numpy()
                               .reshape(int(te.sum()), R) for net in nets])        # (seeds, n_te, R)
        log(f"  arm {arm} ({ARM_LABEL[arm]}): trained {len(nets)} seeds")
    torch.save(states, REF_DIR / "fusion.pt")

    yte = y[te]
    res = {}
    seed_auc = {}
    for arm in ARMS:
        sa = np.array([[auc(P[arm][s][:, r], yte) for r in range(R)] for s in range(len(FUSION_SEEDS))])
        seed_auc[arm] = sa
        pe = P[arm].mean(axis=0)                                # seed ensemble
        eces, diagrams, accs = [], [], []
        for r in range(R):
            e, d = ece(pe[:, r], yte)
            eces.append(e)
            diagrams.append(d)
            accs.append(float(((pe[:, r] >= 0.5) == yte).mean()))
        res[arm] = dict(label=ARM_LABEL[arm], auc=sa.mean(axis=0).tolist(), auc_std=sa.std(axis=0).tolist(),
                        ece=eces, acc=accs, reliability=diagrams)
    gap_fc = seed_auc["A1"] - seed_auc["A2"]
    gap_in = seed_auc["A1"] - seed_auc["A3"]
    heavy, light = [5, 6, 7, 8], [0, 1]
    summ = lambda g, rr: [float(g[:, rr].mean()), float(g[:, rr].mean(axis=1).std())]  # noqa: E731
    best_single = [float(np.nanmax([v[r] if v[r] is not None else np.nan for v in ref["survival"]["auc"]]))
                   for r in range(R)]

    # how much of the fused result rests on the container: the same arm, trained and
    # tested with C8 masked, so only pixel evidence is left
    S8, Sp8 = S.copy(), Sp.copy()
    S8[:, :, 7], Sp8[:, :, 7] = np.nan, np.nan
    X8, M8, A8 = graphs(S8, Sp8)
    S84, Sp84 = S8.copy(), Sp8.copy()
    S84[:, :, 3], Sp84[:, :, 3] = np.nan, np.nan
    X8m, M8m, A8m = graphs(S84, Sp84)
    S8q, Sp8q = S8.copy(), Sp8.copy()
    S8q[:, :, PIXEL_CUES], Sp8q[:, :, PIXEL_CUES] = np.nan, np.nan
    X8q, M8q, A8q = graphs(S8q, Sp8q)
    nets8 = [train(X8, M8, A8, X8m, M8m, A8m, "A1", seed, X8q, M8q, A8q) for seed in FUSION_SEEDS]
    with torch.no_grad():
        P8 = np.stack([torch.sigmoid(net(flat(X8, te), flat(A8["A1"], te), flat(M8, te))[0]).numpy()
                       .reshape(int(te.sum()), R) for net in nets8])
    pixel_only = [float(np.mean([auc(P8[s][:, r], yte) for s in range(len(FUSION_SEEDS))])) for r in range(R)]
    log("  ablation: A1 without the container cue")

    # demo gallery: for each family, the test sample whose fused score at R0 is the
    # median of its family, i.e. a typical case and not the best one
    pe0 = P["A1"].mean(axis=0)[:, 0]
    te_idx = np.nonzero(te)[0]
    gallery = []
    for f in ["real", "real_web", "splice", "copymove", "inpaint", "synth_region", "full_synth"]:
        cand = [j for j in range(len(te_idx)) if fam[te_idx[j]] == f]
        cand.sort(key=lambda j: pe0[j])
        pick = records[te_idx[cand[len(cand) // 2]]]
        gallery.append(dict(id=pick["id"], rung=0, family=f))
        if f == "splice":
            gallery.append(dict(id=pick["id"], rung=2, family=f))
            gallery.append(dict(id=pick["id"], rung=6, family=f))
        if f == "real":
            gallery.append(dict(id=pick["id"], rung=8, family=f))
    ref["gallery"] = gallery

    ref["fusion"] = _clean(dict(
        arms=res, seeds=FUSION_SEEDS, best_single=best_single, pixel_only=pixel_only,
        gap_fc=dict(mean=gap_fc.mean(axis=0).tolist(), std=gap_fc.std(axis=0).tolist()),
        gap_ind=dict(mean=gap_in.mean(axis=0).tolist(), std=gap_in.std(axis=0).tolist()),
        heavy=dict(fc=summ(gap_fc, heavy), ind=summ(gap_in, heavy)),
        light=dict(fc=summ(gap_fc, light), ind=summ(gap_in, light))))

    hfc, hin = summ(gap_fc, heavy), summ(gap_in, heavy)
    se = lambda v: v[1] / np.sqrt(len(FUSION_SEEDS)) + 1e-9  # noqa: E731
    ref["hypotheses"]["H3"] = dict(
        supported=bool(hfc[0] > 2 * se(hfc) and hin[0] > 2 * se(hin)),
        heavy_gap_fc=hfc, heavy_gap_ind=hin,
        light_gap_fc=summ(gap_fc, light), light_gap_ind=summ(gap_in, light))

    # the four negative results the project pre-registered, reported whatever they are
    rstar = ref["survival"]["rstar"]
    a1, a2, a3 = res["A1"], res["A2"], res["A3"]
    ref["negatives"] = [
        dict(key="dead_cues", title="Cues whose R* is R0 (never carried information here)",
             value=[CUE_IDS[k] for k in range(K) if rstar[k] == 0]),
        dict(key="fc_wins", title="Rungs where the fully connected graph matches or beats ours",
             value=[RUNG_IDS[r] for r in range(R) if a2["auc"][r] >= a1["auc"][r]]),
        dict(key="indomain_gap", title="In-domain fusion gap at R0 (ours minus independence)",
             value=round(float(gap_in[:, 0].mean()), 4)),
        dict(key="worse_calibration", title="Rungs where our calibration is worse than the independence arm",
             value=[RUNG_IDS[r] for r in range(R) if a1["ece"][r] > a3["ece"][r]]),
    ]


# ----------------------------------------------------------------------- inference
class Fuser:
    """Loads the three trained arms and fuses cue scores for any number of file states."""

    def __init__(self):
        self.frozen = Frozen()
        states = torch.load(REF_DIR / "fusion.pt", weights_only=True)
        self.nets = {}
        for arm in ARMS:
            self.nets[arm] = []
            for sd in states[arm]:
                net = GraphFusion()
                net.load_state_dict(sd)
                self.nets[arm].append(net.eval())

    def predict(self, scores: np.ndarray, post: np.ndarray, probe: np.ndarray | None = None,
                pixel_domain: np.ndarray | None = None) -> dict:
        """scores (m, K), post (m, R), probe (m, K): the same cues after one plain re-save,
        which is what the lifecycle pipeline contributes to each cue's trust.
        pixel_domain (m,): how far each state lies inside the pixel cues' calibration domain."""
        scores, post = np.atleast_2d(scores).astype(np.float32), np.atleast_2d(post)
        probe = None if probe is None else np.atleast_2d(probe).astype(np.float32)
        if pixel_domain is not None:
            # a cue entirely outside its calibration domain is absent, exactly as a masked cue
            # is: a node with no trust must not contribute even its learned offset
            off = np.asarray(pixel_domain, dtype=np.float32) < 0.05
            scores = scores.copy()
            scores[np.ix_(off, PIXEL_CUES)] = np.nan
            if probe is not None:
                probe = probe.copy()
                probe[np.ix_(off, PIXEL_CUES)] = np.nan
        X, M, A, trust, ev = self.frozen.graphs(scores, post, probe)
        d = self.frozen.decide(scores)
        if pixel_domain is not None:
            f = np.ones_like(trust)
            f[:, PIXEL_CUES] = np.asarray(pixel_domain, dtype=np.float32)[:, None]
            trust = trust * f
            X = np.stack([ev * trust, d * trust, trust], axis=2).astype(np.float32)
        out = dict(trust=trust, evidence=ev, mask=M, adjacency=A["A1"], decision=d,
                   persists=np.ones_like(d) if probe is None else (self.frozen.decide(probe) == d))
        with torch.no_grad():
            for arm in ARMS:
                logits, nodes = zip(*[net(_t(X), _t(A[arm]), _t(M)) for net in self.nets[arm]])
                lg = torch.stack(logits).mean(dim=0)
                out[arm] = torch.sigmoid(lg).numpy()
                if arm == "A1":
                    out["contrib"] = torch.stack(nodes).mean(dim=0).numpy()
                    out["spread"] = torch.sigmoid(torch.stack(logits)).std(dim=0).numpy()
        return out

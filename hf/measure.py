"""04 Decisions, errors, survival  ·  05 Dependence measurement  ·  damage hub and trust.

Everything here is computed from the score tensor S[n, r, k] and nothing else.

  thresholds   tau_k fitted on the calib split at R0, then frozen for every rung
  errors       E[n, r, k] = [ d_k != y ]            one bit per sample, rung, cue
  survival     AUC_k(r) on the test split, and R*_k = lowest rung below the floor
  dependence   Q-statistic, disagreement and correlation over cue *errors*, per rung
               (Kuncheva & Whitaker 2003), on the calib split for the topology and on
               the test split for reporting
  topology     edge(i,k) <=> rho_r[i,k] > theta, theta from the calib split at R0
  trust        P(cue right | rung) as informedness at the frozen threshold
  damage hub   which rung a file looks like, from the damage features alone
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

from .config import AUC_FLOOR, CUE_IDS, EDGE_NULL_Q, K, MAX_FPR, R, REF_DIR, RUNG_IDS, SEED
from .cues import DAMAGE_NAMES

SUBSTRATE = [0, 1, 2]            # C1-C3: the cues that read the compression substrate


def auc(s, y) -> float:
    s, y = np.asarray(s, dtype=float), np.asarray(y)
    ok = ~np.isnan(s)
    s, y = s[ok], y[ok]
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def youden_threshold(s, y, max_fpr: float = MAX_FPR) -> float:
    """Operating point: the most informative threshold whose false-alarm rate on
    authentic files stays within max_fpr. A forensic cue that cries wolf on one
    authentic file in four is not usable, however well it ranks."""
    ok = ~np.isnan(s)
    s, y = s[ok], y[ok]
    u = np.unique(s)
    if len(u) < 2:
        return float(u[0]) + 1.0 if len(u) else 0.0
    cands = np.append((u[:-1] + u[1:]) / 2.0, u[-1] + 1e-3)
    best, best_t = -1.0, float(cands[-1])
    for t in cands:
        d = s >= t
        fpr = d[y == 0].mean()
        if fpr > max_fpr:
            continue
        j = d[y == 1].mean() - fpr
        if j >= best:                      # ties go to the higher threshold: fewer false alarms
            best, best_t = j, float(t)
    return best_t


def informedness(d, y) -> float:
    ok = ~np.isnan(d)
    d, y = d[ok] > 0.5, y[ok]
    if (y == 1).sum() == 0 or (y == 0).sum() == 0:
        return 0.0
    return float(d[y == 1].mean() + (~d[y == 0]).mean() - 1.0)


def dependence(E: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pairwise Q-statistic, disagreement and correlation over an error matrix E[n, k] (NaN = masked)."""
    k = E.shape[1]
    Q = np.full((k, k), np.nan)
    Dis = np.full((k, k), np.nan)
    Rho = np.full((k, k), np.nan)
    for i in range(k):
        for j in range(k):
            ok = ~np.isnan(E[:, i]) & ~np.isnan(E[:, j])
            if ok.sum() < 10:
                continue
            if i == j:
                Q[i, j], Dis[i, j], Rho[i, j] = 1.0, 0.0, 1.0
                continue
            ri, rj = E[ok, i] < 0.5, E[ok, j] < 0.5           # "right" indicators
            n11 = float((ri & rj).sum())
            n00 = float((~ri & ~rj).sum())
            n10 = float((ri & ~rj).sum())
            n01 = float((~ri & rj).sum())
            a, b, c, d = n11 + 0.5, n00 + 0.5, n10 + 0.5, n01 + 0.5   # Haldane correction
            Q[i, j] = (a * b - c * d) / (a * b + c * d)
            Dis[i, j] = (n10 + n01) / ok.sum()
            den = np.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
            Rho[i, j] = (n11 * n00 - n01 * n10) / den if den > 0 else 0.0
    return Q, Dis, Rho


def _null_theta(E: np.ndarray, q: float, B: int = 400) -> tuple[float, np.ndarray]:
    """Independence null for error correlation: shuffle every cue's errors separately."""
    rng = np.random.default_rng([SEED, 21])
    k = E.shape[1]
    pooled, block = [], []
    for _ in range(B):
        Ep = E.copy()
        for j in range(k):
            ok = ~np.isnan(E[:, j])
            Ep[ok, j] = rng.permutation(E[ok, j])
        _, _, rho = dependence(Ep)
        iu = np.triu_indices(k, 1)
        v = rho[iu]
        pooled.append(v[~np.isnan(v)])
        block.append(np.nanmean([rho[0, 1], rho[0, 2], rho[1, 2]]))
    return float(np.quantile(np.concatenate(pooled), q)), np.asarray(block)


def _clean(a):
    """NaN -> None so the result is valid JSON."""
    if isinstance(a, np.ndarray):
        return [_clean(x) for x in a.tolist()]
    if isinstance(a, list):
        return [_clean(x) for x in a]
    if isinstance(a, float) and (np.isnan(a) or np.isinf(a)):
        return None
    return a


# ------------------------------------------------------------------ damage hub
class DamageHub:
    """Which rung does this file look like? Reads Q-tables, lattice, grid, noise and container."""

    def __init__(self, F: np.ndarray, groups: np.ndarray, fit_mask: np.ndarray, with_oof: bool = True):
        self.mu = F[fit_mask].reshape(-1, F.shape[2]).mean(axis=0)
        self.sd = F[fit_mask].reshape(-1, F.shape[2]).std(axis=0) + 1e-6
        X, yr, g = self._flat(F, fit_mask, groups)
        self.model = self._fit(X, yr)
        # out-of-fold posteriors for the samples the hub was fitted on, so that anything
        # trained downstream sees the hub exactly as it behaves on unseen files
        self.oof = None
        if with_oof:
            self.oof = np.zeros((F.shape[0], R, R))
            ug = np.unique(groups[fit_mask])
            folds = np.array_split(np.random.default_rng([SEED, 31]).permutation(ug), 4)
            for fold in folds:
                hold = fit_mask & np.isin(groups, fold)
                keep = fit_mask & ~np.isin(groups, fold)
                mdl = self._fit(*self._flat(F, keep, groups)[:2])
                self.oof[hold] = self._proba(mdl, F[hold].reshape(-1, F.shape[2])).reshape(-1, R, R)
            rest = ~fit_mask
            self.oof[rest] = self.posterior(F[rest].reshape(-1, F.shape[2])).reshape(-1, R, R)
        Z = (X - self.mu) / self.sd
        self.nn = NearestNeighbors(n_neighbors=6).fit(Z)
        d = self.nn.kneighbors(Z)[0][:, 1:].mean(axis=1)
        self.d99 = float(np.quantile(d, 0.99))

    @staticmethod
    def _flat(F, mask, groups):
        idx = np.nonzero(mask)[0]
        X = F[idx].reshape(-1, F.shape[2])
        yr = np.tile(np.arange(R), len(idx))
        return X, yr, np.repeat(groups[idx], R)

    @staticmethod
    def _fit(X, yr):
        return RandomForestClassifier(n_estimators=200, min_samples_leaf=3, random_state=SEED,
                                      n_jobs=1).fit(X, yr)

    @staticmethod
    def _proba(mdl, X):
        p = np.zeros((len(X), R))
        p[:, mdl.classes_] = mdl.predict_proba(X)
        return p

    def posterior(self, X: np.ndarray) -> np.ndarray:
        return self._proba(self.model, np.atleast_2d(X))

    def envelope(self, x: np.ndarray) -> float:
        """Distance to the calibrated region of feature space, in units of its 99th percentile."""
        z = (np.atleast_2d(x) - self.mu) / self.sd
        return float(self.nn.kneighbors(z, n_neighbors=5)[0].mean() / self.d99)


def trust_table(D, Dp, E, y, mask) -> np.ndarray:
    """T[k, r, d, b] = P(cue k is right | rung r, its decision d, decision survives a re-save b).

    Estimated on the calib split with the two classes weighted equally, so a cue
    that fires at random earns 0.5 whatever the class balance. Cells are shrunk
    towards the coarser estimate above them, so a thin cell cannot swing the trust.
    """
    w = np.where(y == 1, 0.5 / max(1, (y[mask] == 1).sum()), 0.5 / max(1, (y[mask] == 0).sum()))
    T = np.full((K, R, 2, 2), 0.5)
    for k in range(K):
        for r in range(R):
            ok = mask & ~np.isnan(D[:, r, k])
            if ok.sum() < 10:
                continue
            right = 1.0 - E[ok, r, k]
            d = D[ok, r, k] > 0.5
            b = (Dp[ok, r, k] > 0.5) == d
            ww = w[ok]
            base = float((ww * right).sum() / ww.sum())
            for dv in (0, 1):
                sd = d == dv
                rate_d = float((ww[sd] * right[sd]).sum() / ww[sd].sum()) if sd.any() else base
                t0 = (sd.sum() * rate_d + 4.0 * base) / (sd.sum() + 4.0)
                for bv in (0, 1):
                    sb = sd & (b == bool(bv))
                    rate = float((ww[sb] * right[sb]).sum() / ww[sb].sum()) if sb.any() else t0
                    T[k, r, dv, bv] = (sb.sum() * rate + 6.0 * t0) / (sb.sum() + 6.0)
    return T


# ------------------------------------------------------------------ measurement
def measure(records: list[dict], S: np.ndarray, F: np.ndarray, Sp: np.ndarray) -> dict:
    y = np.array([r["y"] for r in records])
    split = np.array([r["split"] for r in records])
    scene = np.array([r["scene"][:3] for r in records])
    fam = np.array([r["family"] for r in records])
    tr, ca, te = split == "train", split == "calib", split == "test"

    # ---- thresholds: calib split, R0, frozen
    tau = np.array([youden_threshold(S[ca, 0, k], y[ca]) if k != 6 else np.nan for k in range(K)])
    # C9 is calibrated on held-out data from its own operating domain (hf/erase/calibrate.py)
    cal = None
    if K > 8 and (REF_DIR / "c9_calibration.npz").exists():
        cal = np.load(REF_DIR / "c9_calibration.npz")
        tau[8] = float(cal["tau"])
    D = np.where(np.isnan(S), np.nan, (S >= tau[None, None, :]).astype(float))
    E = np.where(np.isnan(D), np.nan, (D != y[:, None, None]).astype(float))

    # ---- survival
    auc_te = np.array([[auc(S[te, r, k], y[te]) for r in range(R)] for k in range(K)])
    auc_ca = np.array([[auc(S[ca, r, k], y[ca]) for r in range(R)] for k in range(K)])
    acc_te = np.array([[np.nanmean(1 - E[te, r, k]) if not np.all(np.isnan(E[te, r, k])) else np.nan
                        for r in range(R)] for k in range(K)])
    rstar = []
    for k in range(K):
        below = [r for r in range(R) if not np.isnan(auc_te[k, r]) and auc_te[k, r] < AUC_FLOOR]
        rstar.append(None if np.all(np.isnan(auc_te[k])) else (below[0] if below else R))
    trust = np.array([[max(0.0, informedness(D[ca, r, k], y[ca])) for r in range(R)] for k in range(K)])
    trust[6] = 0.0
    # the cue reliability model: trust conditioned on rung, decision and probe persistence
    Dp = np.where(np.isnan(Sp), np.nan, (Sp >= tau[None, None, :]).astype(float))
    T = trust_table(D, Dp, E, y, ca)
    if cal is not None:
        T[8] = cal["T"]
        trust[8] = cal["trust"]
    persist = np.array([[float(np.nanmean((Dp[te, r, k] == D[te, r, k])[~np.isnan(D[te, r, k])]))
                         if not np.all(np.isnan(D[te, r, k])) else np.nan for r in range(R)] for k in range(K)])
    scale = np.array([max(0.3, float(np.nanstd(S[ca & (y == 0), 0, k]))) if k != 6 else 1.0 for k in range(K)])

    # ---- dependence, per rung
    dep_te = [dependence(E[te, r]) for r in range(R)]
    dep_ca = [dependence(E[ca, r]) for r in range(R)]
    theta, null_block = _null_theta(E[ca, 0], EDGE_NULL_Q)
    edges = []
    for r in range(R):
        rho = dep_ca[r][2]
        edges.append([[i, j, round(float(rho[i, j]), 3)] for i in range(K) for j in range(i + 1, K)
                      if not np.isnan(rho[i, j]) and rho[i, j] > theta])

    # ---- hypotheses H1, H2 (H3 is added by the fusion stage)
    live = [k for k in range(K) if rstar[k] is not None]
    order0 = sorted(live, key=lambda k: -auc_te[k, 0])
    changes = []
    for r in range(1, R):
        o = sorted(live, key=lambda k: -auc_te[k, r])
        if o[:3] != order0[:3]:
            changes.append(r)
    r_distinct = sorted({rstar[k] for k in live})
    h1 = dict(supported=bool(len(r_distinct) > 1 and changes),
              rstar=[None if v is None else ("never" if v == R else RUNG_IDS[v]) for v in rstar],
              ranking_r0=[CUE_IDS[k] for k in order0],
              rungs_where_top3_changes=[RUNG_IDS[r] for r in changes])
    blk_te = [float(np.nanmean([dep_te[r][2][0, 1], dep_te[r][2][0, 2], dep_te[r][2][1, 2]])) for r in range(R)]
    others = [(i, j) for i in range(K) for j in range(i + 1, K) if not (i in SUBSTRATE and j in SUBSTRATE)]
    oth_te = [float(np.nanmean([dep_te[r][2][i, j] for i, j in others])) for r in range(R)]
    blk_ca0 = float(np.nanmean([dep_ca[0][2][0, 1], dep_ca[0][2][0, 2], dep_ca[0][2][1, 2]]))
    p_block = float((null_block >= blk_ca0).mean())
    h2_rungs = [RUNG_IDS[r] for r in range(R) if blk_te[r] > theta and blk_te[r] > oth_te[r]]
    h2 = dict(supported=bool(len(h2_rungs) >= 2), block_rho=blk_te, other_rho=oth_te,
              rungs_block_above_theta=h2_rungs, p_block_at_r0=p_block)

    # ---- damage hub
    hub = DamageHub(F, scene, tr | ca)
    post_te = hub.oof[te]                                       # (n_te, R, R)
    pred = post_te.argmax(axis=2)
    true = np.tile(np.arange(R), (int(te.sum()), 1))
    conf = np.zeros((R, R))
    for a, b in zip(true.ravel(), pred.ravel()):
        conf[a, b] += 1
    conf /= conf.sum(axis=1, keepdims=True)
    hub_stats = dict(accuracy=float((pred == true).mean()), within1=float((np.abs(pred - true) <= 1).mean()),
                     confusion=conf.round(3).tolist(), features=DAMAGE_NAMES,
                     importance=[round(float(v), 3) for v in hub.model.feature_importances_])

    # ---- score -> log-likelihood ratio, fitted at R0 on the train split, then frozen
    llr = np.zeros((K, 2))
    prior = float(np.log(y[tr].mean() / (1 - y[tr].mean())))
    for k in range(K):
        s = S[tr, 0, k]
        ok = ~np.isnan(s)
        if ok.sum() < 20 or len(np.unique(y[tr][ok])) < 2:
            continue
        lr = LogisticRegression(C=1.0).fit(s[ok, None], y[tr][ok])
        pk = float(np.log(y[tr][ok].mean() / (1 - y[tr][ok].mean())))
        llr[k] = [float(lr.coef_[0, 0]), float(lr.intercept_[0]) - pk]

    if cal is not None:
        llr[8] = cal["llr"]
        scale[8] = float(cal["scale"])
    np.savez_compressed(REF_DIR / "frozen.npz", tau=tau, llr=llr, trust=trust, T=T, scale=scale, prior=prior,
                        theta=theta, rho_calib=np.stack([d[2] for d in dep_ca]))

    counts = {f: int((fam == f).sum()) for f in sorted(set(fam))}
    ref = dict(
        corpus=dict(n=len(records), n_real=int((y == 0).sum()), n_fake=int((y == 1).sum()),
                    splits={s: int((split == s).sum()) for s in ("train", "calib", "test")},
                    families=counts, scenes=len(set(r["scene"] for r in records))),
        thresholds=_clean(tau), scale=_clean(scale), floor=AUC_FLOOR,
        survival=dict(auc=_clean(auc_te), auc_calib=_clean(auc_ca), acc=_clean(acc_te),
                      rstar=[None if v is None else int(v) for v in rstar]),
        trust=_clean(trust), trust_table=_clean(np.round(T, 3)), persistence=_clean(persist),
        dependence=dict(Q=_clean(np.stack([d[0] for d in dep_te])), Dis=_clean(np.stack([d[1] for d in dep_te])),
                        rho=_clean(np.stack([d[2] for d in dep_te])),
                        rho_calib=_clean(np.stack([d[2] for d in dep_ca])),
                        theta=theta, edges=edges),
        hub=hub_stats, hypotheses=dict(H1=h1, H2=h2),
        c9_calibration=None if cal is None else dict(source="held-out COCO erasure pairs (val split)",
                                                     n=int(cal["n"]), auc=_clean(cal["auc"]),
                                                     tau=float(cal["tau"]), trust=_clean(cal["trust"])),
    )
    ref["_hub"] = hub                      # live object for the fusion stage; removed before saving
    return ref

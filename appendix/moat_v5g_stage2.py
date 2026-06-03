#!/usr/bin/env python
"""
MOAT v5g Stage 2 — Endogenous Attribution Collapse

Stage 1 (moat_v5g.py) showed:
  Externally reducing DirectionalEnergy_B while keeping PE and total
  energy causes AUC_residual to collapse.

Stage 2 (this file) shows:
  SRAAgent under sustained Q-burst generates the SAME directional
  depletion *endogenously* — no externally supplied wrong_strength.

Mechanism:
  1. Q burst → residuals e_t = w_t (burst noise along v_Q)
  2. LS update: B_est += lr * outer(e_t, u_t) / ||u_t||^2
     Under sustained burst, u_t correlates with burst direction (closed loop)
     → B_est accumulates outer(v_Q, v_Q) component
  3. Policy: concentrate energy on dominant direction of B_est - I
     → u_t increasingly along v_Q
  4. More correlation → more contamination (positive feedback)
  5. DirectionalEnergy_B = v_B^T cov_u v_B / trace(cov_u) → min_de

Under H_B:
  B drift produces mean residual along v_B
  → B_est correctly learns v_B
  → policy stays along v_B
  → DirectionalEnergy_B stays high

Primary claim (Stage 2):
  Under H_Q + SRAAgent: DirectionalEnergy_B drops endogenously
  while PE and total energy are preserved.
  This is wrong_strength generated from inside, not injected.

Secondary claim:
  The drop is comparable to Stage 1 wrong_strength sweep,
  confirming the geometry is the same failure mode.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

Array = np.ndarray


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Stage2Cfg:
    seed: int = 42
    n_ep: int = 600        # episodes per hypothesis
    T: int = 60            # steps per episode
    delta_b: float = 0.9
    sigma_w: float = 0.25
    input_energy: float = 2.0
    agent_lr: float = 0.15
    min_de: float = 0.15   # minimum directional energy ratio in policy
    theta_min: float = 30.0
    theta_max: float = 150.0
    # Measurement windows
    early_start: int = 3
    early_end: int = 18
    late_start: int = 38
    late_end: int = 55
    # Collapse thresholds (matching Stage 1)
    pe_thresh: float = 0.15
    energy_thresh: float = 1.0
    de_low_thresh: float = 0.35
    auc_high: float = 0.75
    auc_low: float = 0.60
    # Classifiers
    rff_dim: int = 160
    train_steps: int = 200
    lr_cls: float = 0.08
    n_train_frac: float = 0.7


# ---------------------------------------------------------------------------
# SRAAgent — misattributes Q-burst residuals to B drift
# ---------------------------------------------------------------------------

class SRAAgent:
    """
    Recursive LS agent with no Q-burst model.
    Update: B_est += lr * outer(e_t, u_t) / ||u_t||^2
    Policy: concentrate energy on dominant direction of (B_est - I).

    Contamination path:
      e_t ≈ w_t (burst noise along v_Q) on first steps
      B_est - I accumulates outer(w_t, u_t) / ||u_t||^2
      Dominant left-singular vector of (B_est - I) → v_Q direction
      Policy puts max energy on v_Q → u_t correlated with burst
      Further steps reinforce outer(v_Q, v_Q) in B_est - I
    """
    def __init__(self, cfg: Stage2Cfg, rng: np.random.Generator):
        self.B_est = np.eye(2, dtype=float)
        self.lr = cfg.agent_lr
        self.E = cfg.input_energy
        self.min_de = cfg.min_de
        self._rng = rng
        self._v_est: Array | None = None

    def update(self, e_t: Array, u_t: Array) -> None:
        """Correct recursive LS gradient step."""
        u2 = float(u_t @ u_t) + 1e-8
        self.B_est += self.lr * np.outer(e_t, u_t) / u2
        self._v_est = None

    def dominant_direction(self) -> Array | None:
        """Left singular vector of (B_est - I). None if no signal yet."""
        dB = self.B_est - np.eye(2)
        if np.linalg.norm(dB, 'fro') < 1e-4:
            return None
        U, _, _ = np.linalg.svd(dB)
        return U[:, 0]

    def cov_u(self) -> Array:
        if self._v_est is None:
            self._v_est = self.dominant_direction()
        v = self._v_est
        if v is None:
            return (self.E / 2.0) * np.eye(2)
        vp = np.array([-v[1], v[0]])
        return (self.E * (1 - self.min_de)) * np.outer(v, v) \
             + (self.E * self.min_de)       * np.outer(vp, vp)

    def sample_u(self) -> Tuple[Array, Array]:
        C = self.cov_u()
        return self._rng.multivariate_normal(np.zeros(2), C), C


# ---------------------------------------------------------------------------
# Episode geometry
# ---------------------------------------------------------------------------

def sample_geom(rng: np.random.Generator, cfg: Stage2Cfg) -> Tuple[Array, Array]:
    th = rng.uniform(0.0, 2 * math.pi)
    v_b = np.array([math.cos(th), math.sin(th)])
    dth = math.radians(rng.uniform(cfg.theta_min, cfg.theta_max))
    if rng.random() < 0.5:
        dth = -dth
    vp = np.array([-v_b[1], v_b[0]])
    v_q = math.cos(dth) * v_b + math.sin(dth) * vp
    return v_b, v_q / np.linalg.norm(v_q)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_ep(
    rng: np.random.Generator,
    cfg: Stage2Cfg,
    hyp: str,
    v_b: Array,
    v_q: Array,
    policy: str = "sra",   # "sra" | "probe"
) -> Dict:
    """
    policy="sra"  : adaptive SRAAgent
    policy="probe": fixed isotropic (external reference, no adaptation)
    """
    A = np.eye(2)
    if hyp == "B":
        B_true = np.eye(2) + cfg.delta_b * np.outer(v_b, v_b)
        dq = 0.0
    else:
        B_true = np.eye(2)
        # Match one-step variance to B signal
        dq = cfg.delta_b**2 * cfg.input_energy * 0.5

    agent = SRAAgent(cfg, rng) if policy == "sra" else None
    x = rng.normal(size=2) * 0.1

    residuals, actions = [], []
    de_b_traj, pe_traj, en_traj = [], [], []

    for _ in range(cfg.T):
        if policy == "sra":
            u, Cu = agent.sample_u()
        else:
            # Probe: fixed isotropic
            Cu = (cfg.input_energy / 2.0) * np.eye(2)
            u = rng.multivariate_normal(np.zeros(2), Cu)

        if hyp == "B":
            w = rng.normal(scale=cfg.sigma_w, size=2)
        else:
            Cw = cfg.sigma_w**2 * np.eye(2) + dq * np.outer(v_q, v_q)
            w = rng.multivariate_normal(np.zeros(2), Cw)

        x_next = A @ x + B_true @ u + w
        e_t = x_next - A @ x - (agent.B_est if agent else np.eye(2)) @ u

        if agent is not None:
            agent.update(e_t, u)

        tr = max(np.trace(Cu), 1e-9)
        de_b_traj.append(float(v_b @ Cu @ v_b / tr))
        pe_traj.append(float(np.linalg.eigvalsh(Cu).min()))
        en_traj.append(float(tr))
        residuals.append(e_t.copy())
        actions.append(u.copy())
        x = x_next

    return dict(
        res=np.array(residuals),
        acts=np.array(actions),
        de_b=de_b_traj,
        pe=pe_traj,
        en=en_traj,
    )


# ---------------------------------------------------------------------------
# Classifiers (self-contained, compatible with Stage 1)
# ---------------------------------------------------------------------------

def sigmoid(z: Array) -> Array:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def auc_score(scores: Array, labels: Array) -> float:
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, float)
    ss = scores[order]; i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and ss[j] == ss[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    a = (ranks[labels == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return float(max(a, 1.0 - a))


def standardize(Xtr: Array, Xte: Array) -> Tuple[Array, Array]:
    mu = Xtr.mean(0)
    sd = np.where(Xtr.std(0) < 1e-8, 1.0, Xtr.std(0))
    return (Xtr - mu) / sd, (Xte - mu) / sd


def fit_linear(Xtr, ytr, Xte, yte, rng, cfg):
    Xtr, Xte = standardize(Xtr, Xte)
    Xtr = np.c_[Xtr, np.ones(len(Xtr))]; Xte = np.c_[Xte, np.ones(len(Xte))]
    w = rng.normal(scale=0.02, size=Xtr.shape[1])
    for _ in range(cfg.train_steps):
        p = sigmoid(Xtr @ w)
        g = Xtr.T @ (p - ytr.astype(float)) / len(ytr) + 1e-3 * w
        g[-1] -= 1e-3 * w[-1]
        w -= cfg.lr_cls * g
    return auc_score(Xte @ w, yte)


def fit_rff(Xtr, ytr, Xte, yte, rng, cfg):
    Xtr_s, Xte_s = standardize(Xtr, Xte)
    samp = Xtr_s[rng.choice(len(Xtr_s), min(200, len(Xtr_s)), replace=False)]
    D = np.sum((samp[:, None] - samp[None])**2, axis=-1)
    med = np.median(D[D > 1e-9]) if np.any(D > 1e-9) else 1.0
    g = 1.0 / max(med, 1e-6)
    W = rng.normal(scale=math.sqrt(2 * g), size=(Xtr_s.shape[1], cfg.rff_dim))
    b = rng.uniform(0, 2 * math.pi, cfg.rff_dim)
    sc = math.sqrt(2.0 / cfg.rff_dim)
    Ztr = sc * np.cos(Xtr_s @ W + b)
    Zte = sc * np.cos(Xte_s @ W + b)
    return fit_linear(Ztr, ytr, Zte, yte, rng, cfg)


def classifier_suite(Xtr, ytr, Xte, yte, rng, cfg) -> Dict:
    return {
        "linear": fit_linear(Xtr, ytr, Xte, yte, rng, cfg),
        "rff":    fit_rff(Xtr, ytr, Xte, yte, rng, cfg),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def window_feats(eps: List[Dict], t0: int, t1: int) -> Array:
    return np.array([ep["res"][t0:t1].reshape(-1) for ep in eps])


def window_mean(eps: List[Dict], key: str, t0: int, t1: int) -> float:
    return float(np.mean([np.mean(ep[key][t0:t1]) for ep in eps]))


def evaluate(cfg: Stage2Cfg) -> Dict:
    rng = np.random.default_rng(cfg.seed)

    print("Running SRA episodes...")
    eps_B_sra, eps_Q_sra = [], []
    for i in range(cfg.n_ep):
        v_b, v_q = sample_geom(rng, cfg)
        eps_B_sra.append(run_ep(rng, cfg, "B", v_b, v_q, "sra"))
        eps_Q_sra.append(run_ep(rng, cfg, "Q", v_b, v_q, "sra"))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{cfg.n_ep}")

    print("Running probe episodes...")
    eps_B_prb, eps_Q_prb = [], []
    for _ in range(cfg.n_ep // 2):
        v_b, v_q = sample_geom(rng, cfg)
        eps_B_prb.append(run_ep(rng, cfg, "B", v_b, v_q, "probe"))
        eps_Q_prb.append(run_ep(rng, cfg, "Q", v_b, v_q, "probe"))

    t0e, t1e = cfg.early_start, cfg.early_end
    t0l, t1l = cfg.late_start, cfg.late_end

    n_tr = int(cfg.n_ep * cfg.n_train_frac)
    y_B = np.ones(cfg.n_ep, int)
    y_Q = np.zeros(cfg.n_ep, int)

    def split_eval(XB, XQ, n):
        nb = len(XB); nq = len(XQ)
        ntr_b = int(nb * cfg.n_train_frac); ntr_q = int(nq * cfg.n_train_frac)
        Xtr = np.r_[XB[:ntr_b], XQ[:ntr_q]]
        ytr = np.r_[np.ones(ntr_b, int), np.zeros(ntr_q, int)]
        Xte = np.r_[XB[ntr_b:], XQ[ntr_q:]]
        yte = np.r_[np.ones(nb - ntr_b, int), np.zeros(nq - ntr_q, int)]
        return classifier_suite(Xtr, ytr, Xte, yte, rng, cfg)

    print("Evaluating classifiers...")
    # SRA: early vs late
    aucs_sra_early = split_eval(window_feats(eps_B_sra, t0e, t1e),
                                window_feats(eps_Q_sra, t0e, t1e), cfg.n_ep)
    aucs_sra_late  = split_eval(window_feats(eps_B_sra, t0l, t1l),
                                window_feats(eps_Q_sra, t0l, t1l), cfg.n_ep)
    # Probe reference
    n_pr = cfg.n_ep // 2
    aucs_prb = split_eval(window_feats(eps_B_prb, t0e, t1e),
                          window_feats(eps_Q_prb, t0e, t1e), n_pr)

    # DirectionalEnergy_B trajectories
    de_B_early = window_mean(eps_B_sra, "de_b", t0e, t1e)
    de_B_late  = window_mean(eps_B_sra, "de_b", t0l, t1l)
    de_Q_early = window_mean(eps_Q_sra, "de_b", t0e, t1e)
    de_Q_late  = window_mean(eps_Q_sra, "de_b", t0l, t1l)
    de_prb     = window_mean(eps_Q_prb, "de_b", t0e, t1e)

    pe_Q_late  = window_mean(eps_Q_sra, "pe",   t0l, t1l)
    en_Q_late  = window_mean(eps_Q_sra, "en",   t0l, t1l)

    auc_sra_early_mean = float(np.mean(list(aucs_sra_early.values())))
    auc_sra_late_mean  = float(np.mean(list(aucs_sra_late.values())))
    auc_prb_mean       = float(np.mean(list(aucs_prb.values())))

    # Per-step mean DE_B trajectories for plotting
    de_B_traj = [float(np.mean([ep["de_b"][t] for ep in eps_B_sra])) for t in range(cfg.T)]
    de_Q_traj = [float(np.mean([ep["de_b"][t] for ep in eps_Q_sra])) for t in range(cfg.T)]

    # -----------------------------------------------------------------------
    # Stage 2 criteria
    # -----------------------------------------------------------------------
    # C1: H_B agent learns correctly — DE_B stays high
    de_b_high    = de_B_late > 0.65
    # C2: H_Q agent contaminates — DE_B drops from isotropic start (0.50)
    #     Note: contamination happens within 2–3 steps and stabilises,
    #     so we compare late to the INITIAL value (t=0 = 0.50), not early.
    de_q_low     = de_Q_late < 0.45
    de_q_drop    = de_Q_traj[0] - de_Q_late > 0.05   # absolute drop from start
    # C3: Endogenous contrast between hypotheses
    de_contrast  = de_B_late - de_Q_late > 0.25
    # C4: PE and energy preserved (collapse is directional, not energetic)
    pe_preserved = pe_Q_late >= cfg.pe_thresh
    en_preserved = en_Q_late >= cfg.energy_thresh

    stage2_pass = (de_b_high and de_q_low and de_q_drop and
                   de_contrast and pe_preserved and en_preserved)

    return dict(
        config=asdict(cfg),
        # DirectionalEnergy_B
        de_B_early=de_B_early, de_B_late=de_B_late,
        de_Q_early=de_Q_early, de_Q_late=de_Q_late,
        de_probe=de_prb,
        # PE / Energy
        pe_Q_late=pe_Q_late, en_Q_late=en_Q_late,
        # AUC
        auc_sra_early=aucs_sra_early, auc_sra_early_mean=auc_sra_early_mean,
        auc_sra_late=aucs_sra_late,   auc_sra_late_mean=auc_sra_late_mean,
        auc_probe=aucs_prb,           auc_probe_mean=auc_prb_mean,
        # Trajectories
        de_B_traj=de_B_traj,
        de_Q_traj=de_Q_traj,
        # Criteria
        criteria=dict(
            de_b_high=de_b_high,
            de_q_low=de_q_low,
            de_q_drop=de_q_drop,
            de_contrast=de_contrast,
            pe_preserved=pe_preserved,
            en_preserved=en_preserved,
        ),
        stage2_pass=stage2_pass,
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_summary(r: Dict) -> None:
    print("\nMOAT v5g Stage 2 — Endogenous Attribution Collapse")
    print("=" * 64)
    print("DirectionalEnergy_B (v_B^T cov_u v_B / trace):")
    print(f"  H_B  early={r['de_B_early']:.3f}  late={r['de_B_late']:.3f}  [should stay high]")
    print(f"  H_Q  early={r['de_Q_early']:.3f}  late={r['de_Q_late']:.3f}  [should drop]")
    print(f"  probe={r['de_probe']:.3f}  [isotropic reference: ~0.50]")
    print()
    print(f"PE (H_Q late):     {r['pe_Q_late']:.3f}  thresh={r['config']['pe_thresh']}  "
          f"{'PASS' if r['criteria']['pe_preserved'] else 'fail'}")
    print(f"Energy (H_Q late): {r['en_Q_late']:.3f}  thresh={r['config']['energy_thresh']}  "
          f"{'PASS' if r['criteria']['en_preserved'] else 'fail'}")
    print()
    print("AUC_residual (H_B vs H_Q classification):")
    print(f"  probe  (reference):  {r['auc_probe_mean']:.3f}  {r['auc_probe']}")
    print(f"  SRA early (t={r['config']['early_start']}-{r['config']['early_end']}): "
          f"{r['auc_sra_early_mean']:.3f}  {r['auc_sra_early']}")
    print(f"  SRA late  (t={r['config']['late_start']}-{r['config']['late_end']}):  "
          f"{r['auc_sra_late_mean']:.3f}  {r['auc_sra_late']}")
    print()
    print("Criteria:")
    print(f"  C1 de_b_high   (H_B late >{0.65:.2f}):             {'PASS' if r['criteria']['de_b_high'] else 'fail'}  ({r['de_B_late']:.3f})")
    print(f"  C2 de_q_low    (H_Q late <0.45):             {'PASS' if r['criteria']['de_q_low'] else 'fail'}  ({r['de_Q_late']:.3f})")
    print(f"  C3 de_q_drop   (drop from t=0 >0.05):        {'PASS' if r['criteria']['de_q_drop'] else 'fail'}  ({r['de_Q_traj'][0]:.3f} -> {r['de_Q_late']:.3f})")
    print(f"  C4 de_contrast (H_B-H_Q late >0.25):         {'PASS' if r['criteria']['de_contrast'] else 'fail'}  ({r['de_B_late'] - r['de_Q_late']:.3f})")
    print(f"  C5 pe_preserved:                              {'PASS' if r['criteria']['pe_preserved'] else 'fail'}  ({r['pe_Q_late']:.3f})")
    print(f"  C6 en_preserved:                              {'PASS' if r['criteria']['en_preserved'] else 'fail'}  ({r['en_Q_late']:.3f})")
    print()
    print(f"Stage 2 PASS: {'YES ✓' if r['stage2_pass'] else 'no ✗'}")
    print("=" * 64)
    # Mini trajectory
    T = r['config']['T']
    print("\nDirectionalEnergy_B trajectory (every 5 steps):")
    print("  t   H_B    H_Q")
    for t in range(0, T, 5):
        print(f"  {t:2d}  {r['de_B_traj'][t]:.3f}  {r['de_Q_traj'][t]:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-ep", type=int, default=600)
    parser.add_argument("--T", type=int, default=60)
    parser.add_argument("--out", type=Path, default=Path("moat_v5g_stage2_results.json"))
    parser.add_argument("--quick", action="store_true",
                        help="Small run for sanity check.")
    args = parser.parse_args()

    cfg = Stage2Cfg(seed=args.seed, n_ep=args.n_ep, T=args.T)
    if args.quick:
        cfg = Stage2Cfg(seed=args.seed, n_ep=150, T=60,
                        train_steps=80, rff_dim=80)

    r = evaluate(cfg)
    print_summary(r)
    args.out.write_text(json.dumps(r, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

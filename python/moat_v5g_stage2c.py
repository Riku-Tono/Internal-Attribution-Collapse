#!/usr/bin/env python
"""
MOAT v5g Stage 2c — Policy-Matched Replay

Stage 2b established:
  - endogenous DE_B depletion under H_Q (Stage 2a)
  - agent-internal misattribution: 88.8% of H_Q episodes have
    angle(v_est, v_Q) < angle(v_est, v_B) (Stage 2b)
  - SRA residual late AUC = 0.761 (high, unexplained)

Open question:
  Why is late AUC high?
  Is it because H_B and H_Q agents have DIFFERENT policies
  (H_B → v_B direction, H_Q → v_Q direction), making residuals
  distinguishable through policy-induced structure, not genuine
  residual-geometry distinguishability?

Stage 2c answer: Policy-matched replay.
  1. Save action sequences u_0..T from H_Q SRAAgent episodes.
  2. Replay the SAME actions in H_B environment → residuals r^{B|Q-actions}
  3. Replay the SAME actions in H_Q environment → residuals r^{Q|Q-actions}
  4. Measure AUC_replay(H_B | Q-actions vs H_Q | Q-actions)

If AUC_replay << AUC_sra_late:
  The SRA late AUC was inflated by policy-behavior differences.
  Under equal actions, residual content is less distinguishable.
  → Supports: policy-induced trajectory structure was the main source
    of AUC_residual differences, not residual geometry itself.

If AUC_replay ≈ AUC_sra_late:
  True residual content is distinguishable regardless of policy.
  → Supports: environment produces fundamentally different residuals
    under H_B vs H_Q even with identical actions.

Secondary claim (mechanistic):
  Under H_Q SRA actions (concentrated along v_Q, away from v_B),
  the H_B mean signal (delta_B * (v_B·u) * v_B) is weak because
  u is mostly along v_Q. This further weakens residual distinguishability
  under replayed actions compared to the probe policy.

Full claim structure:
  Stage 1:  external DE_B depletion → AUC collapse              [done]
  Stage 2a: SRAAgent → endogenous DE_B depletion                [done]
  Stage 2b: SRAAgent → internal misattribution (angle evidence)  [done]
  Stage 2c: policy-matched replay → decomposes AUC source       [this file]
  Stage 2d: residual AUC collapse in same endogenous loop       [open]
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
class Stage2cCfg:
    seed: int = 42
    n_ep: int = 600
    T: int = 60
    delta_b: float = 0.9
    sigma_w: float = 0.25
    input_energy: float = 2.0
    agent_lr: float = 0.15
    min_de: float = 0.15
    theta_min: float = 30.0
    theta_max: float = 150.0
    late_start: int = 38
    late_end: int = 55
    early_start: int = 3
    early_end: int = 18
    # Thresholds
    pe_thresh: float = 0.15
    energy_thresh: float = 1.0
    replay_drop_thresh: float = 0.08  # AUC_sra_late - AUC_replay > this → inflated
    # Classifiers
    rff_dim: int = 160
    train_steps: int = 200
    lr_cls: float = 0.08
    n_train_frac: float = 0.7


# ---------------------------------------------------------------------------
# SRAAgent (same as Stage 2a/b)
# ---------------------------------------------------------------------------

class SRAAgent:
    def __init__(self, cfg: Stage2cCfg, rng: np.random.Generator):
        self.B_est = np.eye(2, dtype=float)
        self.lr = cfg.agent_lr
        self.E = cfg.input_energy
        self.min_de = cfg.min_de
        self._rng = rng
        self._v_est: Array | None = None

    def update(self, e_t: Array, u_t: Array) -> None:
        u2 = float(u_t @ u_t) + 1e-8
        self.B_est += self.lr * np.outer(e_t, u_t) / u2
        self._v_est = None

    def dominant_direction(self) -> Array | None:
        dB = self.B_est - np.eye(2)
        if np.linalg.norm(dB, "fro") < 1e-4:
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
             + (self.E * self.min_de) * np.outer(vp, vp)

    def sample_u(self) -> Tuple[Array, Array]:
        C = self.cov_u()
        return self._rng.multivariate_normal(np.zeros(2), C), C


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def sample_geom(rng: np.random.Generator, cfg: Stage2cCfg) -> Tuple[Array, Array]:
    th = rng.uniform(0.0, 2 * math.pi)
    v_b = np.array([math.cos(th), math.sin(th)])
    dth = math.radians(rng.uniform(cfg.theta_min, cfg.theta_max))
    if rng.random() < 0.5:
        dth = -dth
    vp = np.array([-v_b[1], v_b[0]])
    v_q = math.cos(dth) * v_b + math.sin(dth) * vp
    return v_b, v_q / np.linalg.norm(v_q)


# ---------------------------------------------------------------------------
# Episode runners
# ---------------------------------------------------------------------------

def run_sra_ep(
    rng: np.random.Generator,
    cfg: Stage2cCfg,
    hyp: str,
    v_b: Array,
    v_q: Array,
) -> Dict:
    """Standard SRA adaptive episode. Returns residuals, actions, diagnostics."""
    A = np.eye(2)
    dq = 0.0 if hyp == "B" else cfg.delta_b**2 * cfg.input_energy * 0.5
    B_true = (np.eye(2) + cfg.delta_b * np.outer(v_b, v_b)) if hyp == "B" else np.eye(2)

    agent = SRAAgent(cfg, rng)
    x = rng.normal(size=2) * 0.1
    residuals, actions = [], []
    de_b_traj = []

    for _ in range(cfg.T):
        u, Cu = agent.sample_u()
        w = rng.normal(scale=cfg.sigma_w, size=2) if hyp == "B" else \
            rng.multivariate_normal(np.zeros(2),
                                    cfg.sigma_w**2 * np.eye(2) + dq * np.outer(v_q, v_q))
        x_next = A @ x + B_true @ u + w
        e_t = x_next - A @ x - agent.B_est @ u
        agent.update(e_t, u)
        residuals.append(e_t.copy())
        actions.append(u.copy())
        tr = max(np.trace(Cu), 1e-9)
        de_b_traj.append(float(v_b @ Cu @ v_b / tr))
        x = x_next

    # Attribution angle at episode end
    dB = agent.B_est - np.eye(2)
    if np.linalg.norm(dB, "fro") > 1e-4:
        U, _, _ = np.linalg.svd(dB)
        v_est = U[:, 0]
        a_vb = math.degrees(math.acos(float(np.clip(abs(v_est @ v_b), 0, 1))))
        a_vq = math.degrees(math.acos(float(np.clip(abs(v_est @ v_q), 0, 1))))
    else:
        a_vb, a_vq = 90.0, 90.0

    return dict(res=np.array(residuals), acts=np.array(actions),
                de_b=de_b_traj, angle_vb=a_vb, angle_vq=a_vq,
                v_b=v_b, v_q=v_q)


def run_replay_ep(
    rng: np.random.Generator,
    cfg: Stage2cCfg,
    hyp: str,
    v_b: Array,
    v_q: Array,
    fixed_actions: Array,
) -> Dict:
    """
    Replay episode: apply fixed external actions (from an H_Q SRA run).
    Residuals use B_est = I (neutral model, no adaptation).
    Residual definition: e_t = x_{t+1} - x_t - u_t
    Under H_B: e_t = delta_B*(v_B·u)*v_B + w_t  (mean signal along v_B)
    Under H_Q: e_t = burst_noise_along_v_Q + w_t  (zero mean, extra variance)
    """
    A = np.eye(2)
    dq = 0.0 if hyp == "B" else cfg.delta_b**2 * cfg.input_energy * 0.5
    B_true = (np.eye(2) + cfg.delta_b * np.outer(v_b, v_b)) if hyp == "B" else np.eye(2)

    x = rng.normal(size=2) * 0.1
    residuals = []

    for t in range(cfg.T):
        u = fixed_actions[t]
        w = rng.normal(scale=cfg.sigma_w, size=2) if hyp == "B" else \
            rng.multivariate_normal(np.zeros(2),
                                    cfg.sigma_w**2 * np.eye(2) + dq * np.outer(v_q, v_q))
        x_next = A @ x + B_true @ u + w
        # Neutral B_est = I: e_t = x_next - x - u = (B_true - I) @ u + w
        e_t = x_next - x - u
        residuals.append(e_t.copy())
        x = x_next

    return dict(res=np.array(residuals))


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def auc_score(scores, labels):
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


def standardize(Xtr, Xte):
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
        g[-1] -= 1e-3 * w[-1]; w -= cfg.lr_cls * g
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
    return fit_linear(sc * np.cos(Xtr_s @ W + b), ytr,
                      sc * np.cos(Xte_s @ W + b), yte, rng, cfg)


def split_eval(XB, XQ, rng, cfg):
    nb, nq = len(XB), len(XQ)
    ntrb = int(nb * cfg.n_train_frac); ntrq = int(nq * cfg.n_train_frac)
    Xtr = np.r_[XB[:ntrb], XQ[:ntrq]]
    ytr = np.r_[np.ones(ntrb, int), np.zeros(ntrq, int)]
    Xte = np.r_[XB[ntrb:], XQ[ntrq:]]
    yte = np.r_[np.ones(nb - ntrb, int), np.zeros(nq - ntrq, int)]
    return {"linear": fit_linear(Xtr, ytr, Xte, yte, rng, cfg),
            "rff":    fit_rff(Xtr, ytr, Xte, yte, rng, cfg)}


def window_feats(eps: List[Dict], key: str, t0: int, t1: int) -> Array:
    return np.array([ep[key][t0:t1].reshape(-1) for ep in eps])


def window_mean(eps: List[Dict], key: str, t0: int, t1: int) -> float:
    return float(np.mean([np.mean(ep[key][t0:t1]) for ep in eps]))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(cfg: Stage2cCfg) -> Dict:
    rng = np.random.default_rng(cfg.seed)
    t0l, t1l = cfg.late_start, cfg.late_end

    # -----------------------------------------------------------------------
    # 1. SRA adaptive episodes (Stage 2a/b reproduced)
    # -----------------------------------------------------------------------
    print("Running SRA episodes...")
    eps_B_sra, eps_Q_sra = [], []
    for i in range(cfg.n_ep):
        v_b, v_q = sample_geom(rng, cfg)
        eps_B_sra.append(run_sra_ep(rng, cfg, "B", v_b, v_q))
        eps_Q_sra.append(run_sra_ep(rng, cfg, "Q", v_b, v_q))
        if (i + 1) % 150 == 0:
            print(f"  SRA {i+1}/{cfg.n_ep}")

    # -----------------------------------------------------------------------
    # 2. Policy-matched replay using H_Q SRA actions
    # -----------------------------------------------------------------------
    print("Running policy-matched replay...")
    eps_B_replay, eps_Q_replay = [], []
    for i, (ep_B, ep_Q) in enumerate(zip(eps_B_sra, eps_Q_sra)):
        v_b, v_q = ep_Q["v_b"], ep_Q["v_q"]
        hq_actions = ep_Q["acts"]   # actions from H_Q SRA episode
        eps_B_replay.append(run_replay_ep(rng, cfg, "B", v_b, v_q, hq_actions))
        eps_Q_replay.append(run_replay_ep(rng, cfg, "Q", v_b, v_q, hq_actions))
        if (i + 1) % 150 == 0:
            print(f"  Replay {i+1}/{cfg.n_ep}")

    # -----------------------------------------------------------------------
    # 3. Classifiers
    # -----------------------------------------------------------------------
    print("Running classifiers...")

    # SRA (from Stage 2b)
    aucs_sra_late = split_eval(
        window_feats(eps_B_sra, "res", t0l, t1l),
        window_feats(eps_Q_sra, "res", t0l, t1l), rng, cfg)

    # Replay: same H_Q actions in both environments
    aucs_replay = split_eval(
        window_feats(eps_B_replay, "res", t0l, t1l),
        window_feats(eps_Q_replay, "res", t0l, t1l), rng, cfg)

    # Action-only on SRA (already shown in 2b to be ~0.52)
    aucs_act = split_eval(
        window_feats(eps_B_sra, "acts", t0l, t1l),
        window_feats(eps_Q_sra, "acts", t0l, t1l), rng, cfg)

    auc_sra_late_mean   = float(np.mean(list(aucs_sra_late.values())))
    auc_replay_mean     = float(np.mean(list(aucs_replay.values())))
    auc_act_mean        = float(np.mean(list(aucs_act.values())))

    # -----------------------------------------------------------------------
    # 4. Attribution angle (Stage 2b, reproduced)
    # -----------------------------------------------------------------------
    def attr_rates(eps_B, eps_Q):
        correct_B = sum(1 for ep in eps_B if ep["angle_vb"] < ep["angle_vq"])
        error_Q   = sum(1 for ep in eps_Q if ep["angle_vq"] < ep["angle_vb"])
        return correct_B / len(eps_B), error_Q / len(eps_Q)

    rate_B, rate_Q = attr_rates(eps_B_sra, eps_Q_sra)

    mean_a_vb_H_B = float(np.mean([ep["angle_vb"] for ep in eps_B_sra]))
    mean_a_vq_H_B = float(np.mean([ep["angle_vq"] for ep in eps_B_sra]))
    mean_a_vq_H_Q = float(np.mean([ep["angle_vq"] for ep in eps_Q_sra]))
    mean_a_vb_H_Q = float(np.mean([ep["angle_vb"] for ep in eps_Q_sra]))

    # -----------------------------------------------------------------------
    # 5. Stage 2a metrics
    # -----------------------------------------------------------------------
    de_B_late = window_mean(eps_B_sra, "de_b", t0l, t1l)
    de_Q_late = window_mean(eps_Q_sra, "de_b", t0l, t1l)
    de_B_traj = [float(np.mean([ep["de_b"][t] for ep in eps_B_sra])) for t in range(cfg.T)]
    de_Q_traj = [float(np.mean([ep["de_b"][t] for ep in eps_Q_sra])) for t in range(cfg.T)]

    # -----------------------------------------------------------------------
    # 6. Stage 2c criteria
    # -----------------------------------------------------------------------
    # From Stage 2a
    s2a_de_contrast = de_B_late - de_Q_late > 0.25
    # From Stage 2b
    s2b_attr_correct_B = rate_B >= 0.55
    s2b_attr_error_Q   = rate_Q >= 0.55
    # Stage 2c: replay drops AUC
    s2c_replay_drop   = auc_sra_late_mean - auc_replay_mean > cfg.replay_drop_thresh
    # Stage 2c: replay AUC lower than SRA
    s2c_replay_lower  = auc_replay_mean < auc_sra_late_mean

    stage2c_pass = s2a_de_contrast and s2b_attr_correct_B and \
                   s2b_attr_error_Q and s2c_replay_drop

    return dict(
        config=asdict(cfg),
        # AUC
        auc_sra_late=aucs_sra_late, auc_sra_late_mean=auc_sra_late_mean,
        auc_replay=aucs_replay,     auc_replay_mean=auc_replay_mean,
        auc_action=aucs_act,        auc_action_mean=auc_act_mean,
        auc_drop=auc_sra_late_mean - auc_replay_mean,
        # Attribution
        correct_rate_B=rate_B,
        error_rate_Q=rate_Q,
        mean_angle_B_vb=mean_a_vb_H_B, mean_angle_B_vq=mean_a_vq_H_B,
        mean_angle_Q_vq=mean_a_vq_H_Q, mean_angle_Q_vb=mean_a_vb_H_Q,
        # DE
        de_B_late=de_B_late, de_Q_late=de_Q_late,
        de_B_traj=de_B_traj, de_Q_traj=de_Q_traj,
        # Criteria
        criteria=dict(
            s2a_de_contrast=s2a_de_contrast,
            s2b_attr_correct_B=s2b_attr_correct_B,
            s2b_attr_error_Q=s2b_attr_error_Q,
            s2c_replay_drop=s2c_replay_drop,
            s2c_replay_lower=s2c_replay_lower,
        ),
        stage2c_pass=stage2c_pass,
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_summary(r: Dict) -> None:
    print("\nMOAT v5g Stage 2c — Policy-Matched Replay")
    print("=" * 64)
    drop = r["auc_drop"]

    print("\n── Stage 2a (directional depletion) ──────────────────────────")
    print(f"  DE_B:  H_B={r['de_B_late']:.3f}  H_Q={r['de_Q_late']:.3f}"
          f"  contrast={r['de_B_late']-r['de_Q_late']:.3f}")

    print("\n── Stage 2b (attribution angle) ───────────────────────────────")
    print(f"  H_B correct: {r['correct_rate_B']:.3f}"
          f"  (angle_vB={r['mean_angle_B_vb']:.1f}°  angle_vQ={r['mean_angle_B_vq']:.1f}°)")
    print(f"  H_Q error:   {r['error_rate_Q']:.3f}"
          f"  (angle_vQ={r['mean_angle_Q_vq']:.1f}°  angle_vB={r['mean_angle_Q_vb']:.1f}°)")

    print("\n── Stage 2c (policy-matched replay) ───────────────────────────")
    print(f"  SRA residual late (adaptive policy): {r['auc_sra_late_mean']:.3f}"
          f"  {r['auc_sra_late']}")
    print(f"  Action-only late  (policy signature): {r['auc_action_mean']:.3f}"
          f"  {r['auc_action']}")
    print(f"  Replay (H_Q actions, both hyp):       {r['auc_replay_mean']:.3f}"
          f"  {r['auc_replay']}")
    print(f"  AUC drop (SRA - replay): {drop:.3f}"
          f"  (threshold: {r['config']['replay_drop_thresh']})")

    print("\n── Interpretation ─────────────────────────────────────────────")
    if r["criteria"]["s2c_replay_drop"]:
        print("  ✓ AUC drop confirms: SRA late AUC was partly inflated by")
        print("    policy-behavior differences (H_B uses v_B policy, H_Q uses v_Q policy).")
        print("    Under equal actions, residual distinguishability is lower.")
    else:
        print("  ✗ AUC drop not confirmed: true residual content differs")
        print("    regardless of policy — environment itself is distinguishable.")

    if r["auc_replay_mean"] < 0.65:
        print(f"  Replay AUC = {r['auc_replay_mean']:.3f}:")
        print("    Under equal actions, H_B and H_Q are closer to indistinguishable.")
    else:
        print(f"  Replay AUC = {r['auc_replay_mean']:.3f}:")
        print("    Even with equal actions, environment produces distinguishable residuals.")

    print("\n── Criteria ───────────────────────────────────────────────────")
    for k, v in r["criteria"].items():
        print(f"  {k}: {'PASS' if v else 'fail'}")
    print(f"\n  Stage 2c PASS: {'YES ✓' if r['stage2c_pass'] else 'no ✗'}")

    print("\n── DirectionalEnergy_B trajectory ─────────────────────────────")
    print("  t   H_B    H_Q")
    for t in range(0, r["config"]["T"], 5):
        print(f"  {t:2d}  {r['de_B_traj'][t]:.3f}  {r['de_Q_traj'][t]:.3f}")
    print("=" * 64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-ep", type=int, default=600)
    parser.add_argument("--T", type=int, default=60)
    parser.add_argument("--out", type=Path, default=Path("moat_v5g_stage2c_results.json"))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    cfg = Stage2cCfg(seed=args.seed, n_ep=args.n_ep, T=args.T)
    if args.quick:
        cfg = Stage2cCfg(seed=args.seed, n_ep=150, T=60, train_steps=80, rff_dim=80)
    r = evaluate(cfg)
    print_summary(r)
    args.out.write_text(json.dumps(r, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

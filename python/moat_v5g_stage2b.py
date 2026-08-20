#!/usr/bin/env python
"""
MOAT v5g Stage 2b — Attribution Angle Diagnostics

Stage 2a confirmed: SRAAgent endogenously depletes DirectionalEnergy_B under
H_Q while preserving PE and total energy (all 6 criteria passed).

Open question from Stage 2a: late residual AUC = 0.761 (high).
This needs to be decomposed:

  Q1. Is the classifier using action/policy signatures rather than residuals?
      → action-only AUC (AUC_action_late)

  Q2. Is the agent actually misattributing? Is v_est pointing toward v_Q?
      → attribution angle: angle(v_est_final, v_Q) vs angle(v_est_final, v_B)

  Q3. What is the residual AUC under a FIXED probe policy (no policy signature)?
      → probe-late AUC (environment's intrinsic residual distinguishability)

Stage 2b claim (all must hold):
  S2b-C1: H_Q attribution error rate > 0.60
           Majority of H_Q episodes: angle(v_est, v_Q) < angle(v_est, v_B)
           → agent misattributes burst direction as B drift direction.

  S2b-C2: H_B attribution correct rate > 0.60
           Majority of H_B episodes: angle(v_est, v_B) < angle(v_est, v_Q)
           → agent correctly identifies B drift direction.

  S2b-C3: AUC_action_late > AUC_residual_late
           The classifier reads policy behavior more than residual content.
           → high late residual AUC is partly inflated by policy signature.

  S2b-C4: AUC_probe_late < AUC_residual_late
           Without policy signature (fixed probe), residual AUC is lower.
           → policy-induced behavior is driving the residual AUC difference.

If S2b-C1/C2 hold: the agent genuinely misattributes (internal evidence).
If S2b-C3/C4 hold: the external classifier's "success" is policy-driven, not
   residual-geometry-driven → residual distinguishability itself has degraded.

Together, S2b-C1 through S2b-C4 establish the conceptual separation:
  "External classifier distinguishability ≠ agent-internal attribution correctness"
  which is the core diagnostic claim of Stage 2 (not yet Recursive Attribution
  Poisoning, but the attribution-failure half of it).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

Array = np.ndarray


# ---------------------------------------------------------------------------
# Config (extends Stage 2a)
# ---------------------------------------------------------------------------

@dataclass
class Stage2bCfg:
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
    # Windows
    early_start: int = 3
    early_end: int = 18
    late_start: int = 38
    late_end: int = 55
    # Stage 2a thresholds (preserved)
    pe_thresh: float = 0.15
    energy_thresh: float = 1.0
    de_low_thresh: float = 0.45
    # Stage 2b thresholds
    attr_error_thresh: float = 0.55   # majority misattribute
    action_auc_inflation: float = 0.05 # action AUC > residual AUC by this margin
    probe_auc_gap: float = 0.05        # probe AUC < residual AUC by this margin
    # Classifiers
    rff_dim: int = 160
    train_steps: int = 200
    lr_cls: float = 0.08
    n_train_frac: float = 0.7


# ---------------------------------------------------------------------------
# SRAAgent (same as Stage 2a)
# ---------------------------------------------------------------------------

class SRAAgent:
    def __init__(self, cfg: Stage2bCfg, rng: np.random.Generator):
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
             + (self.E * self.min_de) * np.outer(vp, vp)

    def sample_u(self) -> Tuple[Array, Array]:
        C = self.cov_u()
        return self._rng.multivariate_normal(np.zeros(2), C), C


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def sample_geom(rng: np.random.Generator, cfg: Stage2bCfg) -> Tuple[Array, Array]:
    th = rng.uniform(0.0, 2 * math.pi)
    v_b = np.array([math.cos(th), math.sin(th)])
    dth = math.radians(rng.uniform(cfg.theta_min, cfg.theta_max))
    if rng.random() < 0.5:
        dth = -dth
    vp = np.array([-v_b[1], v_b[0]])
    v_q = math.cos(dth) * v_b + math.sin(dth) * vp
    return v_b, v_q / np.linalg.norm(v_q)


# ---------------------------------------------------------------------------
# Episode runner — now returns geometry + final B_est for attribution audit
# ---------------------------------------------------------------------------

def run_ep(
    rng: np.random.Generator,
    cfg: Stage2bCfg,
    hyp: str,
    v_b: Array,
    v_q: Array,
    policy: str = "sra",
    replay_actions: Array | None = None,  # for fixed-action replay
) -> Dict:
    A = np.eye(2)
    if hyp == "B":
        B_true = np.eye(2) + cfg.delta_b * np.outer(v_b, v_b)
        dq = 0.0
    else:
        B_true = np.eye(2)
        dq = cfg.delta_b**2 * cfg.input_energy * 0.5

    agent = SRAAgent(cfg, rng) if policy == "sra" else None
    x = rng.normal(size=2) * 0.1

    residuals, actions = [], []
    de_b_traj, pe_traj, en_traj = [], [], []

    for t in range(cfg.T):
        if replay_actions is not None:
            u = replay_actions[t]
            Cu = (cfg.input_energy / 2.0) * np.eye(2)  # nominal for energy tracking
        elif policy == "sra":
            u, Cu = agent.sample_u()
        else:
            Cu = (cfg.input_energy / 2.0) * np.eye(2)
            u = rng.multivariate_normal(np.zeros(2), Cu)

        if hyp == "B":
            w = rng.normal(scale=cfg.sigma_w, size=2)
        else:
            Cw = cfg.sigma_w**2 * np.eye(2) + dq * np.outer(v_q, v_q)
            w = rng.multivariate_normal(np.zeros(2), Cw)

        x_next = A @ x + B_true @ u + w
        B_est_for_e = agent.B_est if agent else np.eye(2)
        e_t = x_next - A @ x - B_est_for_e @ u

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
        # Attribution audit
        B_est_final=agent.B_est.copy() if agent else np.eye(2),
        v_b=v_b.copy(),
        v_q=v_q.copy(),
    )


# ---------------------------------------------------------------------------
# Attribution angle — the key Stage 2b diagnostic
# ---------------------------------------------------------------------------

def angle_between(a: Array, b: Array) -> float:
    """Angle in degrees between two unit vectors (unsigned)."""
    cos_a = float(np.clip(np.abs(a @ b), 0.0, 1.0))  # unsigned angle
    return math.degrees(math.acos(cos_a))


def attribution_angles(ep: Dict) -> Tuple[float, float]:
    """
    Returns (angle_to_v_B, angle_to_v_Q) for the agent's final v_est.
    Smaller angle = agent believes B drift is in that direction.
    """
    dB = ep["B_est_final"] - np.eye(2)
    if np.linalg.norm(dB, "fro") < 1e-4:
        return 90.0, 90.0  # no signal, effectively uninformative
    U, _, _ = np.linalg.svd(dB)
    v_est = U[:, 0]
    return angle_between(v_est, ep["v_b"]), angle_between(v_est, ep["v_q"])


def attribution_error_stats(eps_B: List[Dict], eps_Q: List[Dict]) -> Dict:
    """
    H_B episodes: correct if angle_to_v_B < angle_to_v_Q (agent found B direction).
    H_Q episodes: error if angle_to_v_Q < angle_to_v_B (agent mistook Q for B).
    """
    correct_B, total_B = 0, 0
    error_Q, total_Q = 0, 0
    angles_B_to_vB, angles_Q_to_vQ = [], []
    angles_B_to_vQ, angles_Q_to_vB = [], []

    for ep in eps_B:
        a_vb, a_vq = attribution_angles(ep)
        if not math.isnan(a_vb):
            total_B += 1
            if a_vb < a_vq:
                correct_B += 1
            angles_B_to_vB.append(a_vb)
            angles_B_to_vQ.append(a_vq)

    for ep in eps_Q:
        a_vb, a_vq = attribution_angles(ep)
        if not math.isnan(a_vb):
            total_Q += 1
            if a_vq < a_vb:   # misattribution: thinks Q direction is B direction
                error_Q += 1
            angles_Q_to_vQ.append(a_vq)
            angles_Q_to_vB.append(a_vb)

    return dict(
        # Rates
        correct_rate_B=correct_B / max(total_B, 1),
        error_rate_Q=error_Q / max(total_Q, 1),
        # Mean angles
        mean_angle_B_to_vB=float(np.mean(angles_B_to_vB)) if angles_B_to_vB else float("nan"),
        mean_angle_B_to_vQ=float(np.mean(angles_B_to_vQ)) if angles_B_to_vQ else float("nan"),
        mean_angle_Q_to_vQ=float(np.mean(angles_Q_to_vQ)) if angles_Q_to_vQ else float("nan"),
        mean_angle_Q_to_vB=float(np.mean(angles_Q_to_vB)) if angles_Q_to_vB else float("nan"),
    )


# ---------------------------------------------------------------------------
# Classifiers (self-contained)
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


def classifier_suite(Xtr, ytr, Xte, yte, rng, cfg) -> Dict:
    return {"linear": fit_linear(Xtr, ytr, Xte, yte, rng, cfg),
            "rff":    fit_rff(Xtr, ytr, Xte, yte, rng, cfg)}


def split_and_eval(XB, XQ, rng, cfg) -> Dict:
    nb, nq = len(XB), len(XQ)
    ntrb = int(nb * cfg.n_train_frac); ntrq = int(nq * cfg.n_train_frac)
    Xtr = np.r_[XB[:ntrb], XQ[:ntrq]]
    ytr = np.r_[np.ones(ntrb, int), np.zeros(ntrq, int)]
    Xte = np.r_[XB[ntrb:], XQ[ntrq:]]
    yte = np.r_[np.ones(nb - ntrb, int), np.zeros(nq - ntrq, int)]
    return classifier_suite(Xtr, ytr, Xte, yte, rng, cfg)


def window_feats(eps: List[Dict], key: str, t0: int, t1: int) -> Array:
    return np.array([ep[key][t0:t1].reshape(-1) for ep in eps])


def window_mean(eps: List[Dict], key: str, t0: int, t1: int) -> float:
    return float(np.mean([np.mean(ep[key][t0:t1]) for ep in eps]))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(cfg: Stage2bCfg) -> Dict:
    rng = np.random.default_rng(cfg.seed)
    t0l, t1l = cfg.late_start, cfg.late_end
    t0e, t1e = cfg.early_start, cfg.early_end

    # -----------------------------------------------------------------------
    # 1. SRA adaptive episodes (Stage 2a, reproduced)
    # -----------------------------------------------------------------------
    print("Running SRA episodes...")
    eps_B_sra, eps_Q_sra = [], []
    geoms: List[Tuple[Array, Array]] = []
    for i in range(cfg.n_ep):
        v_b, v_q = sample_geom(rng, cfg)
        geoms.append((v_b, v_q))
        eps_B_sra.append(run_ep(rng, cfg, "B", v_b, v_q, "sra"))
        eps_Q_sra.append(run_ep(rng, cfg, "Q", v_b, v_q, "sra"))
        if (i + 1) % 150 == 0:
            print(f"  {i+1}/{cfg.n_ep}")

    # -----------------------------------------------------------------------
    # 2. Probe episodes (fixed isotropic policy — removes policy signature)
    # -----------------------------------------------------------------------
    print("Running probe episodes...")
    n_prb = cfg.n_ep // 2
    eps_B_prb, eps_Q_prb = [], []
    for i in range(n_prb):
        v_b, v_q = sample_geom(rng, cfg)
        eps_B_prb.append(run_ep(rng, cfg, "B", v_b, v_q, "probe"))
        eps_Q_prb.append(run_ep(rng, cfg, "Q", v_b, v_q, "probe"))

    # -----------------------------------------------------------------------
    # 3. Classifiers
    # -----------------------------------------------------------------------
    print("Running classifiers...")

    # SRA residual (early and late)
    aucs_sra_early_res = split_and_eval(
        window_feats(eps_B_sra, "res", t0e, t1e),
        window_feats(eps_Q_sra, "res", t0e, t1e), rng, cfg)
    aucs_sra_late_res = split_and_eval(
        window_feats(eps_B_sra, "res", t0l, t1l),
        window_feats(eps_Q_sra, "res", t0l, t1l), rng, cfg)

    # SRA action-only (late) — measures policy signature leakage
    aucs_sra_late_act = split_and_eval(
        window_feats(eps_B_sra, "acts", t0l, t1l),
        window_feats(eps_Q_sra, "acts", t0l, t1l), rng, cfg)

    # Probe residual (late) — environment without policy signature
    aucs_prb_late_res = split_and_eval(
        window_feats(eps_B_prb, "res", t0l, t1l),
        window_feats(eps_Q_prb, "res", t0l, t1l), rng, cfg)

    auc_sra_late_res_mean = float(np.mean(list(aucs_sra_late_res.values())))
    auc_sra_late_act_mean = float(np.mean(list(aucs_sra_late_act.values())))
    auc_sra_early_res_mean = float(np.mean(list(aucs_sra_early_res.values())))
    auc_prb_late_res_mean = float(np.mean(list(aucs_prb_late_res.values())))

    # -----------------------------------------------------------------------
    # 4. Attribution angle analysis (Stage 2b primary)
    # -----------------------------------------------------------------------
    print("Computing attribution angles...")
    attr_stats = attribution_error_stats(eps_B_sra, eps_Q_sra)

    # -----------------------------------------------------------------------
    # 5. Stage 2a metrics (DirectionalEnergy_B, PE, Energy)
    # -----------------------------------------------------------------------
    de_B_traj = [float(np.mean([ep["de_b"][t] for ep in eps_B_sra])) for t in range(cfg.T)]
    de_Q_traj = [float(np.mean([ep["de_b"][t] for ep in eps_Q_sra])) for t in range(cfg.T)]
    de_B_late = window_mean(eps_B_sra, "de_b", t0l, t1l)
    de_Q_late = window_mean(eps_Q_sra, "de_b", t0l, t1l)
    pe_Q_late = window_mean(eps_Q_sra, "pe", t0l, t1l)
    en_Q_late = window_mean(eps_Q_sra, "en", t0l, t1l)

    # -----------------------------------------------------------------------
    # 6. Stage 2b criteria
    # -----------------------------------------------------------------------
    # From Stage 2a (preserved)
    s2a_de_b_high    = de_B_late > 0.65
    s2a_de_q_low     = de_Q_late < 0.45
    s2a_de_contrast  = de_B_late - de_Q_late > 0.25
    s2a_pe_preserved = pe_Q_late >= cfg.pe_thresh
    s2a_en_preserved = en_Q_late >= cfg.energy_thresh

    # Stage 2b
    s2b_attr_correct_B = attr_stats["correct_rate_B"] >= cfg.attr_error_thresh
    s2b_attr_error_Q   = attr_stats["error_rate_Q"]   >= cfg.attr_error_thresh
    # Action AUC > residual AUC → classifier reads policy, not geometry
    s2b_action_inflated = auc_sra_late_act_mean > auc_sra_late_res_mean - cfg.action_auc_inflation
    # Probe AUC < SRA residual AUC → policy signature inflates residual AUC
    s2b_probe_lower    = auc_prb_late_res_mean < auc_sra_late_res_mean - cfg.probe_auc_gap

    stage2a_pass = (s2a_de_b_high and s2a_de_q_low and s2a_de_contrast
                    and s2a_pe_preserved and s2a_en_preserved)
    stage2b_pass = s2b_attr_correct_B and s2b_attr_error_Q

    return dict(
        config=asdict(cfg),
        # Stage 2a
        de_B_late=de_B_late, de_Q_late=de_Q_late,
        pe_Q_late=pe_Q_late, en_Q_late=en_Q_late,
        de_B_traj=de_B_traj, de_Q_traj=de_Q_traj,
        # AUC decomposition
        auc_sra_early_res=aucs_sra_early_res,
        auc_sra_early_res_mean=auc_sra_early_res_mean,
        auc_sra_late_res=aucs_sra_late_res,
        auc_sra_late_res_mean=auc_sra_late_res_mean,
        auc_sra_late_act=aucs_sra_late_act,
        auc_sra_late_act_mean=auc_sra_late_act_mean,
        auc_prb_late_res=aucs_prb_late_res,
        auc_prb_late_res_mean=auc_prb_late_res_mean,
        # Attribution angles
        attribution=attr_stats,
        # Criteria
        stage2a_criteria=dict(
            de_b_high=s2a_de_b_high,
            de_q_low=s2a_de_q_low,
            de_contrast=s2a_de_contrast,
            pe_preserved=s2a_pe_preserved,
            en_preserved=s2a_en_preserved,
        ),
        stage2b_criteria=dict(
            attr_correct_B=s2b_attr_correct_B,
            attr_error_Q=s2b_attr_error_Q,
            action_inflated=s2b_action_inflated,
            probe_lower=s2b_probe_lower,
        ),
        stage2a_pass=stage2a_pass,
        stage2b_pass=stage2b_pass,
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_summary(r: Dict) -> None:
    print("\nMOAT v5g Stage 2b — Attribution Angle Diagnostics")
    print("=" * 68)

    print("\n── Stage 2a (reproduced) ──────────────────────────────────────────")
    print(f"  DirectionalEnergy_B:  H_B={r['de_B_late']:.3f}  H_Q={r['de_Q_late']:.3f}"
          f"  contrast={r['de_B_late']-r['de_Q_late']:.3f}")
    print(f"  PE (H_Q late): {r['pe_Q_late']:.3f}   Energy: {r['en_Q_late']:.3f}")
    for k, v in r["stage2a_criteria"].items():
        print(f"  {k}: {'PASS' if v else 'fail'}")
    print(f"  Stage 2a PASS: {'YES ✓' if r['stage2a_pass'] else 'no ✗'}")

    print("\n── Stage 2b: Attribution Angle ────────────────────────────────────")
    a = r["attribution"]
    print(f"  H_B episodes — correct attribution rate: {a['correct_rate_B']:.3f}")
    print(f"    mean angle(v_est, v_B) = {a['mean_angle_B_to_vB']:.1f}°  "
          f"angle(v_est, v_Q) = {a['mean_angle_B_to_vQ']:.1f}°")
    print(f"  H_Q episodes — misattribution rate:      {a['error_rate_Q']:.3f}")
    print(f"    mean angle(v_est, v_Q) = {a['mean_angle_Q_to_vQ']:.1f}°  "
          f"angle(v_est, v_B) = {a['mean_angle_Q_to_vB']:.1f}°")

    print("\n── AUC decomposition ──────────────────────────────────────────────")
    print(f"  SRA residual early  (t={r['config']['early_start']}-{r['config']['early_end']}): "
          f"{r['auc_sra_early_res_mean']:.3f}  {r['auc_sra_early_res']}")
    print(f"  SRA residual late   (t={r['config']['late_start']}-{r['config']['late_end']}): "
          f"{r['auc_sra_late_res_mean']:.3f}  {r['auc_sra_late_res']}")
    print(f"  SRA action-only late:                    "
          f"{r['auc_sra_late_act_mean']:.3f}  {r['auc_sra_late_act']}")
    print(f"  Probe residual late (no policy sig):     "
          f"{r['auc_prb_late_res_mean']:.3f}  {r['auc_prb_late_res']}")

    print("\n── Stage 2b criteria ──────────────────────────────────────────────")
    c = r["stage2b_criteria"]
    thresh = r["config"]["attr_error_thresh"]
    print(f"  S2b-C1 attr_correct_B (H_B rate>{thresh}):  "
          f"{'PASS' if c['attr_correct_B'] else 'fail'}  ({a['correct_rate_B']:.3f})")
    print(f"  S2b-C2 attr_error_Q   (H_Q rate>{thresh}):  "
          f"{'PASS' if c['attr_error_Q'] else 'fail'}  ({a['error_rate_Q']:.3f})")
    print(f"  S2b-C3 action_inflated (act AUC ≥ res AUC): "
          f"{'PASS' if c['action_inflated'] else 'fail'}  "
          f"(act={r['auc_sra_late_act_mean']:.3f} vs res={r['auc_sra_late_res_mean']:.3f})")
    print(f"  S2b-C4 probe_lower (probe < res - gap):      "
          f"{'PASS' if c['probe_lower'] else 'fail'}  "
          f"(probe={r['auc_prb_late_res_mean']:.3f} vs res={r['auc_sra_late_res_mean']:.3f})")
    print(f"\n  Stage 2b PASS: {'YES ✓' if r['stage2b_pass'] else 'no ✗'}")

    print("\n── DirectionalEnergy_B trajectory (every 5 steps) ─────────────────")
    print("  t   H_B    H_Q")
    for t in range(0, r["config"]["T"], 5):
        print(f"  {t:2d}  {r['de_B_traj'][t]:.3f}  {r['de_Q_traj'][t]:.3f}")
    print("=" * 68)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-ep", type=int, default=600)
    parser.add_argument("--T", type=int, default=60)
    parser.add_argument("--out", type=Path, default=Path("moat_v5g_stage2b_results.json"))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    cfg = Stage2bCfg(seed=args.seed, n_ep=args.n_ep, T=args.T)
    if args.quick:
        cfg = Stage2bCfg(seed=args.seed, n_ep=150, T=60, train_steps=80, rff_dim=80)
    r = evaluate(cfg)
    print_summary(r)
    args.out.write_text(json.dumps(r, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

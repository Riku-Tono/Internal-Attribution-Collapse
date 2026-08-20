#!/usr/bin/env python
"""
MOAT v5g Stage 2d — Multi-Directional Replay & Attribution Margin

Stage 2c showed:
  SRA residual late = 0.762 drops to 0.553 when H_Q actions are replayed.
  The high AUC was policy-geometry dependent, not intrinsic environment separability.

Stage 2d claim:
  AUC is a FUNCTION OF ACTION DIRECTION.
  - Actions along v_B (H_B-correct direction) → high AUC
  - Actions along v_Q (H_Q-misattributed direction) → low AUC
  - Isotropic probe → intermediate AUC
  - Discriminative oracle (action in the direction maximally separating B and Q) → high AUC

  Additionally: agent-internal attribution margin confirms misattribution.
  Under H_Q, the agent's LS update assigns higher "fit" to B-channel
  (B_est drift increases) rather than Q-channel (residual variance).

If these hold together, the Directional Collapse geometry is confirmed as:
  "Which direction actions point determines whether H_B vs H_Q can be
   distinguished from trajectory residuals."

This makes the SRA central claim precise:
  The wrong-attribution policy does not just fail to discriminate —
  it actively points actions in the direction that minimizes discriminability.
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


@dataclass
class Stage2dCfg:
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
    # thresholds
    pe_thresh: float = 0.15
    energy_thresh: float = 1.0
    rff_dim: int = 160
    train_steps: int = 200
    lr_cls: float = 0.08
    n_train_frac: float = 0.7


# ---------------------------------------------------------------------------
# SRAAgent
# ---------------------------------------------------------------------------

class SRAAgent:
    def __init__(self, cfg: Stage2dCfg, rng: np.random.Generator):
        self.B_est = np.eye(2, dtype=float)
        self.lr = cfg.agent_lr
        self.E = cfg.input_energy
        self.min_de = cfg.min_de
        self._rng = rng
        self._v_est: Array | None = None
        # attribution margin tracking
        self.B_drift_norm_history: List[float] = []
        self.residual_sq_history: List[float] = []

    def update(self, e_t: Array, u_t: Array) -> None:
        u2 = float(u_t @ u_t) + 1e-8
        delta = self.lr * np.outer(e_t, u_t) / u2
        self.B_est += delta
        self._v_est = None
        # Track: does B_est drift norm increase? (B-channel absorption)
        self.B_drift_norm_history.append(float(np.linalg.norm(self.B_est - np.eye(2), "fro")))
        self.residual_sq_history.append(float(e_t @ e_t))

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

    def attribution_margin(self) -> float:
        """
        Proxy for "how much is the agent attributing to B-channel vs residual noise?"
        = (final B drift norm) / (mean residual squared)
        High → agent has absorbed more into B_est relative to unexplained noise.
        Under H_Q: this should be HIGH despite no true B drift.
        Under H_B: this should also be HIGH (correct attribution).
        The key: under H_Q, v_est ≈ v_Q, not v_B.
        """
        if not self.B_drift_norm_history:
            return 0.0
        mean_res_sq = float(np.mean(self.residual_sq_history)) if self.residual_sq_history else 1.0
        return self.B_drift_norm_history[-1] / max(mean_res_sq, 1e-6)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def sample_geom(rng: np.random.Generator, cfg: Stage2dCfg) -> Tuple[Array, Array]:
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

def world_step(rng, cfg, hyp, x, B_true, v_q, dq, u):
    A = np.eye(2)
    if hyp == "B":
        w = rng.normal(scale=cfg.sigma_w, size=2)
    else:
        w = rng.multivariate_normal(np.zeros(2),
                                    cfg.sigma_w**2 * np.eye(2) + dq * np.outer(v_q, v_q))
    return A @ x + B_true @ u + w


def run_sra_ep(rng, cfg, hyp, v_b, v_q) -> Dict:
    """Adaptive SRA episode. Returns residuals, actions, diagnostics."""
    dq = 0.0 if hyp == "B" else cfg.delta_b**2 * cfg.input_energy * 0.5
    B_true = (np.eye(2) + cfg.delta_b * np.outer(v_b, v_b)) if hyp == "B" else np.eye(2)
    agent = SRAAgent(cfg, rng)
    x = rng.normal(size=2) * 0.1
    residuals, actions, de_b_traj = [], [], []

    for _ in range(cfg.T):
        u, Cu = agent.sample_u()
        x_next = world_step(rng, cfg, hyp, x, B_true, v_q, dq, u)
        e_t = x_next - x - agent.B_est @ u
        agent.update(e_t, u)
        residuals.append(e_t.copy())
        actions.append(u.copy())
        tr = max(np.trace(Cu), 1e-9)
        de_b_traj.append(float(v_b @ Cu @ v_b / tr))
        x = x_next

    dB = agent.B_est - np.eye(2)
    if np.linalg.norm(dB, "fro") > 1e-4:
        U, _, _ = np.linalg.svd(dB)
        v_est = U[:, 0]
        a_vb = math.degrees(math.acos(float(np.clip(abs(v_est @ v_b), 0, 1))))
        a_vq = math.degrees(math.acos(float(np.clip(abs(v_est @ v_q), 0, 1))))
    else:
        a_vb, a_vq = 90.0, 90.0

    return dict(res=np.array(residuals), acts=np.array(actions), de_b=de_b_traj,
                angle_vb=a_vb, angle_vq=a_vq, v_b=v_b, v_q=v_q,
                attr_margin=agent.attribution_margin())


def run_fixed_policy_ep(rng, cfg, hyp, v_b, v_q, policy_dir: str,
                         fixed_actions: Array | None = None) -> Dict:
    """
    Episode with a fixed (non-adaptive) policy.
    policy_dir in: 'isotropic', 'vB', 'vQ', 'discriminative', 'replay'
    Residual uses neutral B_est = I throughout (no adaptation).
    """
    dq = 0.0 if hyp == "B" else cfg.delta_b**2 * cfg.input_energy * 0.5
    B_true = (np.eye(2) + cfg.delta_b * np.outer(v_b, v_b)) if hyp == "B" else np.eye(2)
    x = rng.normal(size=2) * 0.1
    residuals = []

    # Discriminative direction: unit vector ⊥ to both v_B and v_Q is not ideal.
    # Better: direction midway between v_B and -projection onto v_Q.
    # Use: the direction that maximally separates B-drift signal from Q-burst noise.
    # Under H_B, e = delta_B * (v_B·u)*v_B + noise.
    # Maximize signal/noise: put u along v_B.
    # Under H_Q, e = burst along v_Q + noise.
    # To tell apart: project e onto v_B and v_Q.
    # Oracle: choose u along (v_B + perp) to maximise distinguishability.
    # Practically, oracle = concentrated along v_B (high v_B·u signal under H_B, 0 under H_Q).
    if policy_dir == "vb_oracle":
        # concentrate along v_B: maximises H_B mean signal
        d = v_b.copy()
        vp = np.array([-d[1], d[0]])
        Cu = (cfg.input_energy * (1 - cfg.min_de)) * np.outer(d, d) \
           + (cfg.input_energy * cfg.min_de) * np.outer(vp, vp)
    elif policy_dir == "vB":
        d = v_b.copy()
        vp = np.array([-d[1], d[0]])
        Cu = (cfg.input_energy * (1 - cfg.min_de)) * np.outer(d, d) \
           + (cfg.input_energy * cfg.min_de) * np.outer(vp, vp)
    elif policy_dir == "vQ":
        d = v_q.copy()
        vp = np.array([-d[1], d[0]])
        Cu = (cfg.input_energy * (1 - cfg.min_de)) * np.outer(d, d) \
           + (cfg.input_energy * cfg.min_de) * np.outer(vp, vp)
    else:  # isotropic
        Cu = (cfg.input_energy / 2.0) * np.eye(2)

    for t in range(cfg.T):
        if fixed_actions is not None:
            u = fixed_actions[t]
        else:
            u = rng.multivariate_normal(np.zeros(2), Cu)
        x_next = world_step(rng, cfg, hyp, x, B_true, v_q, dq, u)
        # neutral residual: e = x_next - x - u = (B_true - I)u + w
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
    mu = Xtr.mean(0); sd = np.where(Xtr.std(0) < 1e-8, 1.0, Xtr.std(0))
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


def split_eval(XB, XQ, rng, cfg) -> Dict:
    nb, nq = len(XB), len(XQ)
    ntrb = int(nb * cfg.n_train_frac); ntrq = int(nq * cfg.n_train_frac)
    Xtr = np.r_[XB[:ntrb], XQ[:ntrq]]
    ytr = np.r_[np.ones(ntrb, int), np.zeros(ntrq, int)]
    Xte = np.r_[XB[ntrb:], XQ[ntrq:]]
    yte = np.r_[np.ones(nb - ntrb, int), np.zeros(nq - ntrq, int)]
    return {"linear": fit_linear(Xtr, ytr, Xte, yte, rng, cfg),
            "rff":    fit_rff(Xtr, ytr, Xte, yte, rng, cfg)}


def mean_auc(d: Dict) -> float:
    return float(np.mean([v for v in d.values() if not math.isnan(v)]))


def window_feats(eps: List[Dict], key: str, t0: int, t1: int) -> Array:
    return np.array([ep[key][t0:t1].reshape(-1) for ep in eps])


def window_mean(eps, key, t0, t1) -> float:
    return float(np.mean([np.mean(ep[key][t0:t1]) for ep in eps]))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(cfg: Stage2dCfg) -> Dict:
    rng = np.random.default_rng(cfg.seed)
    t0l, t1l = cfg.late_start, cfg.late_end

    # -----------------------------------------------------------------------
    # 1. SRA adaptive episodes
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
    # 2. Multi-directional replay
    # -----------------------------------------------------------------------
    print("Running multi-directional replay...")
    replay_types = {
        "hq_actions":     None,       # will use H_Q SRA actions
        "hb_actions":     None,       # will use H_B SRA actions
        "isotropic":      "isotropic",
        "vB_policy":      "vB",
        "vQ_policy":      "vQ",
        "vb_oracle": "vb_oracle",
    }

    replay_eps: Dict[str, Tuple[List, List]] = {k: ([], []) for k in replay_types}

    for i, (ep_B, ep_Q) in enumerate(zip(eps_B_sra, eps_Q_sra)):
        v_b, v_q = ep_Q["v_b"], ep_Q["v_q"]

        for rtype, pdesc in replay_types.items():
            if rtype == "hq_actions":
                fa = ep_Q["acts"]
                rb = run_fixed_policy_ep(rng, cfg, "B", v_b, v_q, "replay", fa)
                rq = run_fixed_policy_ep(rng, cfg, "Q", v_b, v_q, "replay", fa)
            elif rtype == "hb_actions":
                fa = ep_B["acts"]
                rb = run_fixed_policy_ep(rng, cfg, "B", v_b, v_q, "replay", fa)
                rq = run_fixed_policy_ep(rng, cfg, "Q", v_b, v_q, "replay", fa)
            else:
                rb = run_fixed_policy_ep(rng, cfg, "B", v_b, v_q, pdesc)
                rq = run_fixed_policy_ep(rng, cfg, "Q", v_b, v_q, pdesc)
            replay_eps[rtype][0].append(rb)
            replay_eps[rtype][1].append(rq)

        if (i + 1) % 150 == 0:
            print(f"  Replay {i+1}/{cfg.n_ep}")

    # -----------------------------------------------------------------------
    # 3. Classifiers
    # -----------------------------------------------------------------------
    print("Running classifiers...")
    aucs_sra_late_res = split_eval(
        window_feats(eps_B_sra, "res", t0l, t1l),
        window_feats(eps_Q_sra, "res", t0l, t1l), rng, cfg)
    aucs_sra_late_act = split_eval(
        window_feats(eps_B_sra, "acts", t0l, t1l),
        window_feats(eps_Q_sra, "acts", t0l, t1l), rng, cfg)

    replay_aucs = {}
    for rtype, (eps_B_r, eps_Q_r) in replay_eps.items():
        d = split_eval(window_feats(eps_B_r, "res", t0l, t1l),
                       window_feats(eps_Q_r, "res", t0l, t1l), rng, cfg)
        replay_aucs[rtype] = {"per_clf": d, "mean": mean_auc(d)}

    # -----------------------------------------------------------------------
    # 4. Attribution angle & margin
    # -----------------------------------------------------------------------
    correct_B = sum(1 for ep in eps_B_sra if ep["angle_vb"] < ep["angle_vq"])
    error_Q   = sum(1 for ep in eps_Q_sra if ep["angle_vq"] < ep["angle_vb"])
    rate_B    = correct_B / len(eps_B_sra)
    rate_Q    = error_Q   / len(eps_Q_sra)

    mean_angle_B_vb = float(np.mean([ep["angle_vb"] for ep in eps_B_sra]))
    mean_angle_B_vq = float(np.mean([ep["angle_vq"] for ep in eps_B_sra]))
    mean_angle_Q_vq = float(np.mean([ep["angle_vq"] for ep in eps_Q_sra]))
    mean_angle_Q_vb = float(np.mean([ep["angle_vb"] for ep in eps_Q_sra]))

    mean_margin_B = float(np.mean([ep["attr_margin"] for ep in eps_B_sra]))
    mean_margin_Q = float(np.mean([ep["attr_margin"] for ep in eps_Q_sra]))

    # -----------------------------------------------------------------------
    # 5. Stage 2a metrics
    # -----------------------------------------------------------------------
    de_B_late = window_mean(eps_B_sra, "de_b", t0l, t1l)
    de_Q_late = window_mean(eps_Q_sra, "de_b", t0l, t1l)
    de_B_traj = [float(np.mean([ep["de_b"][t] for ep in eps_B_sra])) for t in range(cfg.T)]
    de_Q_traj = [float(np.mean([ep["de_b"][t] for ep in eps_Q_sra])) for t in range(cfg.T)]

    # -----------------------------------------------------------------------
    # 6. Criteria
    # -----------------------------------------------------------------------
    auc_sra = mean_auc(aucs_sra_late_res)
    auc_hq  = replay_aucs["hq_actions"]["mean"]
    auc_hb  = replay_aucs["hb_actions"]["mean"]
    auc_vB  = replay_aucs["vB_policy"]["mean"]
    auc_vQ  = replay_aucs["vQ_policy"]["mean"]
    auc_iso = replay_aucs["isotropic"]["mean"]
    auc_dis = replay_aucs["vb_oracle"]["mean"]

    # The key directional pattern:
    # vB_policy AUC > isotropic AUC > vQ_policy AUC
    c_vB_higher_than_vQ = auc_vB > auc_vQ + 0.05
    c_vQ_lower_than_iso = auc_vQ < auc_iso + 0.05
    c_hB_higher_than_hQ = auc_hb > auc_hq + 0.05
    c_sra_drop          = auc_sra - auc_hq > 0.08   # Stage 2c reproduced
    c_attr_correct_B    = rate_B >= 0.55
    c_attr_error_Q      = rate_Q >= 0.55
    c_margin_both_high  = mean_margin_B > 0.5 and mean_margin_Q > 0.5  # both absorb into B

    stage2d_pass = (c_vB_higher_than_vQ and c_hB_higher_than_hQ and
                    c_sra_drop and c_attr_correct_B and c_attr_error_Q)

    return dict(
        config=asdict(cfg),
        # AUC table
        auc_sra_late_res={"per_clf": aucs_sra_late_res, "mean": auc_sra},
        auc_sra_late_act={"per_clf": aucs_sra_late_act, "mean": mean_auc(aucs_sra_late_act)},
        replay_aucs={k: v for k, v in replay_aucs.items()},
        # Attribution
        correct_rate_B=rate_B, error_rate_Q=rate_Q,
        mean_angle_B_vb=mean_angle_B_vb, mean_angle_B_vq=mean_angle_B_vq,
        mean_angle_Q_vq=mean_angle_Q_vq, mean_angle_Q_vb=mean_angle_Q_vb,
        mean_attr_margin_B=mean_margin_B, mean_attr_margin_Q=mean_margin_Q,
        # DE
        de_B_late=de_B_late, de_Q_late=de_Q_late,
        de_B_traj=de_B_traj, de_Q_traj=de_Q_traj,
        # Criteria
        criteria=dict(
            vB_higher_than_vQ=c_vB_higher_than_vQ,
            vQ_lower_than_iso=c_vQ_lower_than_iso,
            hB_higher_than_hQ=c_hB_higher_than_hQ,
            sra_drop_from_2c=c_sra_drop,
            attr_correct_B=c_attr_correct_B,
            attr_error_Q=c_attr_error_Q,
            margin_both_high=c_margin_both_high,
        ),
        stage2d_pass=stage2d_pass,
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_summary(r: Dict) -> None:
    print("\nMOAT v5g Stage 2d — Multi-Directional Replay & Attribution Margin")
    print("=" * 70)

    print("\n── Directional AUC Table ─────────────────────────────────────────────")
    print("  Action source         AUC    (per classifier)")
    rows = [
        ("SRA adaptive (H_B pol)", r["auc_sra_late_res"]),
        ("SRA action-only (leak)", r["auc_sra_late_act"]),
        ("Replay: H_B actions",    r["replay_aucs"]["hb_actions"]),
        ("Replay: H_Q actions",    r["replay_aucs"]["hq_actions"]),
        ("Replay: vB policy",      r["replay_aucs"]["vB_policy"]),
        ("Replay: vQ policy",      r["replay_aucs"]["vQ_policy"]),
        ("Replay: isotropic",      r["replay_aucs"]["isotropic"]),
        ("Replay: vB_oracle (=vB)", r["replay_aucs"]["vb_oracle"]),
    ]
    for label, d in rows:
        m = d["mean"]
        clf = d["per_clf"]
        print(f"  {label:<28} {m:.3f}  lin={clf['linear']:.3f}  rff={clf['rff']:.3f}")

    print("\n── Attribution Angles ────────────────────────────────────────────────")
    print(f"  H_B: correct={r['correct_rate_B']:.3f}"
          f"  angle(v_est,v_B)={r['mean_angle_B_vb']:.1f}°"
          f"  angle(v_est,v_Q)={r['mean_angle_B_vq']:.1f}°")
    print(f"  H_Q: error  ={r['error_rate_Q']:.3f}"
          f"  angle(v_est,v_Q)={r['mean_angle_Q_vq']:.1f}°"
          f"  angle(v_est,v_B)={r['mean_angle_Q_vb']:.1f}°")

    print("\n── Attribution Margin (B_est drift norm / mean residual sq) ──────────")
    print(f"  H_B: {r['mean_attr_margin_B']:.3f}  (correct attribution → B drift)")
    print(f"  H_Q: {r['mean_attr_margin_Q']:.3f}  (misattribution → spurious B drift)")

    print("\n── Directional Energy ────────────────────────────────────────────────")
    print(f"  H_B late: {r['de_B_late']:.3f}  H_Q late: {r['de_Q_late']:.3f}"
          f"  contrast: {r['de_B_late']-r['de_Q_late']:.3f}")

    print("\n── Criteria ──────────────────────────────────────────────────────────")
    for k, v in r["criteria"].items():
        print(f"  {k}: {'PASS' if v else 'fail'}")
    print(f"\n  Stage 2d PASS: {'YES ✓' if r['stage2d_pass'] else 'no ✗'}")

    print("\n── Summary table for paper ───────────────────────────────────────────")
    print("  Stage 1:  external DE_B drop → AUC collapse              DONE")
    print("  Stage 2a: SRAAgent → endogenous DE_B depletion           DONE")
    print("  Stage 2b: attribution angle evidence (88.8% error)       DONE")
    print("  Stage 2c: policy-matched replay AUC drop 0.762→0.553     DONE")
    print("  Stage 2d: directional AUC table confirms action-AUC      "
          + ("DONE" if r["stage2d_pass"] else "partial"))
    print("  Stage 2e: residual AUC collapse in same adaptive loop    OPEN")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-ep", type=int, default=600)
    parser.add_argument("--T", type=int, default=60)
    parser.add_argument("--out", type=Path, default=Path("moat_v5g_stage2d_results.json"))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    cfg = Stage2dCfg(seed=args.seed, n_ep=args.n_ep, T=args.T)
    if args.quick:
        cfg = Stage2dCfg(seed=args.seed, n_ep=150, T=60, train_steps=80, rff_dim=80)
    r = evaluate(cfg)
    print_summary(r)
    args.out.write_text(json.dumps(r, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

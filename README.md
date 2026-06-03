# Internal-Attribution-Collapse
How Attribution Collapse became Internal Attribution Collapse
# Internal Attribution Collapse

### Policy-Geometry Dependent Evidence and Structural Misattribution in Adaptive Systems

## Overview

This repository contains the reference implementation, experimental framework, and supporting materials for **Internal Attribution Collapse**, a failure mode in adaptive systems where:

* External behavior remains distinguishable.
* Internal attribution becomes structurally incorrect.
* Policy adaptation alters the geometry of collected evidence.

The central result is:

> External classifier distinguishability and agent-internal attribution correctness are independent.

An adaptive agent may produce trajectories that are externally classifiable while internally learning the wrong structural explanation for the observed phenomenon.

---

## Core Idea

Traditional evaluation often assumes:

```
High predictive performance
        ↓
Correct understanding
```

This work demonstrates a counterexample.

An agent can:

1. Adapt successfully.
2. Produce apparently separable evidence.
3. Remain structurally wrong about the underlying cause.

The resulting evidence becomes dependent on the geometry induced by the policy itself.

---

## Repository Structure

```text
theory/
    theory.html

appendix/
    appendix.html

experiments/
    sra_agent.py
    moat_v5g.py
    replay_analysis.py

results/
    figures/
    logs/
    tables/
```

---

## Main Contributions

### Internal Attribution Collapse

A failure mode in which:

* evidence remains externally distinguishable,
* but updates are mapped into the wrong latent channel.

### Policy-Geometry Dependent Evidence

Observed separability may arise from policy-induced trajectory geometry rather than correct causal attribution.

### MOAT v5g

A benchmark framework designed to investigate:

* attribution fidelity,
* directional collapse,
* adaptive policy feedback,
* policy-dependent evidence generation.

### SRAAgent

A constructive minimal counterexample demonstrating:

* endogenous directional depletion,
* structural misattribution,
* divergence between external and internal evaluation.

---

## Experimental Highlights

### Stage 2b

Internal attribution failure:

* 88.8% of adversarial episodes align with the wrong structural direction.

### Stage 2c

Policy-matched replay:

* Adaptive AUC: 0.762
* Replay AUC: 0.553

The apparent signal is largely policy-geometry dependent.

### Stage 2d

AUC varies as a function of action direction.

This suggests that evidence quality cannot be interpreted independently from policy geometry.

---

## Positioning

This project is not presented as a replacement for existing approaches such as Active Bayesian Hypothesis Testing (ABHT).

Instead, it provides:

* a benchmark,
* a failure-mode analysis,
* a constructive counterexample,

for studying the gap between:

* external evidence quality,
* internal attribution fidelity.

---

## Current Status

Stage 1–2d: Completed

Stage 2e:

Residual AUC collapse inside the adaptive loop remains an open question.

Both positive and negative outcomes are scientifically informative.

---

## Citation

Draft status.

Multi-AI review relay:

* Claude
* Codex
* ChatGPT
* Gemini
* Perplexity

Baseline comparisons and further validation are ongoing.

---

## License

Research prototype.
Use at your own risk.
The agent may be confidently wrong.
That is, unfortunately, part of the point.

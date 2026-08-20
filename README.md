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

README.md
moat_v5g_stage2d_fixed
moat_v5g_stage2e
docs/
    index.html
    theory.html
    appendix.html
appendix/
    moat_v5g_stage2
    moat_v5g_stage2b
    moat_v5g_stage2c
    moat_v5g_stage2d
    moat_v5g_stage2d_fixed
results/
    moat_v5g_results_summary
    moat_v5g_stage2e_explore_results
    moat_v5g_stage2e_explore_per_seed
ja/
   docs/
    index.html
    theory.html
    appendix.html

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

### Stage 2e

Stage 2e tested whether the same endogenous adaptive loop can also collapse external residual AUC while preserving the mechanism established in Stages 2a–2d.

The revised exploratory protocol used:

* 27 PE-preserving parameter cells,
* 3 seeds per cell,
* 240 episodes per hypothesis and seed,
* a fixed 17-step late evaluation window,
* the conservative score `max(linear AUC, RFF AUC)`,
* simultaneous attribution, directional-depletion, PE, energy, action-leakage, and positive-control gates.

Result:

* 81 / 81 runs completed successfully.
* Exploratory candidates: **0 / 27 cells**.
* Conservative residual AUC range: **0.9535–1.0000**.
* Best cell mean residual AUC: **0.9744**, far above the collapse threshold of 0.60.
* Attribution error, directional depletion, PE preservation, energy preservation, and the vB-aligned positive control passed in **81 / 81 runs**.
* The action-only leakage gate passed in **68 / 81 runs**.

The untouched-seed confirmatory phase was not run because no exploratory cell met the predeclared candidate rule.

The Stage 2e result is therefore negative: the tested endogenous loop did not produce external residual AUC collapse. Internal structural misattribution and external nonlinear distinguishability continued to coexist.

---

## Evaluation Protocol

The Stage 2e implementation separates exploration from confirmation:

* **Exploration** may nominate a parameter cell but cannot declare a Stage 2e success.
* **Confirmation** requires one predeclared cell, fresh seeds, 600 episodes per hypothesis and seed, and 95% bootstrap bounds satisfying every mechanism and evaluation gate.

Low-PE settings cannot count as confirmation. In particular, `min_de=0.03` gives a minimum policy-covariance eigenvalue of `2.0 × 0.03 = 0.06`, below the benchmark PE threshold of 0.15. The PE-preserving grid instead uses `min_de ∈ {0.15, 0.10, 0.075}`.

The residual-collapse metric uses the stronger of the linear and nonlinear evaluators. Averaging the two would allow a near-chance linear score to hide substantial nonlinear separability.

Implementation note: the current reference evaluators are full-batch logistic classifiers on standardized raw and random Fourier features. Historical documents that call them “linear SVM” and “RFF-SVM” should be corrected or read as referring to these logistic implementations.

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

No ABHT agent was run in Stage 2e. The negative Stage 2e result therefore does not establish that ABHT already covers this geometry; that requires a separate direct baseline experiment.

---

## Current Status

Stage 1–2d: Completed.

Stage 2e PE-preserving exploratory grid: Completed.

Stage 2e outcome:

* Endogenous attribution failure: reproduced.
* Directional depletion with PE and energy preservation: reproduced.
* External residual AUC collapse inside the adaptive loop: not observed in the tested grid.
* Confirmatory fresh-seed phase: not triggered because there was no exploratory candidate.

The current supported claim remains narrower than external indistinguishability:

> An adaptive agent can be internally wrong while its residual trajectories remain externally classifiable. External evidence quality and internal attribution fidelity are distinct quantities.

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

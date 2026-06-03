# Internal Attribution Collapse

### 適応システムにおける Policy-Geometry Dependent Evidence と Structural Misattribution

## 概要

本リポジトリは **Internal Attribution Collapse** に関する理論・実験・実装をまとめた研究プロトタイプです。

Internal Attribution Collapse とは、

* 外部からは識別可能な挙動が観測されるにもかかわらず、
* エージェント内部では原因帰属が誤っており、
* その誤った帰属が方策を変化させ、
* 証拠そのものの幾何構造を変形してしまう

という失敗モードです。

本研究の中心命題は次の一文に集約されます。

> 外部識別可能性と内部帰属正確性は独立である。

---

## 問題意識

多くの評価手法では、

```text
高い性能
   ↓
正しい理解
```

が暗黙に仮定されています。

しかし本研究では、

* 外部性能は高い
* 識別性能も高い
* それでも内部的には誤った原因を学習している

という構造的反例を示します。

---

## リポジトリ構成

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

## 主な内容

### Internal Attribution Collapse

外部から観測される残差情報は保持されているにもかかわらず、

内部更新が誤った潜在チャネルへ投影される現象。

### Policy-Geometry Dependent Evidence

観測される識別可能性が、

真の因果構造ではなく方策によって形成された軌道幾何に依存する現象。

### MOAT v5g

以下を測定するためのベンチマーク。

* Attribution Fidelity
* Directional Collapse
* Adaptive Feedback
* Policy-Induced Evidence

### SRAAgent

本研究で用いる最小構成エージェント。

以下を再現する。

* Endogenous Directional Depletion
* Structural Misattribution
* Internal / External Dissociation

---

## 実験結果

### Stage 2b

内部帰属失敗率：

* 88.8%

敵対条件下で推定方向が誤った潜在方向へ収束。

### Stage 2c

Policy-Matched Replay

* Adaptive AUC: 0.762
* Replay AUC: 0.553

高い識別性能の相当部分が方策依存であることを示唆。

### Stage 2d

識別性能は行動方向に依存して変化。

証拠品質そのものが方策幾何と切り離せないことを示す。

---

## 本研究の位置付け

本研究は ABHT（Active Bayesian Hypothesis Testing）の代替理論を主張するものではありません。

むしろ、

* 失敗モード分析
* ベンチマーク設計
* 構成的反例

を通じて、

「外部的に正しく見えること」と
「内部的に正しく理解していること」

の差異を調べることを目的としています。

---

## 現在の状況

Stage 1–2d 完了。

Stage 2e：

適応ループ内部での残差崩壊は未確認。

成功しても失敗しても有益な結果になるよう設計されています。

---

## 注意

これは研究プロトタイプです。

高い性能を示していても、
内部では完全に間違っている可能性があります。

その状況自体を研究対象にしています。

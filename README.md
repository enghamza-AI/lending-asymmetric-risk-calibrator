# 📊 Lending Asymmetric Risk Calibrator

[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-orange?style=for-the-badge)](https://huggingface.co/spaces/enghamza-AI/loan-risk-calibrator)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github)](https://github.com/enghamza-AI/lending-asymmetric-risk-calibrator)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)](https://python.org)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-green?style=for-the-badge)](https://lightgbm.readthedocs.io)

> **Multi-objective loan threshold optimizer** — balances profit, recall, and fairness simultaneously across 2.3M Lending Club loans. Outputs a Pareto frontier with an optimal threshold recommendation driven by YAML business rules.

---

## The Problem

Every loan ML model outputs a probability — say 0.73. That number is not a decision.

Someone has to decide: at what probability do you reject an applicant?

The default answer (0.50) ignores three things that actually matter in banking:

- A missed defaulter costs 10–15× more than a rejected good customer
- Maximizing profit alone often produces discriminatory approval rates
- Risk officers, CEOs, and legal teams have conflicting objectives — and all three are right

This project builds a system that finds **every threshold where no single goal can improve without hurting another** — the Pareto frontier — and recommends the optimal bundle based on configurable business rules.

---

## What This Is Not

This is not a model accuracy project. LightGBM is trained once and frozen.

Everything interesting happens **after** the model — in the decision layer. That is the part most ML tutorials skip entirely.

---

## Live Demo

👉 **[Try it on HuggingFace Spaces](https://huggingface.co/spaces/enghamza-AI/loan-risk-calibrator)**

Adjust business parameters live — loss given default, approval rate target, fairness constraint — and watch the Pareto frontier update in real time.

---

## How It Works

```
2.3M Loan Records
      │
      ▼
LightGBM → P(default) per applicant
      │
      ▼
business_rules.yaml
(loss_given_default, approval_rate_target, fairness_constraint)
      │
      ▼
Threshold Sweep (0.01 → 0.99)
→ profit, recall, fairness gap at each threshold
      │
      ▼
Pareto Frontier
→ remove dominated solutions
→ scipy.optimize finds optimal bundle within constraints
      │
      ▼
Interactive Dashboard
→ Pareto chart · Optimal recommendation · Sensitivity analysis
```

---

## ML Concepts Implemented

| Concept | Where |
|---|---|
| Cost-sensitive threshold optimization | `threshold_optimizer.py` |
| Pareto frontier (multi-objective) | `threshold_optimizer.py` |
| Constrained optimization via scipy | `threshold_optimizer.py` |
| YAML-driven business rule config | `config/business_rules.yaml` |
| Fairness constraint (demographic parity) | `evaluator.py` |
| Sensitivity analysis | `evaluator.py` |
| LightGBM on imbalanced data | `train_model.py` |
| Chunked loading of 2.3M rows | `data_loader.py` |
| Stratified sampling at scale | `data_loader.py` |

---

## Dataset

**Lending Club Public Loan Data — 2007 to 2018**
2.3M loan records · 150+ features · ~18% default rate

Source: [Kaggle — wordsforthewise/lending-club](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

The dataset is imbalanced by design. Standard accuracy metrics are meaningless here — the system uses expected profit and recall as primary objectives.

---

## Project Structure

```
lending-asymmetric-risk-calibrator/
│
├── config/
│   └── business_rules.yaml       # all business parameters live here
│
├── src/
│   ├── data_loader.py            # chunked loading + stratified sampling
│   ├── feature_engineering.py   # cleaning, encoding, scaling
│   ├── train_model.py            # LightGBM training + probability output
│   ├── threshold_optimizer.py   # Pareto frontier + scipy optimization
│   └── evaluator.py             # metrics + sensitivity analysis
│
├── app.py                        # Streamlit dashboard for HuggingFace
├── requirements.txt
├── PROJECT_GUIDE.md              # deep study companion — read this first
└── README.md
```

---

## Business Rules Config

All business logic lives in one place — no hardcoded values in Python:

```yaml
cost_matrix:
  loss_given_default: 15000
  revenue_per_good_loan: 1200
  opportunity_cost_per_rejection: 300

constraints:
  min_approval_rate: 0.60
  max_fairness_gap: 0.05

optimization:
  fairness_attribute: "home_ownership"
  pareto_resolution: 0.01
```

Change a number, re-run, get a new recommendation. No code edits needed.

---

## Key Results

- Pareto frontier across profit · recall · fairness on held-out test set
- Optimal threshold recommendation with plain-English reasoning
- Sensitivity report showing how recommendation shifts when business assumptions change ±10%

---

## Research Connection

Multi-objective threshold optimization with fairness constraints is active published research:

- NeurIPS 2019 — *Fairness Constraints: Mechanisms for Fair Classification*
- ACM FAccT — annual conference track on fair decision systems

This project implements the core idea as a working, deployed engineering system.

---

## Stack

`LightGBM` · `scikit-learn` · `scipy.optimize` · `pandas` · `numpy` · `PyYAML` · `Streamlit` · `Plotly` · `joblib`

---

## Built By

**Hamza** — BSAI Student · AI Systems Engineering Track · Self-taught on top of degree

[![HuggingFace](https://img.shields.io/badge/🤗-enghamza--AI-orange)](https://huggingface.co/enghamza-AI)
[![GitHub](https://img.shields.io/badge/GitHub-enghamza--AI-black?logo=github)](https://github.com/enghamza-AI)

---

*Stage 2 · Week 4 — Cost Matrices + Threshold Tuning*

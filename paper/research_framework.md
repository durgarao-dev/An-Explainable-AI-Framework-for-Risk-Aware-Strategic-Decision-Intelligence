# Research Framework

## Title

**From Data to Strategic Action: An Explainable AI Framework for Risk-Aware Strategic Decision Intelligence**

## Central research problem

Organizations increasingly use predictive AI for business decisions, but prediction alone does not define how a decision should be made. A practical decision-intelligence system must connect predictive probability, explanation, risk/cost assumptions, operational capacity, and human judgment.

## Primary research question

How can explainable AI convert business predictive analytics into transparent, risk-aware strategic decisions while preserving a clear separation between model evidence, decision assumptions, and managerial judgment?

## Supporting questions

1. Does a calibrated tree-based model improve predictive discrimination relative to a transparent logistic-regression baseline?
2. How do probability calibration and explainability affect the usefulness of model outputs for decision-making?
3. How does an explicit utility function change the operational decision threshold compared with a default 0.50 classification threshold?
4. How stable are recommended actions when benefit, action-cost, and missed-opportunity assumptions change?
5. How does a fixed-capacity policy (top 20%) compare with utility-optimized selection?

## Empirical design

```text
UCI Bank Marketing
      |
      v
Leakage-aware feature policy
      |
      v
Train / Calibration / Decision / Test split
      |
      +--> Logistic Regression baseline
      |
      +--> Random Forest primary model
      |
      v
Probability calibration (isotonic regression)
      |
      v
Explainability (SHAP global + local)
      |
      v
Risk-aware utility model
      |
      v
Decision-threshold optimization
      |
      +--> Utility-optimized policy
      +--> Fixed top-20% capacity policy
      |
      v
Strategic decision analysis
```

## Leakage control

The `duration` feature is excluded because it records the duration of the current contact and is not known until after the contact occurs. The paper must not use it in pre-action targeting experiments.

The target `y` is excluded from features. The final test set is not used for model fitting, calibration, threshold selection, or utility tuning.

## Dataset

Use the official UCI Bank Marketing dataset. The dataset contains 45,211 observations and 16 input features in the `bank-full.csv` version; the target is whether the client subscribed to a term deposit. UCI provides DOI `10.24432/C5K306` and a CC BY 4.0 license. The repository downloads the official UCI archive and validates the expected row count.

## Models

### Baseline

Logistic Regression with class-balanced training.

### Primary

Random Forest with fixed, reproducible hyperparameters. The goal is not to claim universal superiority; it provides a nonlinear model whose predictions can be compared with a transparent linear baseline.

## Calibration

Isotonic regression is fitted on a dedicated calibration split. Calibrated probabilities are used in the decision layer.

## Explainability

SHAP is used to produce global feature importance and local feature-attribution outputs. Interpretations must remain model explanations rather than causal claims.

## Risk-aware decision model

For an action probability p:

**EU(action) - EU(no action) = p * benefit - action_cost + p * missed_opportunity_cost**

The implementation expresses this as an incremental-value function. Benefit, action cost, and missed-opportunity cost are explicit scenario assumptions and must never be described as observed financial values for the source bank.

## Decision policy

The primary threshold is selected on the dedicated decision-tuning split by maximizing expected incremental value under the base scenario. The test set is then evaluated once at that frozen threshold.

A fixed top-20% capacity policy is also evaluated to represent a manager who can contact only a bounded share of customers.

## Required results

### Predictive performance

- ROC-AUC
- Average Precision / PR-AUC
- Brier score
- Precision
- Recall
- F1-score
- False-positive rate
- Calibration ECE

### Decision performance

- Selected decision threshold
- Action rate
- Expected incremental value per 100 decisions
- Utility-optimized precision/recall/F1
- Top-20% precision
- Top-20% capture rate
- Scenario sensitivity across benefit/cost assumptions

### Explainability

- Mean absolute SHAP values
- Top 15 global features
- Five example local explanations

## IEEE/Elsevier paper structure

1. Introduction
2. Related Work
3. Research Gap and Research Questions
4. Data and Leakage-Aware Experimental Design
5. Predictive Modeling and Probability Calibration
6. Explainable AI Methodology
7. Risk-Aware Decision Intelligence Framework
8. Experimental Results
9. Discussion
10. Strategic and Managerial Implications
11. Responsible AI, Governance, and Limitations
12. Conclusion and Future Research

## Publication positioning

The strongest fit is the decision-support/information-systems literature rather than a generic ML benchmark paper. Recent *Decision Support Systems* work explicitly frames XAI around enhanced business decision-making and human-AI systems. A 2025 *Computers in Industry* study similarly evaluates context-aware XAI selection in business organizations with real data and end users. A 2024 *Knowledge-Based Systems* article combines SHAP and counterfactual reasoning in a strategic customer-development decision-support framework.

## Integrity requirements

Never fabricate results. Every number in the paper must come from `results/`. Scenario economics are assumptions. SHAP values explain the fitted model; they do not establish causation. Dataset benchmark performance does not establish production effectiveness.

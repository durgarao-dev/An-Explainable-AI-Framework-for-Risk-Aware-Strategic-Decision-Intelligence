# From Data to Strategic Action: An Explainable AI Framework for Risk-Aware Strategic Decision Intelligence

Empirical DBA-AI research project connecting predictive analytics, explainable AI, risk-aware utility, and strategic decision intelligence.

## Research objective

The project tests a concrete decision-support pipeline rather than stopping at predictive accuracy:

**Data -> Predictive AI -> Probability Calibration -> Explainability -> Risk/Utility -> Decision Policy -> Strategic Action**

The primary empirical case uses the UCI Bank Marketing benchmark. UCI describes the dataset as a business classification problem from direct marketing by a Portuguese banking institution, with 45,211 observations in `bank-full.csv` and a target indicating whether a client subscribed to a term deposit.

## Why this dataset

It provides a real business decision context suitable for a DBA paper. The repository intentionally excludes `duration`, because it measures the current contact duration and would not be known before the contact decision. This prevents a post-action leakage problem.

## Experimental design

| Split | Purpose |
|---|---|
| 60% | Model training |
| 15% | Probability calibration |
| 10% | Decision-threshold / utility tuning |
| 15% | Final untouched test |

Models:

- Logistic Regression: transparent baseline.
- Random Forest: primary nonlinear model.

The probabilities are calibrated with isotonic regression using a dedicated calibration split.

## Explainability

SHAP is used for:

- global feature attribution;
- local explanations for high-probability decisions;
- decision-maker interpretation of model outputs.

SHAP values are treated as explanations of the fitted model, not as causal effects.

## Risk-aware decision layer

The study does not treat `0.50` as a universal business decision threshold. Instead, the threshold is tuned on a dedicated decision set using an explicit incremental-value function:

```text
Incremental value =
    P(success) * benefit
    - action cost
    + P(success) * missed-opportunity cost
```

The economic values are **scenario assumptions** for sensitivity analysis, not observed financial values of the source bank.

The experiment also evaluates a fixed-capacity policy that acts on the top 20% of predicted opportunities.

## Data source

Official UCI record:

https://archive.ics.uci.edu/dataset/222/bank

Dataset DOI: `10.24432/C5K306`

License: CC BY 4.0

## RunPod execution

Use Python 3.11 for the pinned environment.

```bash
git clone https://github.com/durgarao-dev/An-Explainable-AI-Framework-for-Risk-Aware-Strategic-Decision-Intelligence.git
cd An-Explainable-AI-Framework-for-Risk-Aware-Strategic-Decision-Intelligence

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

PYTHONPATH=. python -m decision_intelligence.experiment
```

GPU is not required. The workload is CPU-based and the dataset is small enough for a normal CPU pod.

## Outputs

```text
results/
├── figures/
│   ├── roc_curve.png
│   ├── calibration_curve.png
│   ├── decision_utility_curve.png
│   └── shap_global_importance.png
├── metrics/
│   ├── dataset_summary.json
│   ├── dataset_splits.csv
│   ├── model_comparison.csv
│   ├── primary_decision_metrics.csv
│   ├── decision_threshold_sensitivity.csv
│   ├── capacity_policy_metrics.csv
│   ├── utility_scenario_sensitivity.csv
│   ├── shap_global_importance.csv
│   ├── shap_values_sample.csv
│   ├── local_explanations.csv
│   └── experiment_manifest.json
├── predictions/
│   └── test_decisions.csv
└── models/
    ├── random_forest.joblib
    └── isotonic_calibrator.joblib
```

These artifacts become the sole source of truth for the paper's Results section.

## Research questions

1. Does the nonlinear model improve predictive discrimination relative to the logistic baseline?
2. Does probability calibration improve the reliability of AI-generated decision probabilities?
3. How does utility-aware thresholding change the action set compared with a default classification threshold?
4. How sensitive are strategic actions to changes in benefit, cost, and missed-opportunity assumptions?
5. How does a constrained top-20% policy compare with utility optimization?
6. Which factors most strongly drive AI recommendations according to SHAP explanations?

## Publication framing

The target contribution is not a claim that Random Forest is universally better than Logistic Regression. The contribution is the **integration of prediction, calibration, explanation, explicit decision utility, and capacity constraints into an auditable strategic decision-intelligence workflow**.

This framing aligns closely with recent XAI and decision-support research in *Decision Support Systems*, *Knowledge-Based Systems*, and related venues.

## Research integrity

- No fabricated metrics.
- No manual insertion of results before the experiment runs.
- Scenario benefit/cost values are labeled assumptions.
- SHAP is not interpreted causally.
- Benchmark performance is not presented as production performance.
- The exact Git SHA and experiment configuration are recorded in the manifest.

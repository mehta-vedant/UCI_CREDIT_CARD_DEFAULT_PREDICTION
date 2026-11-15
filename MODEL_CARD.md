# Model card: UCI credit-risk prediction prototype

## Purpose

This repository estimates the probability that a client in the UCI Default of Credit Card Clients dataset will default on the following month's payment. It is a learning and engineering prototype for risk review. It is not a fraud detector, credit-approval engine, or a system that should make decisions about real people.

## Data

- **Source:** UCI Default of Credit Card Clients dataset.
- **Population:** 30,000 Taiwan credit-card clients; historical data from 2005.
- **Target:** `default.payment.next.month` (`1` is a recorded default, `0` is no recorded default).
- **Positive rate:** 6,636 / 30,000 (22.12%).
- **Inputs:** 23 supplied fields covering credit limit, coded demographics, six repayment-status values, six bill amounts, and six payment amounts. `ID` is not a model input.

The data is historical, geographically and temporally narrow, and includes demographic attributes. It cannot establish present-day performance, fairness, or legal suitability.

## Pipeline

The serialized artifact at `models/credit_risk_pipeline.joblib` contains the feature engineer, imputers, scaler, categorical encoder, and selected classifier together. The system:

1. validates all 23 required fields and key ranges;
2. repairs documented unknown education/marital-status codes;
3. derives utilisation, payment-ratio, repayment-history, average, trend, and flag features from supplied history;
4. imputes numeric values with a training-fold median, scales numeric values, and one-hot encodes categorical values;
5. predicts default probability with CatBoost; and
6. compares that probability to a threshold chosen on the validation partition.

## Training protocol and current result

The committed artifact was trained with seed 42 using stratified 60% / 20% / 20% train / validation / test partitions (18,000 / 6,000 / 6,000 rows). Logistic Regression, Random Forest, and CatBoost each use a three-fold shuffled `StratifiedKFold` parameter search; ROC-AUC selects the candidate. CatBoost was selected with:

```text
depth=6, iterations=150, learning_rate=0.03, l2_leaf_reg=3
```

The validation partition selected an F1-maximising review threshold of **0.245**. The untouched final test partition produced:

| Metric | Result |
| --- | ---: |
| ROC-AUC | 0.7805 |
| Average precision | 0.5582 |
| Precision | 0.4941 |
| Recall | 0.5998 |
| F1 | 0.5419 |
| Balanced accuracy | 0.7127 |
| Log loss | 0.4298 |

At that threshold, the test confusion matrix is TN=3,858, FP=815, FN=531, TP=796. These figures quantify performance on this historical split only; they are not business approval criteria.

## Interpretability

`python -m src.credit_risk.explain` creates a global SHAP summary for a reproducible 500-row source-data sample. The generated chart describes how the model tends to use features across that sample; it does not prove causation or provide an adequate explanation for an adverse real-world decision.

## Known limits and safeguards

- F1 was used for threshold selection as a reproducible technical policy, not because it represents a bank's cost, fairness, or regulatory policy.
- The project has no external validation, fairness analysis, drift monitoring, authentication, authorisation, audit logging, consent handling, or human-review workflow.
- A score is a statistical estimate, not a factual statement about a person. The Flask interface deliberately labels it as an educational review signal rather than an approval or rejection.
- Before any real use, require representative current data, a documented decision policy, calibration/fairness/privacy assessments, security controls, monitoring, legal review, and meaningful human appeal.

## Reproduce

Run `python -m src.credit_risk.training --seed 42` to recreate the standard search and its artifact, charts, and metadata. Add `--full-search` for the wider five-fold research grid. Run `python -m src.credit_risk.explain` after training to recreate the SHAP report.

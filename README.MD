# Credit risk prediction system

A reproducible machine-learning application that estimates the risk of next-month credit-card payment default. It uses the UCI Default of Credit Card Clients dataset: 30,000 historical Taiwan client records, with a binary target named `default.payment.next.month`.

This is an educational risk-estimation system. It is not an automated lending, denial, or fraud-detection system.

## What this project does

- Validates 23 client inputs before prediction.
- Builds past-only utilisation, payment-ratio, repayment-history, average, trend, and flag features.
- Fits all preprocessing inside a scikit-learn Pipeline.
- Compares Logistic Regression, Random Forest, and CatBoost through shuffled stratified cross-validation and a real parameter grid.
- Uses a three-way stratified split: 60% training, 20% validation, and 20% final test data.
- Selects the model using training cross-validation, selects an F1 threshold on validation data, and reports final metrics once on the untouched test partition.
- Produces ROC, precision-recall, calibration, and threshold-analysis charts.
- Saves a serialized inference pipeline plus metadata, then serves it through Flask HTML and JSON endpoints.

## Project layout

```text
data/raw/                 Source dataset
src/credit_risk/          Feature engineering, training, inference, configuration
models/                   Generated pipeline and metadata after training
reports/                  Generated metric summaries and charts after training
static/project-assets/    Aggregate EDA and recruiter-page visuals
templates/                Flask user interface
tests/                    Unit and application tests
codexcreateddocs2/        Beginner-to-expert project learning course
archive/                  Historical experiments and legacy implementations
```

## Run locally

Python 3.11 is the supported runtime.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
python -m src.credit_risk.training --seed 42
python -m src.credit_risk.explain
python -m src.credit_risk.showcase
pytest -q
python application.py
```

The default training run uses a repeatable three-fold search sized for a laptop. For a wider five-fold search, use `python -m src.credit_risk.training --seed 42 --full-search`; expect it to take substantially longer.

Open `http://localhost:5000` for the dashboard, `http://localhost:5000/project` for the recruiter-facing ML walkthrough, `http://localhost:5000/predictdata` for the form, or `http://localhost:5000/health` to check the loaded artifact. JSON predictions use `POST /api/predict`.

## Example request

```json
{
  "LIMIT_BAL": 100000,
  "SEX": 1,
  "EDUCATION": 2,
  "MARRIAGE": 1,
  "AGE": 30,
  "PAY_0": 0,
  "PAY_2": 0,
  "PAY_3": 0,
  "PAY_4": 0,
  "PAY_5": 0,
  "PAY_6": 0,
  "BILL_AMT1": 10000,
  "BILL_AMT2": 10000,
  "BILL_AMT3": 10000,
  "BILL_AMT4": 10000,
  "BILL_AMT5": 10000,
  "BILL_AMT6": 10000,
  "PAY_AMT1": 2000,
  "PAY_AMT2": 2000,
  "PAY_AMT3": 2000,
  "PAY_AMT4": 2000,
  "PAY_AMT5": 2000,
  "PAY_AMT6": 2000
}
```

## Docker

```powershell
docker build -t credit-risk .
docker run -p 5000:5000 credit-risk
```

The image includes the committed, evaluated pipeline in `models/`, so the deployed service uses the exact artifact described by `model_metadata.json`. Re-run the training command locally and commit the resulting artifact only when you intentionally want to publish a newly evaluated model.

## Deploy the application

Use Render as the primary host for the Flask application. In the Render dashboard, create a regular Docker Web Service from this GitHub repository, select the root `Dockerfile`, and set `GET /health` as the health-check path. See the [refactored project guide](project_guide/index.html#deploy) for the click-by-click deployment path and the comparison with Hugging Face Spaces and Vercel.

## Responsible use and current limitations

The source data is historical, limited to one population and period, and includes demographic attributes. A strong offline metric does not establish fairness, calibration, legal compliance, or suitability for a real credit decision. Production use requires current representative data, documented business costs, calibration monitoring, fairness and privacy review, security controls, human oversight, and an appeal process.

Read [MODEL_CARD.md](MODEL_CARD.md) for the current data, training protocol, exact metrics, explainability scope, and limitations.

## Learning material

Start with the beginner-first [refactored project guide](project_guide/index.html). It explains the live pipeline, metrics, code, local running, Render deployment, and interview story. The older [documentation course](codexcreateddocs2/README.html) is retained as a historical reverse-engineering reference for the pre-overhaul repository. For the current runnable system, treat this README, [MODEL_CARD.md](MODEL_CARD.md), the generated `models/model_metadata.json`, and the active `src/credit_risk/` code as the source of truth.

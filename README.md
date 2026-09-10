# Customer Churn Prediction and Analytics

A portfolio project that analyzes IBM Telco Customer Churn data, builds leakage-safe classification models, and exports Power BI-ready results. It focuses on analytical decision support.

## Objectives

- Explore churn patterns by customer, contract, service, tenure, and payment attributes.
- Compare Logistic Regression, Decision Tree, and Random Forest models.
- Export reproducible metrics, feature importance, charts, and scored holdout customers.

## Tech stack
Python, Pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Jupyter, and SQL.

## Dataset
Download the public **IBM Telco Customer Churn** CSV and save it exactly as `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`. The dataset is deliberately not bundled. The project raises a clear error rather than inventing data or results.

## Install and run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.train_models
```

Run from the project root. Outputs appear in `outputs/`; charts are in `outputs/figures/`. The supplied SQL is in `sql/churn_analysis.sql`.

## Methodology
`TotalCharges` is made numeric and blank values are median-imputed. The target is mapped from Yes/No to 1/0. A stratified train/test split occurs before fitting preprocessing. Numeric columns use median imputation and scaling; categorical columns use most-frequent imputation and one-hot encoding. These transformations are inside each model pipeline to prevent leakage.

The models are Logistic Regression (transparent baseline), Decision Tree (readable nonlinear rules), and Random Forest (more stable nonlinear ensemble). Compare accuracy, precision, recall, F1, and ROC-AUC in `outputs/model_results.csv`. Accuracy alone is insufficient because a model can score well by predicting the majority non-churn class while missing churners.

## EDA and Power BI
Generated charts cover contract, payment method, internet service, senior status, partner/dependents, support/security, tenure, and monthly charges. Import `churn_predictions.csv` and `model_results.csv` into Power BI. Recommended dashboard cards: total customers, churned customers, churn rate, and average monthly charges; visuals: churn by contract, tenure, payment method, internet service, and top model factors. No dashboard file is claimed or included.

## Project structure

```text
data/       source CSV (user supplied)
src/        cleaning, EDA, training, evaluation
sql/        interview-friendly analytical queries
reports/    concise analysis report
outputs/    generated results and figures
```

## Results and insights
Metrics and insights are generated only after the actual dataset is run. Treat segment differences and feature importance as associations, not causation. Use the outputs to select retention segments, then validate interventions experimentally.

## Future improvements
Add time-based validation, threshold tuning based on retention cost, monitoring for data drift, and controlled campaign experiments.


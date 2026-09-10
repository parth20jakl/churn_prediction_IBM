# Customer Churn Analysis Report

## Problem statement
Identify customer characteristics associated with churn and evaluate models that support retention prioritization.

## Dataset and methods
This project uses the IBM Telco Customer Churn CSV. Cleaning converts `TotalCharges` to numeric and replaces blank values with its median; records without a valid Yes/No churn target are excluded. Features include tenure bands, monthly-charge bands, and a count of subscribed services.

## Reproducible results
Run `python -m src.train_models` after placing the dataset in `data/`. The command generates EDA charts, model metrics, feature-importance files, and scored holdout records. Results are intentionally not stated here until that real-data run occurs.

## Interpretation and limitations
EDA and model importance show associations, not causal effects. Metrics must be interpreted using the generated test split; accuracy alone can obscure minority-class churn performance. Data is historical and may not reflect future pricing, products, or customer behavior.

## Recommendations
Base decisions on the generated segment rates and validated holdout performance. Prioritize segments with high observed churn and strong predicted risk, then measure retention interventions with controlled tests.

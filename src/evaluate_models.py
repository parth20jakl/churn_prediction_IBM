import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def evaluate_model(name, pipeline, x_test, y_test):
    predictions = pipeline.predict(x_test); probabilities = pipeline.predict_proba(x_test)[:, 1]
    return {"Model": name, "Accuracy": accuracy_score(y_test, predictions), "Precision": precision_score(y_test, predictions, zero_division=0), "Recall": recall_score(y_test, predictions, zero_division=0), "F1": f1_score(y_test, predictions, zero_division=0), "ROC_AUC": roc_auc_score(y_test, probabilities), "ConfusionMatrix": confusion_matrix(y_test, predictions).tolist()}

def feature_importance(pipeline, model_name):
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    model = pipeline.named_steps["model"]
    values = model.coef_.ravel() if model_name == "Logistic Regression" else model.feature_importances_
    return pd.DataFrame({"feature": names, "importance": values, "absolute_importance": abs(values)}).sort_values("absolute_importance", ascending=False)

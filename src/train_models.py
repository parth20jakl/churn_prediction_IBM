import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from src.data_preprocessing import load_data, clean_data, engineer_features, get_model_data
from src.eda import create_eda_figures
from src.evaluate_models import evaluate_model, feature_importance
from src.utils import OUTPUT_DIR, ensure_output_dirs

def main():
    ensure_output_dirs()
    data = engineer_features(clean_data(load_data()))
    create_eda_figures(data)
    x, y = get_model_data(data)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=42, stratify=y)
    numeric = x.select_dtypes(include="number").columns.tolist(); categorical = [c for c in x.columns if c not in numeric]
    preprocessor = ColumnTransformer([("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric), ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)])
    models = {"Logistic Regression": LogisticRegression(max_iter=2000, C=0.5), "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=42), "Random Forest": RandomForestClassifier(n_estimators=600, max_depth=12, min_samples_leaf=2, max_features="sqrt", random_state=42, n_jobs=1)}
    results=[]
    for name, model in models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)]); pipe.fit(x_train, y_train)
        results.append(evaluate_model(name, pipe, x_test, y_test))
        feature_importance(pipe, name).head(20).to_csv(OUTPUT_DIR / f"{name.lower().replace(' ', '_')}_importance.csv", index=False)
    results_frame = pd.DataFrame(results).drop(columns="ConfusionMatrix")
    results_frame.to_csv(OUTPUT_DIR / "model_results.csv", index=False)
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(data=results_frame, x="Model", y="Accuracy", color="#2878B5")
    ax.set_title("Holdout Accuracy Comparison"); ax.set_ylim(0, 1); ax.set_ylabel("Accuracy")
    for index, value in enumerate(results_frame["Accuracy"]): ax.text(index, value + .015, f"{value:.1%}", ha="center")
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "figures" / "model_accuracy_comparison.png", dpi=160); plt.close()
    # Export scored holdout customers for Power BI; this is not a production score.
    best = Pipeline([("preprocessor", preprocessor), ("model", models["Random Forest"])]).fit(x_train, y_train)
    export = data.loc[x_test.index, ["customerID", "Churn", "Contract", "tenure", "MonthlyCharges", "PaymentMethod", "InternetService"]].copy()
    export["ChurnProbability"] = best.predict_proba(x_test)[:, 1]; export.to_csv(OUTPUT_DIR / "churn_predictions.csv", index=False)
    print(results_frame.to_string(index=False))

if __name__ == "__main__": main()

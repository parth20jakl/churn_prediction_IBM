"""EDA figures and reproducible churn-rate summaries."""
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import FIGURE_DIR, ensure_output_dirs

def churn_rate_table(df, column):
    return df.groupby(column, observed=False)["ChurnFlag"].agg(Customers="size", ChurnRate="mean").reset_index()

def create_eda_figures(df):
    ensure_output_dirs(); sns.set_theme(style="whitegrid")
    categorical = ["Contract", "PaymentMethod", "InternetService", "SeniorCitizen", "Partner", "Dependents", "TechSupport", "OnlineSecurity"]
    for column in categorical:
        summary = churn_rate_table(df, column).sort_values("ChurnRate", ascending=False)
        ax = sns.barplot(data=summary, x=column, y="ChurnRate", color="#2878B5")
        ax.set_title(f"Churn rate by {column}"); ax.set_ylabel("Churn rate"); ax.tick_params(axis="x", rotation=30)
        plt.tight_layout(); plt.savefig(FIGURE_DIR / f"churn_by_{column}.png", dpi=160); plt.close()
    for column in ["tenure", "MonthlyCharges"]:
        ax = sns.histplot(data=df, x=column, hue="Churn", bins=25, stat="density", common_norm=False)
        ax.set_title(f"Distribution of {column} by churn")
        plt.tight_layout(); plt.savefig(FIGURE_DIR / f"{column}_by_churn.png", dpi=160); plt.close()

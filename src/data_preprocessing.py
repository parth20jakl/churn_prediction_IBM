import pandas as pd
from src.utils import DATA_PATH

def load_data(path=DATA_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. Add the IBM Telco CSV to data/.")
    return pd.read_csv(path)

def clean_data(df):
    """Convert TotalCharges, remove records without a usable target, and encode target."""
    data = df.copy()
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data["TotalCharges"] = data["TotalCharges"].fillna(data["TotalCharges"].median())
    data = data.dropna(subset=["Churn"])
    data["ChurnFlag"] = data["Churn"].map({"Yes": 1, "No": 0})
    if data["ChurnFlag"].isna().any():
        raise ValueError("Churn must contain Yes/No values.")
    return data

def engineer_features(df):
    data = df.copy()
    data["TenureGroup"] = pd.cut(data["tenure"], [-1, 12, 24, 48, 72], labels=["0-12", "13-24", "25-48", "49-72"])
    data["MonthlyChargeBand"] = pd.cut(data["MonthlyCharges"], [-1, 35, 70, float("inf")], labels=["Low", "Medium", "High"])
    service_columns = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    data["ServiceCount"] = data[service_columns].eq("Yes").sum(axis=1)
    data["AverageChargeToDate"] = data["TotalCharges"] / (data["tenure"] + 1)
    data["IsMonthToMonthFiber"] = ((data["Contract"] == "Month-to-month") & (data["InternetService"] == "Fiber optic")).astype(int)
    return data

def get_model_data(df):
    """Drop identifier and raw target; retain engineered predictors."""
    return df.drop(columns=["customerID", "Churn", "ChurnFlag"]), df["ChurnFlag"]

import getpass
import mysql.connector
from src.data_preprocessing import load_data, clean_data

DATABASE = "churn_prediction"
TABLE = "telco_churn"

def main():
    password = getpass.getpass("MySQL root password: ")
    connection = mysql.connector.connect(
        host="127.0.0.1", port=3306, user="root", password="Parth@18"
    )
    cursor = connection.cursor()
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DATABASE}")
    cursor.execute(f"USE {DATABASE}")
    data = clean_data(load_data()).drop(columns="ChurnFlag")
    cursor.execute(f"""CREATE TABLE IF NOT EXISTS {TABLE} (
        customerID VARCHAR(30), gender VARCHAR(10), SeniorCitizen INT,
        Partner VARCHAR(5), Dependents VARCHAR(5), tenure INT,
        PhoneService VARCHAR(5), MultipleLines VARCHAR(30), InternetService VARCHAR(30),
        OnlineSecurity VARCHAR(30), OnlineBackup VARCHAR(30), DeviceProtection VARCHAR(30),
        TechSupport VARCHAR(30), StreamingTV VARCHAR(30), StreamingMovies VARCHAR(30),
        Contract VARCHAR(30), PaperlessBilling VARCHAR(5), PaymentMethod VARCHAR(60),
        MonthlyCharges DECIMAL(10,2), TotalCharges DECIMAL(12,2), Churn VARCHAR(5)
    )""")

    columns = ", ".join(f"`{column}`" for column in data.columns)
    placeholders = ", ".join(["%s"] * len(data.columns))
    insert = f"INSERT INTO {TABLE} ({columns}) VALUES ({placeholders})"
    rows = [tuple(None if value is None else str(value) for value in row) for row in data.itertuples(index=False, name=None)]
    cursor.executemany(insert, rows)
    connection.commit()
    print(f"Imported {cursor.rowcount} rows into {DATABASE}.{TABLE}.")
    cursor.close(); connection.close()

if __name__ == "__main__":
    main()

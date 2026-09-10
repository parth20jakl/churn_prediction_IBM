-- Replace telco_churn with your imported CSV table.
SELECT COUNT(*) AS total_customers, SUM(CASE WHEN Churn='Yes' THEN 1 ELSE 0 END) AS churned_customers,
       AVG(CASE WHEN Churn='Yes' THEN 1.0 ELSE 0.0 END) AS churn_rate FROM telco_churn;

SELECT Contract, COUNT(*) AS customers, AVG(CASE WHEN Churn='Yes' THEN 1.0 ELSE 0.0 END) AS churn_rate
FROM telco_churn GROUP BY Contract ORDER BY churn_rate DESC;

SELECT PaymentMethod, COUNT(*) AS customers, AVG(CASE WHEN Churn='Yes' THEN 1.0 ELSE 0.0 END) AS churn_rate
FROM telco_churn GROUP BY PaymentMethod ORDER BY churn_rate DESC;

SELECT Churn, AVG(MonthlyCharges) AS average_monthly_charge FROM telco_churn GROUP BY Churn;

SELECT CASE WHEN tenure <= 12 THEN '0-12' WHEN tenure <= 24 THEN '13-24' WHEN tenure <= 48 THEN '25-48' ELSE '49-72' END AS tenure_group,
       COUNT(*) AS customers, AVG(CASE WHEN Churn='Yes' THEN 1.0 ELSE 0.0 END) AS churn_rate
FROM telco_churn GROUP BY 1 ORDER BY 1;

SELECT Contract, InternetService, PaymentMethod, COUNT(*) AS customers,
       AVG(CASE WHEN Churn='Yes' THEN 1.0 ELSE 0.0 END) AS churn_rate
FROM telco_churn GROUP BY Contract, InternetService, PaymentMethod HAVING COUNT(*) >= 20 ORDER BY churn_rate DESC;

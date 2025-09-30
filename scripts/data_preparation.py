# import pandas as pd
# from sklearn.preprocessing import StandardScaler

# # Lecture du dataset
# file_path = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
# df = pd.read_csv(file_path)

# #  Nettoyage des données
# df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(" ", pd.NA), errors="coerce")
# df = df.dropna(subset=["TotalCharges"])
# df = df.drop(columns=["customerID"])

# # Conversion de TotalCharges en numérique
# df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
# df = df.dropna(subset=["TotalCharges"])

# numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
# scaler = StandardScaler()
# df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# df.to_csv("telco_churn_prepared.csv", index=False)

# print("✅ Préparation terminée. Fichier sauvegardé sous telco_churn_prepared.csv")

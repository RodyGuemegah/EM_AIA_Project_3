import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. Chargement du dataset
file_path = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
df = pd.read_csv(file_path)

# 2. Nettoyage des données
# Suppression des lignes avec des valeurs manquantes
# (ou imputation si besoin)
df = df.replace(' ', pd.NA)
df = df.dropna()

# 3. Encodage des variables catégorielles
categorical_cols = df.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# 4. Normalisation des variables numériques
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# 5. Sauvegarde du dataset préparé
df.to_csv('telco_churn_prepared.csv', index=False)

print('Préparation des données terminée. Fichier sauvegardé sous telco_churn_prepared.csv.')

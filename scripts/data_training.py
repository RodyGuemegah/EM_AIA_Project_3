
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

#Chargerment des données
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")


print("Dimensions :", df.shape)
print("\nTypes de données :")
print(df.info())
print("\nExtrait du jeu de données :")
# display(df.head())

print("\nRépartition de la variable cible (Churn) :")
print(df["Churn"].value_counts())
sns.countplot(x="Churn", data=df)
plt.title("Répartition Churn vs Non-Churn")
plt.show()

# 5. Vérification des valeurs manquantes
print("\nValeurs manquantes :")
print(df.isna().sum())

# 6. Vérification des doublons
print("\nDoublons :", df.duplicated().sum())

#  Nettoyage des colonnes

if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)

# Conversion de colonnes numériques mal typées
if df["TotalCharges"].dtype == "object":
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Remplacer les valeurs manquantes par la médiane
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# 8. Encodage de la cible (Yes/No → 1/0)
df["Churn"] = LabelEncoder().fit_transform(df["Churn"])

# 9. Encodage des variables catégorielles (One-Hot Encoding)
df = pd.get_dummies(df, drop_first=True)

# 10. Vérification finale
print("\nDimensions après nettoyage :", df.shape)
print(df.head())

# 11. Séparation features (X) et target (y)
X = df.drop("Churn", axis=1)
y = df["Churn"]

print("\nShape X :", X.shape)
print("Shape y :", y.shape)

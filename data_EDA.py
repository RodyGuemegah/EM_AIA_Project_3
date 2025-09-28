import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score

# Chargement du dataset
df = pd.read_csv("telco_churn_prepared.csv")

# Encodage des colonnes
categorical_cols = ["Contract", "PaymentMethod", "InternetService", "AgeCategory"] if "AgeCategory" in df.columns else ["Contract", "PaymentMethod", "InternetService"]
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le
    
# Liste des colonnes représentant des services
services = ["PhoneService", "MultipleLines", "InternetService", 
            "OnlineSecurity", "OnlineBackup", "DeviceProtection", 
            "TechSupport", "StreamingTV", "StreamingMovies"]

# Convertion en binaire
for col in services:
    df[col] = df[col].apply(lambda x: 0 if x in ["No", "No internet service", "No phone service"] else 1)

# Créer la colonne "NumServices" = somme de tous les services
df["NumServices"] = df[services].sum(axis=1)
    

# Séparation des features
features = ["tenure", "MonthlyCharges", "TotalCharges", "NumServices"] + categorical_cols
X = df[features]
y = df["Churn"]


numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges", "NumServices"]
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:,1]

#  Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1], yticklabels=[0,1])
plt.xlabel("Prédiction")
plt.ylabel("Réel")
plt.title("Matrice de Confusion")
plt.show()


print(classification_report(y_test, y_pred))

#Courbe ROC et AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
auc_score = roc_auc_score(y_test, y_pred_prob)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.plot([0,1], [0,1], 'k--')
plt.xlabel("Taux de Faux Positifs (FPR)")
plt.ylabel("Taux de Vrai Positifs (TPR)")
plt.title("Courbe ROC")
plt.legend(loc="lower right")
plt.show()

print(f"AUC Score : {auc_score:.3f}")

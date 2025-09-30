import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


file_path = "data/telco_churn_prepared.csv"
df = pd.read_csv(file_path)

# Encodage de la cible Churn (Yes/No → 1/0)
if df['Churn'].dtype == 'object':
	df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

# 2. Séparation de la cible avant encodage
y = df['Churn']
X = df.drop('Churn', axis=1)

# 3. Encodage des variables catégorielles sur X uniquement
X = pd.get_dummies(X, drop_first=True)

# 4. Séparation en train/test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Création et entraînement du modèle
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 6. Prédiction et évaluation
from sklearn.metrics import classification_report, confusion_matrix
y_pred = model.predict(X_test)
print('Matrice de confusion :')
print(confusion_matrix(y_test, y_pred))
print('\nRapport de classification :')
print(classification_report(y_test, y_pred))

# Courbe ROC et score AUC
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Probabilités prédites pour la classe positive
y_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
auc_score = roc_auc_score(y_test, y_proba)

plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Taux de faux positifs (FPR)')
plt.ylabel('Taux de vrais positifs (TPR)')
plt.title('Courbe ROC - Régression Logistique')
plt.legend(loc='lower right')
plt.show()

print(f'Score AUC : {auc_score:.4f}')
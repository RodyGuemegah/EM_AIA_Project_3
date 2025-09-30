import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

file_path = 'data/telco_churn_prepared.csv'  # adapte le chemin si besoin

try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Fichier non trouvé : {file_path}")
    exit()

# Encodage de la cible 
if df['Churn'].dtype == 'object':
    df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})


X = df.drop('Churn', axis=1)
X = pd.get_dummies(X, drop_first=True)
y = df['Churn']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


y_pred_rf = rf_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred_rf)
print('Random Forest - Matrice de confusion :')
print(confusion_matrix(y_test, y_pred_rf))
print('\nRandom Forest - Rapport de classification :')
print(classification_report(y_test, y_pred_rf))

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non Churn', 'Churn'], yticklabels=['Non Churn', 'Churn'])
plt.xlabel('Prédictions')
plt.ylabel('Réel')
plt.title('Matrice de confusion - Random Forest')
plt.show()



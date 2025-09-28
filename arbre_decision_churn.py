import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


df = pd.read_csv("telco_churn_prepared.csv")
y = df['Churn']
X = df.drop('Churn', axis=1)
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(
    max_depth=5, #se limite à une profondeur de 5 pour éviter le surapprentissage
    min_samples_split=20, # un noeud doit contenir au moins 20 échantillons pour être divisé
    min_samples_leaf=10, # les feuilles  doivent contenir 10 échantillons
    
    random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print('Matrice de confusion :')
print(confusion_matrix(y_test, y_pred))
print('\nRapport de classification :')
print(classification_report(y_test, y_pred))

# Visualisation de l'arbre de décision
plt.figure(figsize=(30,15))
tree.plot_tree(model,
               filled=True,
               feature_names=X.columns,
               class_names=['No Churn', 'Churn'],
               fontsize=10,
               max_depth=4 #afficher seulement les 4 premiers niveaux
               )
plt.title("Arbre de Décision pour la Prédiction du Churn")
plt.show()

# Interprétation des règles de décision et complexité du modèle
from sklearn.tree import export_text
rules = export_text(model, feature_names=list(X.columns))
print('\nRègles de décision de l\'arbre :')
print(rules)

print(f"\nProfondeur de l'arbre : {model.get_depth()}")
print(f"Nombre de feuilles : {model.get_n_leaves()}")

# Modèle Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Prédictions et évaluation
y_pred_rf = rf.predict(X_test)
print('Random Forest - Matrice de confusion :')
print(confusion_matrix(y_test, y_pred_rf))
print('\nRandom Forest - Rapport de classification :')
print(classification_report(y_test, y_pred_rf))

# Importance des caractéristiques
importances = pd.Series(rf.feature_importances_, index=X.columns)
importances.sort_values(ascending=False).head(10).plot(kind='barh')
plt.title('Top 10 variables importantes (Random Forest)')
plt.xlabel('Importance')
plt.show()

# Optimisation des hyperparamètres
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 10],
    'min_samples_leaf': [1, 5]
}
gs = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1)
gs.fit(X_train, y_train)
print('Meilleurs hyperparamètres :', gs.best_params_)
print('Score sur le test :', gs.score(X_test, y_test))
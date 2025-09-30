import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_csv("data/telco_churn_prepared.csv")
y = df['Churn']
X = df.drop('Churn', axis=1)
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(
    max_depth=5, #se limite à une profondeur de 5
    min_samples_split=20, 
    min_samples_leaf=10,
    
    random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(' Arbre de decision - Matrice de confusion :')
print(confusion_matrix(y_test, y_pred))
print('\nRapport de classification :')
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

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

# from sklearn.tree import export_text
# rules = export_text(model, feature_names=list(X.columns))
# print('\nRègles de décision de l\'arbre :')
# print(rules)

# print(f"\nProfondeur de l'arbre : {model.get_depth()}")
# print(f"Nombre de feuilles : {model.get_n_leaves()}")


plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non Churn', 'Churn'], yticklabels=['Non Churn', 'Churn'])
plt.xlabel('Prédictions')
plt.ylabel('Réel')
plt.title('Matrice de confusion - Arbre de Décision')
plt.savefig('results/abre_decision/matrice_confusion_arbre.png')
plt.show() 
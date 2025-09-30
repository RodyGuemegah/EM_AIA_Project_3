# Prédiction du Churn Client - Documentation du Projet

Ce projet vise à prédire le churn (départ client) dans le secteur des télécommunications à partir de données réelles. Voici les principales étapes qui m'ont permis de réaliser ce projet :

## 1. Collecte et Préparation des Données
- On commence par le chargement du dataset brut (CSV).
- On met en place un nettoyage des données : suppression des colonnes inutiles, conversion des colonnes numériques mal typées, gestion des valeurs manquantes.
- Encodage des variables catégorielles (Yes/No → 0/1, autres catégories → indicatrices).
- Création de variables synthétiques (ex : nombre de services souscrits).
- Normalisation des variables numériques.
### Extrait du code illustant ce qui a été fait 
![Exemple de code](results\extrait_code\Screen_code_data_training.png)


## 2. Analyse Exploratoire (EDA)
- Visualisation de la répartition du churn et des variables clés (tenure, MonthlyCharges, TotalCharges, type de contrat, etc.)
- Analyse des profils de clients churnés selon plusieurs variable: selon Type de contrat, tenure .
- Repartition churn vs non churn (graphique).
- L'utilisation d'un Random Forest (exploratoire) à été nécéssaire afin d'identifier le Top 10 des variables les plus importantes:



## 3. Séparation des Données
- Séparation des features (X) et de la cible (y).
- Division du dataset en train/test pour évaluer les modèles.

## 4. Entraînement des Modèles
- Régression Logistique : modèle de base pour la classification binaire.
- Arbre de Décision : modèle interprétable, visualisation des règles.
- Random Forest : ensemble d’arbres pour améliorer la robustesse et la précision.

## 5. Évaluation et Comparaison
- Matrices de confusion, rapports de classification (accuracy, recall, precision, F1-score).
- Courbes ROC et calcul du score AUC.
- Visualisation de l’importance des variables pour la prédiction du churn.
- Ajustement du seuil de décision et pondération des classes pour mieux détecter les churners.

## 6. Sélection et Sauvegarde du Modèle
- Comparaison des performances des modèles.
- Choix du modèle le plus fiable selon la métrique la plus pertinente.
- Sauvegarde du modèle entraîné (fichier .pkl) pour une utilisation future.

## 7. Visualisation et Interprétation
- Dashboards et graphiques pour explorer les résultats et faciliter la prise de décision.

---

**Structure du projet :**
- `scripts/` : scripts Python pour la préparation, l’entraînement et l’évaluation des modèles.
- `notebooks/` : notebooks pour l’analyse exploratoire et la visualisation.
- `data/` : fichiers de données.
- `models/` : modèles sauvegardés.
- `result/` : graphiques et résultats intermédiaires.

**Auteur :** Rody Guemegah

**Usage :**
- Exécutez les scripts dans l’ordre pour reproduire le workflow complet.
- Modifiez les paramètres des modèles pour optimiser la détection du churn selon vos besoins.

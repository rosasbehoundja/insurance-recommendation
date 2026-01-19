# Insurance Recommendation System

Un système de recommandation de produits d'assurance basé sur la classification multi-label. Ce projet prédit quels produits d'assurance un client est susceptible de souscrire en fonction de ses caractéristiques démographiques et professionnelles.

## Description du Projet

Ce projet utilise des techniques de classification multi-label pour prédire la probabilité qu'un client souscrive à différents produits d'assurance. Le dataset contient des informations sur les clients (âge, sexe, statut marital, branche, profession) et leur historique de souscription à 21 produits d'assurance.

## Structure du Projet

```
insurance-recommendation/
├── app.py                      # Application web Flask
├── main.py                     # Point d'entrée principal du projet
├── retrain.py                  # Script de réentraînement automatique
├── pyproject.toml              # Configuration du projet et dépendances
├── README.md                   # Documentation du projet
├── data/
│   ├── Train.csv               # Données d'entraînement
│   ├── Test.csv                # Données de test
│   ├── SampleSubmission.csv    # Format de soumission attendu
│   ├── VariableDefinitions.txt # Description des variables
│   ├── eda/
│   │   └── train_.csv          # Données prétraitées
│   └── submissions/            # Fichiers de soumission générés
├── notebooks/
│   ├── eda-insurance-recommendation.ipynb    # Analyse exploratoire des données
│   └── train-insurance-recommendation.ipynb  # Notebook d'entraînement
├── scripts/
│   ├── preprocessing.py        # Prétraitement des données
│   ├── training.py             # Entraînement des modèles
│   └── predict.py              # Pipeline de prédiction
├── templates/
│   ├── index.html              # Page principale de prédiction
│   ├── result.html             # Page de résultats
│   ├── error.html              # Page d'erreur
│   └── retrain.html            # Page de réentraînement
└── weights/
    ├── catboost.pkl            # Modèle CatBoost entraîné
    ├── xgboost.pkl             # Modèle XGBoost entraîné
    ├── deep_learning.keras     # Modèle Deep Learning
    └── model_comparison.csv    # Comparaison des performances
```

## Variables du Dataset

| Variable | Description |
|----------|-------------|
| `ID` | Identifiant unique du client |
| `join_date` | Date d'adhésion |
| `sex` | Sexe du client |
| `marital_status` | Statut marital |
| `birth_year` | Année de naissance |
| `branch_code` | Code de la branche |
| `occupation_code` | Code de profession |
| `occupation_category_code` | Catégorie de profession |
| `P5DA`, `RIBP`, ... , `ECY3` | 21 codes produits (variables cibles) |

## Modèles Implémentés

Quatre modèles de classification multi-label ont été entraînés et comparés :

| Modèle | F1 Micro | F1 Macro | Precision | Recall | Hamming Loss |
|--------|----------|----------|-----------|--------|--------------|
| **XGBoost** | **0.846** | 0.451 | 0.863 | 0.831 | **0.032** |
| **Deep Learning** | 0.845 | 0.382 | 0.864 | 0.826 | 0.033 |
| CatBoost | 0.714 | 0.416 | 0.623 | 0.836 | 0.072 |
| Random Forest | 0.692 | 0.418 | 0.586 | 0.844 | 0.081 |

Le modèle **XGBoost** obtient les meilleures performances globales.

## Installation

### Prérequis

- Python >= 3.12
- uv (gestionnaire de packages recommandé) ou pip

### Installation avec uv

```bash
# Cloner le repository
git clone https://github.com/rosasbehoundja/insurance-recommendation.git
cd insurance-recommendation

# Installer les dépendances avec uv
uv sync
```

### Installation avec pip

```bash
# Cloner le repository
git clone https://github.com/rosasbehoundja/insurance-recommendation.git
cd insurance-recommendation

# Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate     # Windows

# Installer les dépendances
pip install -r requirements.txt
```

## Utilisation

### Pipeline complet

```bash
# Exécuter le pipeline complet (prétraitement + prédiction)
python main.py --full-pipeline
```

### Prétraitement des données

```bash
# Prétraiter les données d'entraînement
python main.py --preprocess --input data/Train.csv --output data/eda/train_.csv
```

### Prédiction

```bash
# Faire des prédictions avec un modèle spécifique
python main.py --predict --model catboost --test data/Test.csv --output submission.csv

# Modèles disponibles: xgboost, catboost, random_forest, deep_learning
python main.py --predict --model xgboost --test data/Test.csv
```

### Utilisation via Python

```python
from scripts.predict import predict, create_submission
from scripts.preprocessing import preprocess
import pandas as pd

# Charger et prédire
df_test = pd.read_csv("data/Test.csv")
predictions = predict(df_test, model_name="xgboost", weights_dir="weights")

# Créer le fichier de soumission
create_submission(df_test['ID'], predictions, "submission.csv")
```

## Application Web

L'application web Flask permet de faire des prédictions via une interface utilisateur et de gérer le réentraînement des modèles.

### Lancement

```bash
python app.py
```

L'application est accessible sur `http://localhost:5000`.

### Pages disponibles

| Route | Description |
|-------|-------------|
| `/` | Page principale de prédiction |
| `/retrain` | Interface de réentraînement des modèles |

### API REST

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/api/predict` | POST | Faire une prédiction |
| `/api/retrain` | POST | Lancer un réentraînement |
| `/api/retrain/schedule` | POST | Programmer un réentraînement périodique |
| `/api/retrain/schedule` | DELETE | Arrêter la programmation |
| `/api/retrain/status` | GET | Vérifier le statut de la programmation |

## Réentraînement Automatique

Le script `retrain.py` permet de réentraîner les modèles automatiquement.

### Ligne de commande

```bash
# Réentraîner tous les modèles une fois
python retrain.py

# Réentraîner un modèle spécifique
python retrain.py --model xgboost
python retrain.py --model deep_learning

# Réentraînement programmé (toutes les heures)
python retrain.py --schedule 3600

# Réentraînement programmé (tous les jours)
python retrain.py --schedule 86400
```

### Via l'interface web

1. Accéder à `http://localhost:5000/retrain`
2. Choisir le modèle et lancer le réentraînement manuel
3. Ou configurer un réentraînement programmé avec l'intervalle souhaité

### Utilisation programmatique

```python
from retrain import retrain_models, run_scheduled
import threading

# Réentraînement simple
success = retrain_models(model="all")

# Réentraînement programmé avec arrêt possible
stop_event = threading.Event()
thread = threading.Thread(
    target=run_scheduled,
    args=(3600, "all", stop_event),
    daemon=True
)
thread.start()

# Pour arrêter
stop_event.set()
```

## Pipeline de Traitement

### 1. Prétraitement (`scripts/preprocessing.py`)

- Conversion et standardisation des dates
- Calcul de l'âge à partir de l'année de naissance
- Normalisation des variables catégorielles
- One-Hot Encoding des variables catégorielles
- Standardisation des variables numériques

### 2. Entraînement (`scripts/training.py`)

- **Random Forest** : Ensemble d'arbres de décision
- **XGBoost** : Gradient Boosting optimisé
- **CatBoost** : Gradient Boosting pour données catégorielles
- **Deep Learning** : Réseau de neurones avec TensorFlow/Keras

### 3. Prédiction (`scripts/predict.py`)

- Chargement automatique des modèles et transformers
- Prétraitement des nouvelles données
- Génération des prédictions multi-label

## Dépendances

```toml
dependencies = [
    "flask>=3.0.0",
    "matplotlib>=3.10.8",
    "numpy>=2.4.1",
    "pandas>=2.3.3",
    "seaborn>=0.13.2",
    "scikit-learn>=1.4.0",
    "xgboost>=2.0.0",
    "catboost>=1.2.0",
    "tensorflow>=2.15.0",
]
```

## Notebooks

- **`eda-insurance-recommendation.ipynb`** : Analyse exploratoire des données, visualisations et insights
- **`train-insurance-recommendation.ipynb`** : Entraînement interactif des modèles et comparaison des performances

## Auteurs

Projet réalisé par:

- Rosas Behoundja
- Maurel Dossa
- Ronel Rodriguez


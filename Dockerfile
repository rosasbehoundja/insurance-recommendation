# 1. Utiliser une image Python officielle et légère
FROM python:3.9-slim

# 2. Définir le répertoire de travail à l'intérieur du conteneur
WORKDIR /app

# 3. Installer les dépendances système nécessaires (pour compiler XGBoost/CatBoost)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Copier le fichier des dépendances en premier (optimise le cache Docker)
COPY requirements.txt .

# 5. Installer les librairies Python
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copier tout le contenu de ton projet dans le conteneur
COPY . .

# 7. Créer les dossiers de sortie s'ils manquent
RUN mkdir -p data/eda weights

# 8. Exposer le port que FastAPI va utiliser
EXPOSE 8000

# 9. Commande pour lancer l'API au démarrage du conteneur
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
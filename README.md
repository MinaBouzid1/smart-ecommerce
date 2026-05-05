# Smart eCommerce Intelligence

Un projet Python full-stack pour l’analyse de produits e-commerce :
- scraping multi-source (Shopify, WooCommerce, sites JavaScript dynamiques)
- pipeline de normalisation et de prétraitement
- scoring et sélection des meilleurs produits avec du machine learning
- dashboard Streamlit interactif
- chatbot LLM pour interroger les données en langage naturel

---

## 📌 Objectif

Ce projet a été conçu pour construire une intelligence e-commerce capable de :
- récolter des produits depuis plusieurs boutiques en ligne
- nettoyer et normaliser les données produit
- détecter les meilleurs produits via un score composite et un modèle ML
- offrir une interface d’exploration visuelle
- fournir un assistant conversationnel pour analyser le dataset

---

## 🧩 Architecture

### 1. Scraping

Le scraping est orchestré depuis `main.py` via `agents/orchestrator.py`.
Les sources prises en charge sont :
- `ShopifyAgent` pour les shops Shopify publics
- `WooCommerceAgent` pour les boutiques WooCommerce (clé API requise)
- `PlaywrightAgent` pour les sites dynamiques JavaScript

Les résultats sont normalisés (`utils/normalizer.py`) puis sauvegardés dans :
- `data/raw/products.csv`
- `data/raw/products.db`

### 2. Prétraitement

Le pipeline effectue un nettoyage des champs, une transformation de features et
une préparation des données pour le scoring et le dashboard.

### 3. Machine Learning

Le pipeline `pipeline/smart_ecommerce_pipeline.yaml` définit des composants
Kubernetes/Kubeflow pour :
- prétraitement
- entraînement ML
- génération des meilleurs produits

Le code ML utilise notamment :
- `KMeans` pour clustering
- `XGBClassifier` / `scikit-learn` pour classer les produits top
- un score composite custom calculé à partir de prix, stock, remise, proba, etc.

### 4. Dashboard

Le dashboard Streamlit se trouve dans `dashboard/app.py`.
Il propose :
- une vue globale avec KPI
- un suivi des top produits
- un module de clustering
- des analyses de prix
- une page de données et de journal

### 5. Chatbot LLM

La brique LLM est dans `llm/` avec un chatbot capable de répondre en langage naturel
en s’appuyant sur les données du dataset.
Le composant principal est `llm/chatbot.py`.

---

## 🚀 Installation

### Pré-requis

- Python 3.10+
- Git

### Créer un environnement virtuel

```bash
python -m venv venv
venv\Scripts\activate
```

### Installer les dépendances générales

```bash
pip install -r requirements.txt
```

### Installer les dépendances dashboard

```bash
pip install -r requirements_dashboard.txt
```

### Installer les dépendances LLM (optionnel)

```bash
pip install -r requirements_llm.txt
```

> Si vous souhaitez tout installer en une fois :
> `pip install -r requirements.txt -r requirements_dashboard.txt -r requirements_llm.txt`

---

## ▶️ Exécution

### 1. Lancer le scraping

```bash
python main.py
```

Le script `main.py` utilise un exemple de configuration pour Shopify et WooCommerce.
Il génère un DataFrame normalisé et le sauvegarde dans `data/raw`.

### 2. Lancer le dashboard

```bash
streamlit run dashboard/app.py
```

Le dashboard charge automatiquement les fichiers suivants en priorité :
- `data/processed/products_llm_enriched.csv`
- `data/processed/products_enriched.csv`
- `data/raw/products.csv`

### 3. Utiliser le chatbot

Le chatbot est intégré dans le dashboard et peut aussi être utilisé via les fichiers
Python dans `llm/`.

---

## 📁 Structure des dossiers

- `agents/` : agents de scraping et orchestration
- `dashboard/` : application Streamlit et composants visuels
- `data/` : données raw et processed
- `docker/` : Dockerfiles pour les composants du projet
- `llm/` : configuration et implémentation du chatbot LLM
- `ml/` : composants ML, clustering, scoring, classification
- `pipeline/` : définition de pipeline Kubeflow
- `utils/` : normalisation, stockage, helpers

---

## 📝 Fichiers importants

- `main.py` : point d’entrée principal pour lancer le scraping
- `agents/orchestrator.py` : orchestration multi-agents
- `dashboard/app.py` : interface Streamlit
- `pipeline/smart_ecommerce_pipeline.yaml` : définition du pipeline ML
- `requirements.txt` : dépendances scraping / data
- `requirements_dashboard.txt` : dépendances dashboard
- `requirements_llm.txt` : dépendances LLM

---


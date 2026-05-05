# ◈ Smart eCommerce Intelligence

> **ML & DM Pipelines · A2A Agents · LLMs · MLOps · Business Intelligence**

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Kubeflow](https://img.shields.io/badge/Kubeflow-2.7-0097A7?style=flat-square&logo=kubeflow&logoColor=white)](https://kubeflow.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-10b981?style=flat-square)](LICENSE)
[![FST Tanger](https://img.shields.io/badge/FST-Tanger%20LSI2-1a2e4a?style=flat-square)](https://www.fstt.ac.ma)

---

## 📌 Vue d'ensemble

**Smart eCommerce Intelligence** est un système complet d'analyse e-commerce qui collecte automatiquement des données produits depuis des boutiques en ligne, les analyse avec des algorithmes de Machine Learning, et génère des insights business via des LLMs — le tout orchestré dans un pipeline MLOps industriel et visualisé dans un dashboard BI premium.

```
Shopify API ──┐
              ├──► Agents A2A ──► ML Pipeline ──► LLM Enrichissement ──► Dashboard BI
Playwright ───┘      (A2A)        (KMeans +         (Groq LLaMA)          (Streamlit)
                                   XGBoost)
                                      │
                                  Kubeflow + Docker + GitHub Actions (CI/CD)
                                      │
                                  MCP Architecture (Anthropic)
```

---

## ✨ Fonctionnalités

| Module | Description | Technologie |
|--------|-------------|-------------|
| 🕷️ **Scraping A2A** | Collecte distribuée multi-sources en parallèle | Scrapy · Playwright · Requests |
| 🧠 **ML Pipeline** | Clustering · Classification · Règles d'association · Top-K | Scikit-learn · XGBoost · mlxtend |
| ⚙️ **MLOps** | Pipeline reproductible · CI/CD automatisé | Kubeflow · Docker · GitHub Actions |
| 📊 **Dashboard BI** | Interface premium 7 pages avec chatbot intégré | Streamlit · Plotly |
| 🤖 **LLM Intelligence** | Résumés · Rapports · Recommandations marketing | Groq · LangChain · LLaMA 3.1 |
| 🔒 **Architecture MCP** | Agents responsables · Permissions · Audit log | Anthropic MCP 2025-03-26 |

---

## 🏗️ Architecture du Projet

```
smart-ecommerce/
│
├── 📁 agents/                    # Agents A2A de scraping
│   ├── base_agent.py             # Classe mère commune (retry, headers)
│   ├── shopify_agent.py          # API publique Shopify /products.json
│   ├── woocommerce_agent.py      # REST API WooCommerce v3
│   ├── playwright_agent.py       # Sites JavaScript dynamiques
│   └── orchestrator.py           # Exécution parallèle des agents
│
├── 📁 ml/                        # Pipeline Machine Learning
│   ├── preprocessing.py          # Nettoyage + Feature Engineering
│   ├── clustering.py             # KMeans · DBSCAN · PCA
│   ├── classification.py         # RandomForest · XGBoost + évaluation
│   ├── association_rules.py      # Algorithme Apriori (mlxtend)
│   ├── scoring.py                # Score composite + sélection Top-K
│   └── pipeline.py               # Orchestrateur ML complet
│
├── 📁 llm/                       # Intelligence LLM
│   ├── config.py                 # Configuration Groq API
│   ├── prompts.py                # Bibliothèque de prompts LangChain
│   ├── summarizer.py             # Résumés automatiques produits
│   ├── analyst.py                # Rapport de tendances + profils clients
│   ├── marketing.py              # Recommandations marketing par produit
│   ├── chatbot.py                # Chatbot Q&A avec mémoire conversationnelle
│   └── pipeline_llm.py           # Orchestrateur LLM complet
│
├── 📁 mcp/                       # Architecture MCP (Anthropic)
│   ├── permissions.py            # Système de permissions par manifeste
│   ├── audit_log.py              # Journal JSONL de toutes les actions
│   ├── client.py                 # MCP Client — interface unifiée
│   ├── host.py                   # MCP Host — démonstration complète
│   └── servers/
│       ├── scraping_server.py    # MCP Server Scraping
│       ├── ml_server.py          # MCP Server ML
│       └── llm_server.py         # MCP Server LLM
│
├── 📁 dashboard/                 # Dashboard Streamlit Premium
│   ├── app.py                    # Application principale (fichier unique)
│   └── assets/style.css          # Design system Dark Intelligence
│
├── 📁 pipeline/                  # Kubeflow MLOps
│   ├── pipeline.py               # Définition pipeline (kfp SDK)
│   └── smart_ecommerce_pipeline.yaml
│
├── 📁 docker/                    # Conteneurisation
│   ├── scraper/Dockerfile
│   ├── ml/Dockerfile
│   └── dashboard/Dockerfile
│
├── 📁 tests/                     # Tests unitaires pytest
│   ├── test_preprocessing.py
│   ├── test_scraping.py
│   └── test_ml.py
│
├── 📁 data/
│   ├── raw/products.csv          # Dataset brut scraping
│   ├── processed/                # Données enrichies ML + LLM
│   └── logs/mcp_audit.jsonl      # Journal d'audit MCP
│
├── .github/workflows/ci_cd.yml   # GitHub Actions CI/CD
├── docker-compose.yml            # Test local sans Kubernetes
├── mcp_config.yaml               # Configuration déclarative MCP
├── main.py                       # Point d'entrée principal
├── requirements.txt              # Dépendances scraping
├── requirements_llm.txt          # Dépendances LLM
└── requirements_dashboard.txt    # Dépendances dashboard
```

---

## 🚀 Installation et Lancement

### Prérequis

- Python 3.10+
- Git
- Docker Desktop (optionnel, pour le déploiement)
- Clé API Groq gratuite → [console.groq.com](https://console.groq.com/keys)

### 1. Cloner le dépôt

```bash
git clone https://github.com/ton-username/smart-ecommerce.git
cd smart-ecommerce
```

### 2. Environnement virtuel

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / Mac
source venv/bin/activate
```

### 3. Installer les dépendances

```bash
# Dépendances principales
pip install -r requirements.txt

# Dépendances ML
pip install scikit-learn==1.4.0 xgboost==2.0.3 numpy==1.26.0 mlxtend==0.23.1 matplotlib==3.8.0

# Dépendances Dashboard
pip install -r requirements_dashboard.txt

# Dépendances LLM
pip install -r requirements_llm.txt

# Navigateur Playwright (pour les sites JS)
pip install playwright==1.42.0
playwright install chromium

# Kubeflow SDK
pip install kfp==2.7.0
```

### 4. Configuration `.env`

Crée un fichier `.env` à la racine du projet :

```env
GROQ_API_KEY=gsk_ta_cle_ici
LLM_MODEL=llama-3.1-8b-instant
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=1000
```

> 🔑 Obtiens une clé Groq **gratuite** sur [console.groq.com/keys](https://console.groq.com/keys)

---

## ▶️ Exécution du Pipeline Complet

 les commandes **dans cet ordre** :

```bash
# ── ÉTAPE 1 : Scraping ────────────────────────────────────────────
python main.py
# → Collecte ~3200 produits depuis 8 boutiques Shopify + sites JS
# → Sauvegarde : data/raw/products.csv

# ── ÉTAPE 2 : Pipeline ML ─────────────────────────────────────────
python -m ml.pipeline
# → KMeans (4 clusters) · XGBoost · Règles Apriori · Score Top-K
# → Sauvegarde : data/processed/products_enriched.csv

# ── ÉTAPE 3 : Pipeline LLM ────────────────────────────────────────
python -m llm.pipeline_llm
# → Résumés produits · Rapport tendances · Recommandations marketing
# → Sauvegarde : data/processed/products_llm_enriched.csv

# ── ÉTAPE 4 : Démonstration MCP ──────────────────────────────────
python -m mcp.host
# → Démontre l'architecture MCP avec permissions et audit log
# → Sauvegarde : data/logs/mcp_audit.jsonl

# ── ÉTAPE 5 : Compiler le pipeline Kubeflow ──────────────────────
python pipeline/pipeline.py
# → Génère : pipeline/smart_ecommerce_pipeline.yaml

# ── ÉTAPE 6 : Lancer le Dashboard ────────────────────────────────
streamlit run dashboard/app.py
# → Ouvre : http://localhost:8501
```

### Lancement avec Docker Compose

```bash
docker-compose up --build
# Dashboard disponible sur http://localhost:8501
```

---

## 📊 Dashboard Business Intelligence

Le dashboard est accessible sur **http://localhost:8501** après lancement.

### Pages disponibles

| Page | Description |
|------|-------------|
| **Vue Globale** | KPIs dynamiques · Top-15 produits · Segments ML · Treemap catégories |
| **Top Produits** | Classement filtrable · Export CSV · Graphique scores |
| **Clustering** | Segments KMeans · Scatter plots · Stats par segment |
| **Analyse Prix** | Distributions · Box plots remises · Rating vs Prix |
| **Chatbot IA** | Q&A en langage naturel sur le dataset via Groq LLaMA |
| **Journal MCP** | Audit log temps réel · Statut serveurs · Permissions |
| **Données** | Exploration libre · Recherche · Export personnalisé |

---

## 🧠 Machine Learning — Algorithmes Implémentés

### Clustering Non Supervisé

```python
# KMeans — Segmentation des produits
km = KMeans(n_clusters=4, random_state=42, n_init=10)
# → Silhouette Score obtenu : ~0.796 (excellent)

# DBSCAN — Détection d'anomalies
db = DBSCAN(eps=0.15, min_samples=5)
# → Identifie les produits atypiques (outliers prix)

# PCA — Visualisation 2D
pca = PCA(n_components=2)
# → Projection des clusters dans l'espace bidimensionnel
```

### Classification Supervisée

```python
# RandomForest avec gestion du déséquilibre
rf = RandomForestClassifier(
    n_estimators=200, max_depth=6,
    class_weight="balanced", random_state=42
)

# XGBoost avec scale_pos_weight
xgb = XGBClassifier(
    n_estimators=200, learning_rate=0.05,
    scale_pos_weight=n0/n1  # gère le déséquilibre
)
```

### Validation

- **Séparation train/test** : 80% / 20% stratifiée
- **Validation croisée** : StratifiedKFold 5-fold
- **Métriques** : Accuracy · Precision · Recall · F1-Score · Matrice de confusion

### Score Composite Top-K

```
Score = Rating (30%) + Popularité (25%) + Proba ML (25%) + QualitéPrix (15%) + Stock (5%)
```

---

## 🤖 LLM & Prompt Engineering

Quatre modules LLM orchestrés par LangChain :

| Module | Fonction | Tokens ~|
|--------|----------|---------|
| **Summarizer** | Résumés commerciaux 2-3 phrases | ~150/produit |
| **Analyst** | Rapport marché structuré Markdown | ~2000 |
| **Marketing** | Stratégie complète par produit | ~600/produit |
| **Chatbot** | Q&A conversationnel avec mémoire | ~800/échange |

**Modèle utilisé** : `llama-3.1-8b-instant` via **Groq** (gratuit, ultra-rapide)

---

## 🔒 Architecture MCP (Model Context Protocol)

Implémentation conforme à la [spécification Anthropic MCP 2025-03-26](https://modelcontextprotocol.io/specification/2025-03-26).

### Principes implémentés

- **Moindre privilège** — chaque serveur déclare uniquement les permissions nécessaires
- **Déclaration d'intention** — chaque action est journalisée avant exécution
- **Isolation** — les serveurs ne communiquent pas directement entre eux
- **Traçabilité** — log JSONL complet (qui · quoi · quand · résultat · durée)
- **Approbation manuelle** — `approved=True` requis explicitement

### Serveurs MCP

```python
# Chaque serveur déclare son manifeste
MANIFEST = ServerManifest(
    server_id   = "scraping-server",
    permissions = [Permission.READ_DATA, Permission.CALL_EXTERNAL, Permission.WRITE_DATA],
    tools       = ["scrape_shopify", "get_raw_data", "get_scraping_status"],
    approved    = True,
)
```

---

## ⚙️ MLOps — Kubeflow + CI/CD

### Pipeline Kubeflow

```
Scraping A2A ──► Preprocessing ML ──► ML Training + Scoring ──► Top-K Export
   (500m CPU)       (500m CPU)            (1 CPU · 1Gi RAM)
   (512Mi RAM)      (512Mi RAM)
```

### GitHub Actions CI/CD

```yaml
# Déclenché sur chaque push vers main
Jobs:
  1. test   → pytest tests/ --cov=ml --cov-fail-under=70
  2. build  → docker build + push (scraper · ml · dashboard) en parallèle
  3. deploy → compilation pipeline Kubeflow YAML
```

---

## 🧪 Tests

```bash
# Lancer tous les tests
pytest tests/ -v

# Avec couverture de code
pytest tests/ -v --cov=ml --cov=agents --cov=utils --cov-report=term-missing

# Test spécifique
pytest tests/test_preprocessing.py -v
```

**Tests implémentés** :
- `test_load_returns_dataframes` — vérifie le chargement des données
- `test_features_created` — vérifie la création de toutes les features ML
- `test_no_nan_in_numeric` — assure l'absence de valeurs manquantes
- `test_normalized_columns_range` — valide la normalisation [0, 1]
- `test_no_duplicate_uid` — vérifie l'unicité des identifiants

---

## 📦 Sources de Données

### Boutiques Shopify (API publique gratuite)

| Boutique | Catégorie | URL |
|----------|-----------|-----|
| Allbirds | Chaussures écologiques | allbirds.com |
| Gymshark | Sport & Fitness | gymshark.com |
| Kith | Streetwear premium | kith.com |
| Taylor Stitch | Mode durable | taylorstitch.com |
| Tentree | Éco-responsable | tentree.com |
| MVMT Watches | Montres & accessoires | mvmtwatches.com |
| Chubbies | Mode casual | chubbiesshorts.com |
| Outer Known | Mode durable | outerknown.com |

### Sites JavaScript (Playwright)

Sites dont le contenu est chargé dynamiquement par JavaScript — nécessitent un navigateur headless.

---

## 📈 Résultats Obtenus

```
Produits collectés    : 3 200+
Boutiques scrapées    : 8 Shopify + 3 Playwright
Features ML créées    : 33 colonnes
Silhouette Score      : ~0.796 (KMeans K=4)
Accuracy XGBoost      : ~0.82-0.90
Top-K sélectionnés    : 20 meilleurs produits
Résumés LLM générés   : 20 descriptions enrichies
Recommandations mkt   : 10 fiches stratégiques
```

---

## 🛠️ Stack Technique Complète

```
Scraping      Requests · BeautifulSoup4 · Scrapy · Playwright
ML/DM         Scikit-learn · XGBoost · mlxtend · NumPy · Pandas
MLOps         Kubeflow Pipelines · Docker · GitHub Actions
Dashboard     Streamlit · Plotly · CSS custom
LLM           Groq API · LangChain · llama-3.1-8b-instant
MCP           Anthropic MCP 2025-03-26 (implémentation custom)
Tests         pytest · pytest-cov
Stockage      CSV · SQLite · JSONL
```

---

## 📋 Configuration Requise

| Composant | Version |
|-----------|---------|
| Python | 3.10+ |
| Scikit-learn | 1.4.0 |
| XGBoost | 2.0.3 |
| Streamlit | 1.32.0 |
| Plotly | 5.20.0 |
| langchain-groq | 0.1.9 |
| groq | 0.9.0 |
| kfp | 2.7.0 |
| playwright | 1.42.0 |
| mlxtend | 0.23.1 |

---

## 📖 Documentation

| Ressource | Lien |
|-----------|------|
| Anthropic MCP Spec | [modelcontextprotocol.io](https://modelcontextprotocol.io/specification/2025-03-26) |
| Kubeflow Pipelines | [kubeflow.org](https://www.kubeflow.org/docs/components/pipelines/) |
| Groq API | [console.groq.com/docs](https://console.groq.com/docs) |
| LangChain | [python.langchain.com](https://python.langchain.com/) |
| Shopify API | [shopify.dev](https://shopify.dev/docs/api/storefront) |

---

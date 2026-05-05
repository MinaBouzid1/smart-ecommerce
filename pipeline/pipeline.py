"""
Pipeline Kubeflow pour Smart eCommerce Intelligence.
Exécute : Scraping → Preprocessing → ML → Dashboard

Installation :
    pip install kfp==2.7.0

Déploiement local (Minikube) :
    python pipeline/pipeline.py   # génère pipeline.yaml
    kfp run create --experiment-name ecommerce pipeline.yaml
"""

import kfp
from kfp import dsl
from kfp.dsl import Dataset, Input, Output, Metrics, Model
from typing import NamedTuple


# ══════════════════════════════════════════════════════════════════
#  COMPOSANT 1 — Scraping
# ══════════════════════════════════════════════════════════════════

@dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=[
        "requests==2.31.0",
        "beautifulsoup4==4.12.2",
        "pandas==2.2.0",
        "fake-useragent==1.4.0",
    ]
)
def scraping_component(
    shopify_urls: list,
    output_csv: Output[Dataset],
    max_pages: int = 5,
) -> NamedTuple("Outputs", [("nb_products", int)]):
    """Scrape les shops Shopify et sauvegarde les données brutes."""
    import requests
    import pandas as pd
    import time
    import logging
    from collections import namedtuple

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("ScrapingComponent")

    all_products = []

    for shop_url in shopify_urls:
        shop_url = shop_url.rstrip("/")
        logger.info(f"Scraping : {shop_url}")

        for page in range(1, max_pages + 1):
            try:
                resp = requests.get(
                    f"{shop_url}/products.json",
                    params={"limit": 250, "page": page},
                    headers={"User-Agent": "Mozilla/5.0"},
                    timeout=15
                )
                resp.raise_for_status()
                products = resp.json().get("products", [])

                if not products:
                    break

                for p in products:
                    variant = p.get("variants", [{}])[0]
                    prix = float(variant.get("price", 0) or 0)
                    prix_compare = float(variant.get("compare_at_price") or 0)
                    remise = round((1 - prix / prix_compare) * 100, 1) \
                             if prix_compare > prix > 0 else 0.0

                    all_products.append({
                        "source":        "shopify",
                        "shop_url":      shop_url,
                        "product_id":    str(p.get("id", "")),
                        "nom":           p.get("title", "").strip(),
                        "categorie":     p.get("product_type", "").strip(),
                        "marque":        p.get("vendor", "").strip(),
                        "tags":          ", ".join(p.get("tags", [])),
                        "prix":          prix,
                        "prix_compare":  prix_compare,
                        "remise_pct":    remise,
                        "disponible":    variant.get("available", False),
                        "stock":         variant.get("inventory_quantity", 0) or 0,
                        "nb_images":     len(p.get("images", [])),
                        "nb_variants":   len(p.get("variants", [])),
                        "rating":        0.0,
                        "nb_reviews":    0,
                        "date_creation": p.get("created_at", ""),
                    })

                time.sleep(1)

            except Exception as e:
                logger.warning(f"Erreur page {page} sur {shop_url}: {e}")
                break

    df = pd.DataFrame(all_products)

    # Déduplication
    if not df.empty:
        df = df.drop_duplicates(subset=["shop_url", "product_id"])
        df["uid"] = df["source"] + "_" + df["product_id"].astype(str)

    df.to_csv(output_csv.path, index=False)
    logger.info(f"Sauvegardé : {len(df)} produits → {output_csv.path}")

    Outputs = namedtuple("Outputs", ["nb_products"])
    return Outputs(nb_products=len(df))


# ══════════════════════════════════════════════════════════════════
#  COMPOSANT 2 — Preprocessing
# ══════════════════════════════════════════════════════════════════

@dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=[
        "pandas==2.2.0",
        "scikit-learn==1.4.0",
        "numpy==1.26.0",
    ]
)
def preprocessing_component(
    input_csv:  Input[Dataset],
    output_csv: Output[Dataset],
) -> NamedTuple("Outputs", [("nb_features", int)]):
    """Nettoie, encode et crée les features ML."""
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler, LabelEncoder
    from sklearn.impute import SimpleImputer
    from collections import namedtuple

    df = pd.read_csv(input_csv.path)
    print(f"[Preprocessing] {len(df)} produits en entrée")

    # Features métier
    df["ratio_qualite_prix"] = np.where(
        df["prix"] > 0,
        df["rating"] / np.log1p(df["prix"]),
        0
    )
    df["score_popularite"] = (
        df["rating"] * 0.4 +
        np.log1p(df["nb_reviews"]) * 0.4 +
        df["disponible"].astype(int) * 0.2
    )
    df["en_promo"] = (df["remise_pct"] > 0).astype(int)

    # Encodage
    le = LabelEncoder()
    df["categorie_enc"] = le.fit_transform(df["categorie"].fillna("inconnu"))
    df["source_enc"]    = le.fit_transform(df["source"].fillna("inconnu"))

    # Imputation
    numeric = ["prix", "remise_pct", "rating", "nb_reviews",
               "stock", "ratio_qualite_prix", "score_popularite"]
    imp = SimpleImputer(strategy="median")
    df[numeric] = imp.fit_transform(df[numeric])

    # Normalisation
    scaler  = MinMaxScaler()
    to_norm = ["prix", "rating", "nb_reviews", "stock",
               "remise_pct", "ratio_qualite_prix"]
    scaled  = scaler.fit_transform(df[to_norm])
    for i, col in enumerate(to_norm):
        df[f"{col}_norm"] = scaled[:, i]

    df.to_csv(output_csv.path, index=False)
    print(f"[Preprocessing] {len(df.columns)} features → {output_csv.path}")

    Outputs = namedtuple("Outputs", ["nb_features"])
    return Outputs(nb_features=len(df.columns))


# ══════════════════════════════════════════════════════════════════
#  COMPOSANT 3 — ML Training + Scoring
# ══════════════════════════════════════════════════════════════════

@dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=[
        "pandas==2.2.0",
        "scikit-learn==1.4.0",
        "xgboost==2.0.3",
        "numpy==1.26.0",
        "mlxtend==0.23.1",
    ]
)
def ml_component(
    input_csv:   Input[Dataset],
    output_csv:  Output[Dataset],
    top_k_csv:   Output[Dataset],
    metrics_out: Output[Metrics],
    k: int = 20,
) -> NamedTuple("Outputs", [("accuracy", float), ("f1_score", float)]):
    """KMeans + XGBoost + scoring composite → Top-K produits."""
    import pandas as pd
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score
    from xgboost import XGBClassifier
    from collections import namedtuple

    df = pd.read_csv(input_csv.path)
    print(f"[ML] {len(df)} produits en entrée")

    CLUSTER_FEATURES = [
        "prix_norm", "rating_norm", "nb_reviews_norm",
        "stock_norm", "remise_pct_norm", "ratio_qualite_prix_norm"
    ]
    CLASS_FEATURES = [
        "prix", "remise_pct", "stock", "nb_images",
        "nb_variants", "en_promo", "categorie_enc", "source_enc"
    ]

    # Colonnes manquantes → 0
    for col in CLUSTER_FEATURES + CLASS_FEATURES:
        if col not in df.columns:
            df[col] = 0

    # ── KMeans ────────────────────────────────────────────────────
    X_cl = df[CLUSTER_FEATURES].fillna(0).values
    km   = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["cluster"] = km.fit_predict(X_cl)

    # ── Cible par quantile ────────────────────────────────────────
    score = pd.Series(np.zeros(len(df)), index=df.index)
    if df["prix"].sum() > 0:
        px = df["prix"].clip(lower=0)
        score += (1 - (px - px.min()) / (px.max() - px.min() + 1e-9)) * 40
    if df["remise_pct"].sum() > 0:
        r = df["remise_pct"].fillna(0)
        score += (r / (r.max() + 1e-9)) * 30
    if df["stock"].sum() > 0:
        st = df["stock"].fillna(0)
        score += (st / (st.max() + 1e-9)) * 30
    if score.std() < 1e-6:
        np.random.seed(42)
        score += np.random.rand(len(score)) * 10

    df["top_produit"] = (score >= score.quantile(0.70)).astype(int)

    # ── XGBoost ───────────────────────────────────────────────────
    X = df[CLASS_FEATURES].fillna(0).astype(float)
    y = df["top_produit"]
    acc, f1 = 0.0, 0.0

    if y.nunique() >= 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        n0, n1 = (y == 0).sum(), (y == 1).sum()
        xgb = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            scale_pos_weight=n0 / (n1 + 1e-9),
            random_state=42, eval_metric="logloss", verbosity=0
        )
        xgb.fit(X_train, y_train,
                eval_set=[(X_test, y_test)], verbose=False)

        y_pred = xgb.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        f1  = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

        df["proba_top"] = xgb.predict_proba(X)[:, 1]
        print(f"[ML] XGBoost → Accuracy={acc:.4f} | F1={f1:.4f}")
    else:
        df["proba_top"] = 0.5

    # ── Score composite ───────────────────────────────────────────
    def safe_norm(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-9)

    df["score_composite"] = (
        safe_norm(df["proba_top"])    * 40 +
        safe_norm(1 / (df["prix"] + 1)) * 30 +
        safe_norm(df["remise_pct"].fillna(0)) * 20 +
        safe_norm(df["stock"].fillna(0)) * 10
    ).round(2)

    df["rang"] = df["score_composite"].rank(ascending=False, method="min").astype(int)

    # ── Kubeflow Metrics ──────────────────────────────────────────
    metrics_out.log_metric("accuracy", acc)
    metrics_out.log_metric("f1_score", f1)
    metrics_out.log_metric("nb_products", len(df))
    metrics_out.log_metric("nb_top_products", int(df["top_produit"].sum()))

    # ── Sauvegarde ────────────────────────────────────────────────
    df.to_csv(output_csv.path, index=False)

    top_k = df.nlargest(k, "score_composite")
    top_k.to_csv(top_k_csv.path, index=False)

    print(f"[ML] Enrichi sauvegardé → {output_csv.path}")
    print(f"[ML] Top-{k} sauvegardé → {top_k_csv.path}")

    Outputs = namedtuple("Outputs", ["accuracy", "f1_score"])
    return Outputs(accuracy=acc, f1_score=f1)


# ══════════════════════════════════════════════════════════════════
#  PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════

@dsl.pipeline(
    name="smart-ecommerce-pipeline",
    description="Scraping → Preprocessing → ML → Top-K products"
)
def smart_ecommerce_pipeline(
    shopify_urls: list = [
        "https://allbirds.com",
        "https://gymshark.com",
    ],
    max_pages: int = 5,
    k: int = 20,
):
    # Step 1 — Scraping
    scraping_task = scraping_component(
        shopify_urls=shopify_urls,
        max_pages=max_pages,
    )
    scraping_task.set_display_name("Scraping A2A")
    scraping_task.set_cpu_limit("500m")
    scraping_task.set_memory_limit("512Mi")

    # Step 2 — Preprocessing (attend le scraping)
    preprocess_task = preprocessing_component(
        input_csv=scraping_task.outputs["output_csv"],
    )
    preprocess_task.set_display_name("Preprocessing ML")
    preprocess_task.after(scraping_task)

    # Step 3 — ML Training (attend le preprocessing)
    ml_task = ml_component(
        input_csv=preprocess_task.outputs["output_csv"],
        k=k,
    )
    ml_task.set_display_name("ML Training + Scoring")
    ml_task.set_cpu_limit("1000m")
    ml_task.set_memory_limit("1Gi")
    ml_task.after(preprocess_task)


# ── Génération du fichier YAML ─────────────────────────────────────
if __name__ == "__main__":
    from kfp import compiler
    compiler.Compiler().compile(
        pipeline_func=smart_ecommerce_pipeline,
        package_path="pipeline/smart_ecommerce_pipeline.yaml"
    )
    print("Pipeline compilé : pipeline/smart_ecommerce_pipeline.yaml")
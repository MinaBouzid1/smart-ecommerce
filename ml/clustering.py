import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

CLUSTER_FEATURES = [
    "prix_norm", "rating_norm", "nb_reviews_norm",
    "stock_norm", "remise_pct_norm", "ratio_qualite_prix_norm"
]

def run_kmeans(df_ml: pd.DataFrame, k: int = 4) -> pd.DataFrame:
    X = df_ml[CLUSTER_FEATURES].values

    K_range = range(2, min(10, len(df_ml) // 10 + 2))
    inertias    = []
    silhouettes = []

    for ki in K_range:
        km = KMeans(n_clusters=ki, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels))  # ← calculé pour TOUS les k

    # ── Plot elbow + silhouette ────────────────────────────────────
    k_list = list(K_range)   # même longueur que inertias et silhouettes

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(k_list, inertias,    "o-", color="#534AB7")
    axes[0].set_title("Méthode du coude (Elbow)")
    axes[0].set_xlabel("Nombre de clusters K")
    axes[0].set_ylabel("Inertie")

    axes[1].plot(k_list, silhouettes, "o-", color="#1D9E75")  # ← même liste k_list
    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("Nombre de clusters K")
    axes[1].set_ylabel("Score")

    plt.tight_layout()
    plt.savefig("data/processed/elbow_silhouette.png", dpi=120)
    plt.close()
    print("[KMeans] Graphique elbow/silhouette sauvegardé")

    # ── KMeans final ───────────────────────────────────────────────
    km_final = KMeans(n_clusters=k, random_state=42, n_init=10)
    df_ml["cluster_kmeans"] = km_final.fit_predict(X)

    sil = silhouette_score(X, df_ml["cluster_kmeans"])
    print(f"[KMeans] K={k} | Silhouette Score = {sil:.3f}")

    cluster_summary = df_ml.groupby("cluster_kmeans")[
        ["prix", "rating", "nb_reviews", "remise_pct"]
    ].mean().round(2)
    print("\n[KMeans] Profil des clusters :")
    print(cluster_summary.to_string())

    labels_map = _name_clusters(cluster_summary)
    df_ml["segment"] = df_ml["cluster_kmeans"].map(labels_map)

    return df_ml

def _name_clusters(summary: pd.DataFrame) -> dict:
    labels = {}
    for idx, row in summary.iterrows():
        if row["rating"] >= 4.5 and row["prix"] > 100:
            labels[idx] = "Premium populaire"
        elif row["rating"] >= 4.0 and row["nb_reviews"] > 500:
            labels[idx] = "Bestseller"
        elif row["remise_pct"] > 20:
            labels[idx] = "Produit en promo"
        elif row["prix"] < 30:
            labels[idx] = "Entrée de gamme"
        else:
            labels[idx] = f"Segment {idx}"
    return labels

def run_dbscan(df_ml: pd.DataFrame, eps: float = 0.15, min_samples: int = 5) -> pd.DataFrame:
    X = df_ml[CLUSTER_FEATURES].values
    db = DBSCAN(eps=eps, min_samples=min_samples)
    df_ml["cluster_dbscan"] = db.fit_predict(X)

    n_outliers = (df_ml["cluster_dbscan"] == -1).sum()
    n_clusters = len(set(df_ml["cluster_dbscan"])) - (1 if -1 in df_ml["cluster_dbscan"].values else 0)

    print(f"[DBSCAN] {n_clusters} clusters | {n_outliers} anomalies détectées")
    df_ml["is_anomalie"] = (df_ml["cluster_dbscan"] == -1).astype(int)
    return df_ml

def run_pca_visualization(df_ml: pd.DataFrame, output_path: str = "data/processed/pca_plot.png"):
    X = df_ml[CLUSTER_FEATURES].values
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)

    df_viz = pd.DataFrame(coords, columns=["PC1", "PC2"])
    df_viz["segment"] = df_ml["segment"].values if "segment" in df_ml.columns else "N/A"

    colors = ["#534AB7", "#1D9E75", "#D85A30", "#BA7517", "#D4537E"]
    fig, ax = plt.subplots(figsize=(10, 7))
    for i, seg in enumerate(df_viz["segment"].unique()):
        mask = df_viz["segment"] == seg
        ax.scatter(
            df_viz.loc[mask, "PC1"],
            df_viz.loc[mask, "PC2"],
            label=seg, alpha=0.7, s=40,
            color=colors[i % len(colors)]
        )
    variance = pca.explained_variance_ratio_.sum() * 100
    ax.set_title(f"PCA — Visualisation des segments produits\nVariance expliquée : {variance:.1f}%")
    ax.set_xlabel("Composante 1")
    ax.set_ylabel("Composante 2")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f"[PCA] Graphique sauvegardé : {output_path}")
    return df_viz
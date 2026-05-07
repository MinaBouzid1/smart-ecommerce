import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from ml.preprocessing import load_and_prepare
from ml.clustering import run_kmeans, run_dbscan, run_pca_visualization, _name_clusters
from ml.classification import create_target, train_and_evaluate, _evaluate_model
from ml.scoring import compute_composite_score, get_top_k
from ml.association_rules import run_association_rules, _limit_vocabulary


# ══════════════════════════════════════════════════════════
#  FIXTURE — Dataset de test
# ══════════════════════════════════════════════════════════

@pytest.fixture
def sample_df():
    """DataFrame réaliste pour tous les tests ML."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "nom": [f"Product {i}" for i in range(n)],
        "prix": np.random.uniform(10, 500, n).round(2),
        "rating": np.random.uniform(1, 5, n).round(1),
        "nb_reviews": np.random.randint(0, 5000, n),
        "remise_pct": np.random.uniform(0, 50, n).round(1),
        "stock": np.random.randint(0, 200, n),
        "disponible": np.random.choice([True, False], n),
        "categorie": np.random.choice(["A", "B", "C", "D"], n),
        "source": np.random.choice(["shopify", "woocommerce"], n),
        "nb_images": np.random.randint(1, 5, n),
        "nb_variants": np.random.randint(1, 4, n),
        "shop_url": np.random.choice(["shop1.com", "shop2.com"], n),
        "marque": np.random.choice(["Nike", "Adidas", "Puma"], n),
    })


@pytest.fixture
def sample_df_ml(sample_df):
    """DataFrame prétraité prêt pour ML."""
    _, df_ml = load_and_prepare(sample_df)
    return df_ml


# ══════════════════════════════════════════════════════════
#  TESTS — Preprocessing
# ══════════════════════════════════════════════════════════

def test_load_and_prepare_returns_tuple(sample_df, tmp_path):
    csv_path = tmp_path / "test.csv"
    sample_df.to_csv(csv_path, index=False)
    df_raw, df_ml = load_and_prepare(str(csv_path))
    assert isinstance(df_raw, pd.DataFrame)
    assert isinstance(df_ml, pd.DataFrame)
    assert len(df_ml) == len(sample_df)


def test_preprocessing_creates_features(sample_df_ml):
    assert "ratio_qualite_prix" in sample_df_ml.columns
    assert "score_popularite" in sample_df_ml.columns
    assert "en_promo" in sample_df_ml.columns
    assert "segment_prix" in sample_df_ml.columns
    assert "categorie_enc" in sample_df_ml.columns


def test_preprocessing_normalization(sample_df_ml):
    norm_cols = [c for c in sample_df_ml.columns if c.endswith("_norm")]
    assert len(norm_cols) > 0
    for col in norm_cols:
        assert sample_df_ml[col].min() >= 0
        assert sample_df_ml[col].max() <= 1


# ══════════════════════════════════════════════════════════
#  TESTS — Clustering
# ══════════════════════════════════════════════════════════

def test_run_kmeans(sample_df_ml):
    result = run_kmeans(sample_df_ml.copy(), k=4)
    assert "cluster_kmeans" in result.columns
    assert "segment" in result.columns
    assert result["cluster_kmeans"].nunique() <= 4


def test_run_dbscan(sample_df_ml):
    result = run_dbscan(sample_df_ml.copy(), eps=0.3, min_samples=5)
    assert "cluster_dbscan" in result.columns
    assert "is_anomalie" in result.columns
    assert result["is_anomalie"].isin([0, 1]).all()


def test_run_pca_visualization(sample_df_ml, tmp_path):
    output = tmp_path / "pca.png"
    result = run_pca_visualization(sample_df_ml.copy(), str(output))
    assert isinstance(result, pd.DataFrame)
    assert "PC1" in result.columns
    assert "PC2" in result.columns


def test_name_clusters():
    summary = pd.DataFrame({
        "rating": [4.8, 3.5, 4.2, 2.0],
        "prix": [150, 80, 120, 20],
        "nb_reviews": [1000, 100, 600, 50],
        "remise_pct": [0, 30, 0, 0]
    })
    labels = _name_clusters(summary)
    assert isinstance(labels, dict)
    assert len(labels) == 4


# ══════════════════════════════════════════════════════════
#  TESTS — Classification
# ══════════════════════════════════════════════════════════

def test_create_target(sample_df_ml):
    result = create_target(sample_df_ml.copy())
    assert "top_produit" in result.columns
    assert result["top_produit"].isin([0, 1]).all()
    assert result["top_produit"].sum() > 0  # au moins quelques tops


def test_train_and_evaluate(sample_df_ml):
    results, df_out, xgb, rf = train_and_evaluate(sample_df_ml.copy())
    assert isinstance(results, dict)
    assert "random_forest" in results
    assert "xgboost" in results
    assert 0 <= results["xgboost"]["accuracy"] <= 1
    assert "proba_top_xgb" in df_out.columns


# ══════════════════════════════════════════════════════════
#  TESTS — Scoring
# ══════════════════════════════════════════════════════════

def test_compute_composite_score(sample_df_ml):
    # Ajoute proba_consensus si manquant
    if "proba_consensus" not in sample_df_ml.columns:
        sample_df_ml["proba_consensus"] = np.random.random(len(sample_df_ml))

    result = compute_composite_score(sample_df_ml.copy())
    assert "score_composite" in result.columns
    assert "rang" in result.columns
    assert result["score_composite"].min() >= 0
    assert result["score_composite"].max() <= 100


def test_get_top_k(sample_df_ml, tmp_path):
    if "proba_consensus" not in sample_df_ml.columns:
        sample_df_ml["proba_consensus"] = np.random.random(len(sample_df_ml))
    df_scored = compute_composite_score(sample_df_ml.copy())

    top = get_top_k(df_scored, k=10)
    assert isinstance(top, pd.DataFrame)
    assert len(top) == 10
    assert top["score_composite"].is_monotonic_decreasing


# ══════════════════════════════════════════════════════════
#  TESTS — Association Rules
# ══════════════════════════════════════════════════════════

def test_run_association_rules(sample_df):
    rules = run_association_rules(sample_df.copy())
    assert isinstance(rules, pd.DataFrame)  # peut être vide mais pas d'erreur


def test_limit_vocabulary():
    transactions = [
        ["a", "b", "c"],
        ["a", "b", "d"],
        ["a", "c", "e"],
        ["f", "g", "h"]
    ]
    result = _limit_vocabulary(transactions, max_items=3)
    assert all(len(t) >= 2 for t in result)
    # Seuls les items fréquents (a, b, c) doivent rester
    all_items = {item for t in result for item in t}
    assert "f" not in all_items  # item rare supprimé

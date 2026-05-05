"""
Tests unitaires pour le module preprocessing.
Lance avec : pytest tests/ -v
"""
import pytest
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ml.preprocessing import load_and_prepare


# ── Fixture : dataset minimal de test ─────────────────────────────
@pytest.fixture
def sample_csv(tmp_path):
    """Crée un CSV de test temporaire."""
    data = pd.DataFrame({
        "source":      ["shopify"] * 50,
        "shop_url":    ["https://test.com"] * 50,
        "product_id":  [str(i) for i in range(50)],
        "nom":         [f"Produit {i}" for i in range(50)],
        "description": ["desc"] * 50,
        "categorie":   (["Electronics"] * 25 + ["Sport"] * 25),
        "marque":      ["BrandA"] * 50,
        "tags":        ["tag1, tag2"] * 50,
        "prix":        np.random.uniform(10, 500, 50),
        "prix_compare": np.random.uniform(500, 600, 50),
        "disponible":  [True, False] * 25,
        "stock":       np.random.randint(0, 100, 50),
        "date_creation": ["2024-01-01"] * 50,
        "date_update":   ["2024-06-01"] * 50,
        "nb_images":   np.random.randint(1, 5, 50),
        "nb_variants": np.random.randint(1, 3, 50),
        "remise_pct":  np.random.uniform(0, 40, 50),
        "rating":      np.random.uniform(0, 5, 50),
        "nb_reviews":  np.random.randint(0, 1000, 50),
        "uid":         [f"shopify_{i}" for i in range(50)],
    })
    path = tmp_path / "products.csv"
    data.to_csv(path, index=False)
    return str(path)


# ── Tests ──────────────────────────────────────────────────────────

def test_load_returns_dataframes(sample_csv):
    """load_and_prepare doit retourner deux DataFrames non-vides."""
    df_raw, df_ml = load_and_prepare(sample_csv)
    assert isinstance(df_raw, pd.DataFrame)
    assert isinstance(df_ml, pd.DataFrame)
    assert len(df_raw) == 50
    assert len(df_ml) == 50


def test_features_created(sample_csv):
    """Les features ML doivent être présentes après preprocessing."""
    _, df_ml = load_and_prepare(sample_csv)
    required = [
        "ratio_qualite_prix", "score_popularite", "en_promo",
        "categorie_enc", "source_enc",
        "prix_norm", "rating_norm", "nb_reviews_norm"
    ]
    for col in required:
        assert col in df_ml.columns, f"Colonne manquante : {col}"


def test_no_nan_in_numeric(sample_csv):
    """Pas de NaN dans les colonnes numériques après preprocessing."""
    _, df_ml = load_and_prepare(sample_csv)
    numeric_cols = ["prix", "rating", "nb_reviews", "stock", "remise_pct"]
    for col in numeric_cols:
        assert df_ml[col].isna().sum() == 0, f"NaN dans {col}"


def test_normalized_columns_range(sample_csv):
    """Les colonnes normalisées doivent être dans [0, 1]."""
    _, df_ml = load_and_prepare(sample_csv)
    norm_cols = ["prix_norm", "rating_norm", "nb_reviews_norm"]
    for col in norm_cols:
        assert df_ml[col].min() >= -0.01, f"{col} < 0"
        assert df_ml[col].max() <= 1.01, f"{col} > 1"


def test_no_duplicate_uid(sample_csv):
    """Pas de doublons sur l'uid."""
    df_raw, _ = load_and_prepare(sample_csv)
    assert df_raw["uid"].duplicated().sum() == 0
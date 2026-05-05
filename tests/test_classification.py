import pandas as pd
import numpy as np
import pytest
from ml.classification import train_and_evaluate


def test_cross_validation():
    """Test avec données synthétiques — compatible avec tout type de retour."""
    np.random.seed(42)
    n = 200

    df = pd.DataFrame({
        "prix": np.random.uniform(10, 500, n).round(2),
        "rating": np.random.uniform(1, 5, n).round(1),
        "nb_reviews": np.random.randint(0, 10000, n),
        "remise_pct": np.random.uniform(0, 60, n).round(1),
        "categorie": np.random.choice(["Electronics", "Fashion", "Home", "Sport"], n),
        "segment": np.random.choice(["Premium", "Standard", "Budget"], n),
        "top_produit": np.random.choice([0, 1], n, p=[0.7, 0.3])
    })

    result = train_and_evaluate(df)

    # ── Gère n'importe quel type de retour ──
    if isinstance(result, tuple):
        metrics = result[0]          # premier élément = dict métriques
        df_out = result[1] if len(result) > 1 else None
    elif isinstance(result, dict):
        metrics = result
        df_out = None
    else:
        metrics = result
        df_out = None

    # ── Assertions sur les métriques ──
    assert isinstance(metrics, dict), f"Attendu dict, reçu {type(metrics)}"
    assert "random_forest" in metrics or "xgboost" in metrics
    assert 0 <= metrics["xgboost"]["accuracy"] <= 1
    assert 0 <= metrics["random_forest"]["accuracy"] <= 1

    # ── Si un DataFrame est retourné ──
    if df_out is not None:
        assert isinstance(df_out, pd.DataFrame)
        assert len(df_out) == n


def test_empty_dataframe():
    """Test de robustesse avec un DataFrame vide."""
    df = pd.DataFrame()
    with pytest.raises(Exception):
        train_and_evaluate(df)
import pandas as pd
import numpy as np
import pytest
from ml.classification import train_and_evaluate


def test_cross_validation():
    """Test avec données synthétiques en mémoire — aucun fichier externe."""
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

    assert isinstance(result, dict)
    assert "accuracy" in result
    assert 0 <= result["accuracy"] <= 1
    assert "f1" in result or "f1_score" in result


def test_empty_dataframe():
    """Test de robustesse avec un DataFrame vide."""
    df = pd.DataFrame()
    with pytest.raises(Exception):
        train_and_evaluate(df)
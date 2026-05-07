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

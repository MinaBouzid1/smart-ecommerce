from pathlib import Path
from ml.preprocessing import load_and_prepare
from ml.clustering import run_kmeans, run_dbscan, run_pca_visualization
from ml.classification import train_and_evaluate
from ml.association_rules import run_association_rules
from ml.scoring import compute_composite_score, get_top_k

def run_ml_pipeline(csv_path: str = "data/raw/products.csv", k: int = 20):
    """
    Pipeline ML complet en 5 étapes séquentielles.
    """
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("  PIPELINE ML — Smart eCommerce Intelligence")
    print("="*60)

    # ── Étape 1 : Prétraitement ────────────────────────────────────
    print("\n[1/5] Prétraitement des données...")
    df_raw, df_ml = load_and_prepare(csv_path)

    # ── Étape 2 : Clustering ──────────────────────────────────────
    print("\n[2/5] Clustering (KMeans + DBSCAN)...")
    df_ml = run_kmeans(df_ml, k=4)
    df_ml = run_dbscan(df_ml)
    run_pca_visualization(df_ml)

    # ── Étape 3 : Classification supervisée ───────────────────────
    print("\n[3/5] Classification (RF + XGBoost)...")
    results, df_ml, xgb_model, rf_model = train_and_evaluate(df_ml)

    # ── Étape 4 : Règles d'association ────────────────────────────
    print("\n[4/5] Règles d'association...")
    rules = run_association_rules(df_raw)

    # ── Étape 5 : Scoring + Top-K ─────────────────────────────────
    print("\n[5/5] Scoring composite + Sélection Top-K...")
    df_ml = compute_composite_score(df_ml)
    top_k = get_top_k(df_ml, k=k)

    # Sauvegarde du dataset enrichi complet
    df_ml.to_csv("data/processed/products_enriched.csv", index=False)
    print("\n[Pipeline] Dataset enrichi sauvegardé : data/processed/products_enriched.csv")

    print("\n" + "="*60)
    print("  PIPELINE TERMINÉ")
    print(f"  {len(df_ml)} produits analysés | Top-{k} sélectionnés")
    print("  Graphiques dans : data/processed/")
    print("="*60)

    return df_ml, top_k, results


if __name__ == "__main__":
    df_ml, top_k, metrics = run_ml_pipeline(k=20)
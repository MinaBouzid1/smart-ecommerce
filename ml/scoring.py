import pandas as pd
import numpy as np

def compute_composite_score(df_ml: pd.DataFrame) -> pd.DataFrame:
    """
    Score composite sur 100 points combinant :
    - Rating (qualité perçue)          : 30%
    - Popularité (volume d'avis)        : 25%
    - Probabilité ML (XGBoost+RF)       : 25%
    - Rapport qualité/prix              : 15%
    - Disponibilité stock               : 5%
    """

    def safe_norm(series: pd.Series) -> pd.Series:
        """Normalise entre 0 et 1, gère les cas dégénérés."""
        mn, mx = series.min(), series.max()
        if mx == mn:
            return pd.Series(np.zeros(len(series)), index=series.index)
        return (series - mn) / (mx - mn)

    # Normalisation de chaque composante
    score_rating     = safe_norm(df_ml["rating"])
    score_reviews    = safe_norm(np.log1p(df_ml["nb_reviews"]))
    score_ml         = safe_norm(df_ml.get("proba_consensus", pd.Series(0, index=df_ml.index)))
    score_qp         = safe_norm(df_ml["ratio_qualite_prix"])
    score_dispo      = df_ml["disponible"].astype(float)

    # Score composite pondéré
    df_ml["score_composite"] = (
        score_rating  * 30 +
        score_reviews * 25 +
        score_ml      * 25 +
        score_qp      * 15 +
        score_dispo   *  5
    ).round(2)

    # Rang global
    df_ml["rang"] = df_ml["score_composite"].rank(ascending=False, method="min").astype(int)

    print(f"[Scoring] Score calculé pour {len(df_ml)} produits")
    print(f"  Moyenne : {df_ml['score_composite'].mean():.1f}/100")
    print(f"  Max     : {df_ml['score_composite'].max():.1f}/100")

    return df_ml

def get_top_k(df_ml: pd.DataFrame, k: int = 20,
              categorie: str = None,
              source: str = None) -> pd.DataFrame:
    """
    Retourne les K meilleurs produits selon le score composite.
    
    Filtres optionnels :
    - categorie : filtrer par catégorie de produit
    - source    : filtrer par plateforme (shopify / woocommerce)
    """
    df = df_ml.copy()

    if categorie:
        df = df[df["categorie"].str.lower().str.contains(categorie.lower(), na=False)]
    if source:
        df = df[df["source"] == source]

    top_k = df.nlargest(k, "score_composite")

    # Colonnes d'affichage
    display_cols = [
        "rang", "nom", "categorie", "source", "prix",
        "rating", "nb_reviews", "remise_pct",
        "disponible", "score_composite", "segment"
    ]
    display_cols = [c for c in display_cols if c in top_k.columns]

    print(f"\n{'='*60}")
    print(f"  TOP {k} PRODUITS" + (f" — {categorie}" if categorie else ""))
    print(f"{'='*60}")
    print(top_k[display_cols].to_string(index=False))

    top_k.to_csv(f"data/processed/top_{k}_produits.csv", index=False)
    print(f"\n[Top-K] Sauvegardé dans data/processed/top_{k}_produits.csv")

    return top_k
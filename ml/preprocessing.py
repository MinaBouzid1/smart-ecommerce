import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer

def load_and_prepare(csv_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Charge le CSV brut et retourne :
    - df_raw   : DataFrame lisible (pour affichage)
    - df_ml    : DataFrame prêt pour les algorithmes ML
    """
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    print(f"[Preprocessing] {len(df)} produits chargés")

    df_ml = df.copy()

    # ── 1. Création de features métier ─────────────────────────────
    # Ratio qualité/prix (plus c'est haut, mieux c'est)
    df_ml["ratio_qualite_prix"] = np.where(
        df_ml["prix"] > 0,
        df_ml["rating"] / np.log1p(df_ml["prix"]),
        0
    )

    # Score de popularité brut
    df_ml["score_popularite"] = (
        df_ml["rating"]     * 0.4 +
        np.log1p(df_ml["nb_reviews"]) * 0.4 +
        df_ml["disponible"].astype(int) * 0.2
    )

    # Indicateur promotion active
    df_ml["en_promo"] = (df_ml["remise_pct"] > 0).astype(int)

    # Catégorie de prix
    df_ml["segment_prix"] = pd.cut(
        df_ml["prix"],
        bins=[0, 20, 50, 100, 200, float("inf")],
        labels=["très_bas", "bas", "moyen", "élevé", "premium"]
    )

    # ── 2. Encodage catégoriel ─────────────────────────────────────
    le = LabelEncoder()
    df_ml["categorie_enc"] = le.fit_transform(
        df_ml["categorie"].fillna("inconnu")
    )
    df_ml["source_enc"] = le.fit_transform(
        df_ml["source"].fillna("inconnu")
    )

    # ── 3. Imputation des valeurs manquantes ───────────────────────
    numeric_features = [
        "prix", "remise_pct", "rating", "nb_reviews",
        "stock", "ratio_qualite_prix", "score_popularite"
    ]
    imputer = SimpleImputer(strategy="median")
    df_ml[numeric_features] = imputer.fit_transform(df_ml[numeric_features])

    # ── 4. Normalisation (0 à 1) pour clustering ──────────────────
    scaler = MinMaxScaler()
    features_to_scale = ["prix", "rating", "nb_reviews", "stock",
                         "remise_pct", "ratio_qualite_prix", "score_popularite"]
    scaled = scaler.fit_transform(df_ml[features_to_scale])
    df_scaled = pd.DataFrame(scaled, columns=[f"{c}_norm" for c in features_to_scale])
    df_ml = pd.concat([df_ml, df_scaled], axis=1)

    print(f"[Preprocessing] Features créées : {list(df_ml.columns)}")
    return df, df_ml
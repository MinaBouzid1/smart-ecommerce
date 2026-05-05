import pandas as pd
import re

def normalize_products(products: list[dict]) -> pd.DataFrame:
    """
    Reçoit une liste brute de produits (toutes sources confondues),
    retourne un DataFrame propre, uniforme et dédupliqué.
    """
    df = pd.DataFrame(products)

    if df.empty:
        return df

    # ── 1. Colonnes garanties ──────────────────────────────────────
    required_cols = [
        "source", "shop_url", "product_id", "nom", "description",
        "categorie", "prix", "remise_pct", "rating", "nb_reviews",
        "disponible", "stock", "date_creation"
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = None

    # ── 2. Nettoyage texte ─────────────────────────────────────────
    df["nom"] = df["nom"].astype(str).str.strip().str.title()
    df["nom"] = df["nom"].apply(lambda x: re.sub(r'\s+', ' ', x))
    df["categorie"] = df["categorie"].fillna("inconnu").str.lower().str.strip()

    # ── 3. Typage numérique ────────────────────────────────────────
    df["prix"]       = pd.to_numeric(df["prix"],       errors="coerce").fillna(0.0)
    df["remise_pct"] = pd.to_numeric(df["remise_pct"], errors="coerce").fillna(0.0)
    df["rating"]     = pd.to_numeric(df["rating"],     errors="coerce").fillna(0.0)
    df["nb_reviews"] = pd.to_numeric(df["nb_reviews"], errors="coerce").fillna(0).astype(int)
    df["stock"]      = pd.to_numeric(df["stock"],      errors="coerce").fillna(0).astype(int)

    # ── 4. Disponibilité booléenne ─────────────────────────────────
    df["disponible"] = df["disponible"].map(
        lambda x: True if str(x).lower() in ["true", "1", "yes", "oui"] else False
    )

    # ── 5. Suppression doublons ────────────────────────────────────
    df = df.drop_duplicates(subset=["shop_url", "product_id"]).reset_index(drop=True)

    # ── 6. Filtre qualité minimale ─────────────────────────────────
    df = df[df["nom"].str.len() > 2]   # pas de noms vides
    df = df[df["prix"] >= 0]            # pas de prix négatifs

    # ── 7. Colonne id unique global ────────────────────────────────
    df["uid"] = df["source"] + "_" + df["product_id"].astype(str)

    print(f"[Normalizer] {len(df)} produits propres (sources: {df['source'].unique()})")
    return df
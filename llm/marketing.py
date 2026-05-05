"""
Module de recommandations marketing automatiques par produit.
"""
import pandas as pd
import logging
from langchain_core.output_parsers import StrOutputParser

from llm.config import get_llm
from llm.prompts import MARKETING_PROMPT

logger = logging.getLogger("LLM.Marketing")


def generate_marketing_recommendations(
    df: pd.DataFrame,
    n_top: int = 10,
) -> pd.DataFrame:
    """
    Génère des recommandations marketing pour les N meilleurs produits.

    Returns:
        DataFrame avec colonne 'recommandation_marketing' ajoutée.
    """
    llm   = get_llm(temperature=0.6, max_tokens=600)
    chain = MARKETING_PROMPT | llm | StrOutputParser()

    df = df.copy()
    df["recommandation_marketing"] = ""

    score_col = "score_composite" if "score_composite" in df.columns else "prix"
    top = df.nlargest(n_top, score_col)

    logger.info(f"Génération de {n_top} recommandations marketing...")

    for i, (idx, row) in enumerate(top.iterrows()):
        try:
            reco = chain.invoke({
                "nom":            str(row.get("nom", "Produit"))[:60],
                "categorie":      str(row.get("categorie", "N/A")),
                "shop_url":       str(row.get("shop_url", "N/A")),
                "prix":           f"{float(row.get('prix', 0)):.2f}",
                "remise_pct":     f"{float(row.get('remise_pct', 0)):.1f}",
                "score_composite": f"{float(row.get('score_composite', 0)):.2f}",
                "segment":        str(row.get("segment", "N/A")),
                "stock":          str(int(row.get("stock", 0))),
                "en_promo":       "Oui" if row.get("en_promo", 0) == 1 else "Non",
            })
            df.at[idx, "recommandation_marketing"] = reco.strip()
            logger.info(f"  [{i+1}/{n_top}] {row.get('nom','')[:40]}")

        except Exception as e:
            logger.warning(f"  Erreur produit {idx}: {e}")

    return df
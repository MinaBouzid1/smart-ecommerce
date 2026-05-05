"""
Module de résumé automatique des descriptions produits.
Traite les produits par batch pour optimiser les coûts API.
"""
import pandas as pd
import time
import logging
from pathlib import Path
from langchain_core.output_parsers import StrOutputParser

from llm.config import get_llm
from llm.prompts import PRODUCT_SUMMARY_PROMPT

logger = logging.getLogger("LLM.Summarizer")


def summarize_products(
    df: pd.DataFrame,
    n_samples: int = 50,
    batch_size: int = 5,
    delay: float = 1.0,
) -> pd.DataFrame:
    """
    Résume automatiquement les descriptions des produits via LLM.

    Args:
        df         : DataFrame enrichi
        n_samples  : nombre de produits à résumer (limité pour le coût)
        batch_size : nombre de produits traités avant pause
        delay      : pause entre batches (rate limiting)

    Returns:
        DataFrame avec colonne 'description_llm' ajoutée
    """
    llm    = get_llm(temperature=0.4)
    chain  = PRODUCT_SUMMARY_PROMPT | llm | StrOutputParser()
    df     = df.copy()

    # Sélection : top produits avec description disponible
    mask = df["description"].notna() & (df["description"].str.len() > 20)
    candidates = df[mask].nlargest(n_samples, "score_composite") \
                 if "score_composite" in df.columns \
                 else df[mask].head(n_samples)

    logger.info(f"Résumé de {len(candidates)} produits...")
    df["description_llm"] = ""

    for i, (idx, row) in enumerate(candidates.iterrows()):
        try:
            summary = chain.invoke({
                "nom":         str(row.get("nom", "Produit inconnu")),
                "categorie":   str(row.get("categorie", "N/A")),
                "prix":        f"{float(row.get('prix', 0)):.2f}",
                "description": str(row.get("description", ""))[:800],
            })
            df.at[idx, "description_llm"] = summary.strip()
            logger.info(f"  [{i+1}/{len(candidates)}] {row.get('nom','')[:40]}")

        except Exception as e:
            logger.warning(f"  Erreur produit {idx}: {e}")
            df.at[idx, "description_llm"] = ""

        # Pause entre batches
        if (i + 1) % batch_size == 0:
            time.sleep(delay)

    n_done = (df["description_llm"] != "").sum()
    logger.info(f"Résumés générés : {n_done}/{len(candidates)}")
    return df
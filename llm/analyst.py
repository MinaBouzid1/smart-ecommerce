"""
Module d'analyse de tendances et génération de rapport automatique.
"""
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from langchain_core.output_parsers import StrOutputParser

from llm.config import get_llm
from llm.prompts import TREND_ANALYSIS_PROMPT, CLIENT_PROFILE_PROMPT

logger = logging.getLogger("LLM.Analyst")


def _build_context(df: pd.DataFrame) -> dict:
    """Construit le contexte statistique pour le prompt d'analyse."""

    # Top catégories
    top_cats = (
        df.groupby("categorie")["nom"]
        .count()
        .nlargest(5)
        .reset_index()
    )
    top_cats_str = "\n".join(
        f"  - {row['categorie']} : {row['nom']} produits"
        for _, row in top_cats.iterrows()
    )

    # Top produits
    score_col = "score_composite" if "score_composite" in df.columns else "prix"
    top_prods = df.nlargest(10, score_col)[["nom", "categorie", "prix", score_col]]
    top_prods_str = "\n".join(
        f"  - {row['nom'][:40]} ({row['categorie']}) — {row['prix']:.0f}€ "
        f"— score: {row[score_col]:.2f}"
        for _, row in top_prods.iterrows()
    )

    # Segments
    seg_str = "Non disponible"
    if "segment" in df.columns:
        segs = df["segment"].value_counts()
        seg_str = "\n".join(
            f"  - {seg}: {cnt} produits ({cnt/len(df)*100:.1f}%)"
            for seg, cnt in segs.items()
        )

    return {
        "nb_produits":    len(df),
        "nb_shops":       df["shop_url"].nunique() if "shop_url" in df.columns else 1,
        "nb_categories":  df["categorie"].nunique(),
        "prix_moyen":     f"{df['prix'].mean():.2f}",
        "prix_median":    f"{df['prix'].median():.2f}",
        "prix_min":       f"{df['prix'].min():.2f}",
        "prix_max":       f"{df['prix'].max():.2f}",
        "pct_promo":      f"{(df['remise_pct'] > 0).mean() * 100:.1f}",
        "remise_moyenne": f"{df[df['remise_pct'] > 0]['remise_pct'].mean():.1f}"
                          if (df['remise_pct'] > 0).any() else "0",
        "top_categories": top_cats_str,
        "top_produits":   top_prods_str,
        "segments":       seg_str,
    }


def generate_trend_report(df: pd.DataFrame) -> str:
    """
    Génère un rapport d'analyse de tendances complet via LLM.
    Retourne le rapport en texte Markdown.
    """
    llm   = get_llm(temperature=0.3, max_tokens=2000)
    chain = TREND_ANALYSIS_PROMPT | llm | StrOutputParser()

    logger.info("Génération du rapport de tendances...")

    context = _build_context(df)

    try:
        report = chain.invoke(context)
        logger.info("Rapport généré avec succès")
        return report

    except Exception as e:
        logger.error(f"Erreur génération rapport : {e}")
        return f"Erreur lors de la génération : {e}"


def save_report(report: str, output_dir: str = "data/processed") -> str:
    """Sauvegarde le rapport en Markdown et retourne le chemin."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"{output_dir}/rapport_llm_{timestamp}.md"

    header = (
        f"# Rapport d'Intelligence eCommerce\n"
        f"*Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}*\n\n"
        f"---\n\n"
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(header + report)

    logger.info(f"Rapport sauvegardé : {path}")
    return path


def generate_client_profiles(df: pd.DataFrame) -> dict[str, str]:
    """
    Génère un profil client pour chaque segment détecté.
    Retourne un dict {segment: profil_texte}.
    """
    if "segment" not in df.columns:
        logger.warning("Colonne 'segment' absente — profils non générés.")
        return {}

    llm    = get_llm(temperature=0.5)
    chain  = CLIENT_PROFILE_PROMPT | llm | StrOutputParser()
    profiles = {}

    for segment in df["segment"].unique():
        seg_df   = df[df["segment"] == segment]
        top_prods = seg_df.nlargest(5, "score_composite") \
                   if "score_composite" in seg_df.columns \
                   else seg_df.head(5)

        prods_str = "\n".join(
            f"  - {row['nom'][:40]} — {row['prix']:.0f}€"
            for _, row in top_prods.iterrows()
        )

        try:
            profile = chain.invoke({
                "segment":       segment,
                "produits":      prods_str,
                "prix_moyen":    f"{seg_df['prix'].mean():.2f}",
                "remise_moyenne": f"{seg_df['remise_pct'].mean():.1f}",
                "score_moyen":   f"{seg_df['score_composite'].mean():.2f}"
                                 if "score_composite" in seg_df.columns else "N/A",
            })
            profiles[segment] = profile
            logger.info(f"  Profil généré : {segment}")

        except Exception as e:
            logger.warning(f"  Erreur profil {segment}: {e}")
            profiles[segment] = f"Erreur : {e}"

    return profiles
"""
Pipeline LLM complet : exécute tous les modules dans l'ordre.
Lance avec : python -m llm.pipeline_llm
"""
import pandas as pd
import logging
from pathlib import Path

from llm.config import get_llm, get_model_info
from llm.summarizer import summarize_products
from llm.analyst import generate_trend_report, save_report, generate_client_profiles
from llm.marketing import generate_marketing_recommendations

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s"
)
logger = logging.getLogger("LLM.Pipeline")


def run_llm_pipeline(
    csv_path: str = "data/processed/products_enriched.csv",
    n_summaries: int = 30,
    n_marketing: int = 10,
    skip_summaries: bool = False,
) -> dict:
    """
    Pipeline LLM complet en 4 étapes.

    Args:
        csv_path       : chemin vers le dataset enrichi
        n_summaries    : nb de résumés produits à générer
        n_marketing    : nb de recommandations marketing
        skip_summaries : skip l'étape résumés (économise des tokens)

    Returns:
        dict avec le rapport, les profils et le df enrichi LLM
    """
    print("\n" + "="*60)
    print("  PIPELINE LLM — Smart eCommerce Intelligence")
    print("="*60)

    # ── Chargement ─────────────────────────────────────────────────
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset non trouvé : {csv_path}\n"
            "Lance d'abord : python -m ml.pipeline"
        )

    df = pd.read_csv(csv_path)
    print(f"\n[LLM] Dataset chargé : {len(df)} produits")
    print(f"[LLM] Modèle utilisé : {get_model_info()['model']}")

    results = {}

    # ── Étape 1 : Résumés produits ─────────────────────────────────
    if not skip_summaries:
        print(f"\n[1/4] Résumés automatiques ({n_summaries} produits)...")
        df = summarize_products(df, n_samples=n_summaries)
        n_done = (df["description_llm"] != "").sum()
        print(f"  {n_done} résumés générés")
    else:
        print("\n[1/4] Résumés — ignorés (skip_summaries=True)")
        df["description_llm"] = ""

    # ── Étape 2 : Rapport de tendances ────────────────────────────
    print("\n[2/4] Génération du rapport de tendances...")
    report = generate_trend_report(df)
    report_path = save_report(report)
    results["report"]      = report
    results["report_path"] = report_path
    print(f"  Rapport sauvegardé : {report_path}")

    # ── Étape 3 : Profils clients par segment ─────────────────────
    print("\n[3/4] Génération des profils clients...")
    profiles = generate_client_profiles(df)
    results["profiles"] = profiles
    print(f"  {len(profiles)} profils générés : {list(profiles.keys())}")

    # ── Étape 4 : Recommandations marketing ───────────────────────
    print(f"\n[4/4] Recommandations marketing ({n_marketing} produits)...")
    df = generate_marketing_recommendations(df, n_top=n_marketing)
    n_reco = (df["recommandation_marketing"] != "").sum()
    print(f"  {n_reco} recommandations générées")

    # ── Sauvegarde finale ─────────────────────────────────────────
    output_path = "data/processed/products_llm_enriched.csv"
    df.to_csv(output_path, index=False)
    results["df"]          = df
    results["output_path"] = output_path

    print("\n" + "="*60)
    print("  PIPELINE LLM TERMINÉ")
    print(f"  Dataset LLM : {output_path}")
    print(f"  Rapport     : {report_path}")
    print("="*60)

    return results


if __name__ == "__main__":
    results = run_llm_pipeline(
        n_summaries=20,      # réduit pour limiter les coûts
        n_marketing=10,
        skip_summaries=False,
    )

    print("\n=== EXTRAIT DU RAPPORT ===")
    print(results["report"][:800])
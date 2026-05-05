import concurrent.futures
from agents.shopify_agent import ShopifyAgent
from agents.woocommerce_agent import WooCommerceAgent
from agents.playwright_agent import PlaywrightAgent
from utils.normalizer import normalize_products
from utils.storage import save_to_csv, save_to_sqlite
import logging

logger = logging.getLogger("Orchestrator")

def run_scraping_pipeline(config: dict) -> "pd.DataFrame":
    """
    Lance tous les agents en parallèle et retourne un DataFrame normalisé.
    
    config = {
        "shopify_urls": [...],
        "woocommerce": [{"url": ..., "key": ..., "secret": ...}],
        "playwright_urls": [...],
    }
    """
    all_results = []
    agents = []

    # ── Préparation des agents ──────────────────────────────────────
    if config.get("shopify_urls"):
        agents.append(ShopifyAgent(shop_urls=config["shopify_urls"]))

    for woo in config.get("woocommerce", []):
        agents.append(WooCommerceAgent(
            shop_url=woo["url"],
            consumer_key=woo["key"],
            consumer_secret=woo["secret"]
        ))

    if config.get("playwright_urls"):
        agents.append(PlaywrightAgent(urls=config["playwright_urls"]))

    # ── Exécution parallèle ─────────────────────────────────────────
    logger.info(f"Lancement de {len(agents)} agents en parallèle...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(agent.run): agent.name for agent in agents}

        for future in concurrent.futures.as_completed(futures):
            agent_name = futures[future]
            try:
                results = future.result()
                all_results.extend(results)
                logger.info(f"[OK] {agent_name} → {len(results)} produits")
            except Exception as e:
                logger.error(f"[ERREUR] {agent_name} : {e}")

    # ── Normalisation ───────────────────────────────────────────────
    df = normalize_products(all_results)

    # ── Sauvegarde ──────────────────────────────────────────────────
    save_to_csv(df, "data/raw/products.csv")
    save_to_sqlite(df, "data/raw/products.db")

    return df
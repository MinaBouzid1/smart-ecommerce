"""
MCP Server — Scraping Shopify.
Déclare ses permissions et expose ses outils de façon contrôlée.
"""
import time
import logging
import pandas as pd
from pathlib import Path

from mcp.permissions import (
    PermissionManager, ServerManifest,
    Permission, permission_manager
)
from mcp.audit_log import AuditLogger, ActionStatus, audit_logger

logger = logging.getLogger("MCP.ScrapingServer")

# ── Manifeste déclaratif ───────────────────────────────────────────
MANIFEST = ServerManifest(
    server_id    = "scraping-server",
    description  = "Scrape les données produits depuis les boutiques Shopify",
    permissions  = [
        Permission.READ_DATA,
        Permission.CALL_EXTERNAL,
        Permission.WRITE_DATA,
    ],
    tools        = ["scrape_shopify", "get_raw_data", "get_scraping_status"],
    data_sources = ["shopify-api", "data/raw/products.csv"],
    approved     = True,    # approuvé manuellement par l'équipe
)

# Enregistrement au démarrage
permission_manager.register_server(MANIFEST)


class ScrapingServer:
    """
    MCP Server dédié au scraping.
    Chaque outil vérifie ses permissions avant d'agir.
    Toutes les actions sont auditées.
    """
    SERVER_ID = "scraping-server"

    def __init__(self):
        self.pm  = permission_manager
        self.log = audit_logger

    def scrape_shopify(self, urls: list[str], max_pages: int = 3) -> dict:
        """
        Outil : scrape des boutiques Shopify.
        Permission requise : CALL_EXTERNAL + WRITE_DATA
        """
        start = time.time()

        # Vérification permissions
        try:
            self.pm.require(self.SERVER_ID, Permission.CALL_EXTERNAL, "scrape_shopify")
            self.pm.require(self.SERVER_ID, Permission.WRITE_DATA,    "scrape_shopify")
        except PermissionError as e:
            self.log.log(self.SERVER_ID, "scrape_shopify", "scraping",
                         ActionStatus.DENIED, error=str(e))
            return {"success": False, "error": str(e)}

        # Déclaration d'intention (principe MCP)
        logger.info(
            f"[ScrapingServer] INTENTION : scraper {len(urls)} boutiques "
            f"(max {max_pages} pages chacune)"
        )

        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from agents.orchestrator import run_scraping_pipeline

            df = run_scraping_pipeline({
                "shopify_urls":   urls,
                "woocommerce":    [],
                "playwright_urls": [],
            })

            duration = (time.time() - start) * 1000
            self.log.log(
                self.SERVER_ID, "scrape_shopify",
                f"Scraped {len(df)} products from {len(urls)} shops",
                ActionStatus.SUCCESS, duration,
                details={"nb_products": len(df), "shops": urls}
            )
            return {"success": True, "nb_products": len(df), "data": df}

        except Exception as e:
            duration = (time.time() - start) * 1000
            self.log.log(self.SERVER_ID, "scrape_shopify", "scraping",
                         ActionStatus.FAILURE, duration, error=str(e))
            return {"success": False, "error": str(e)}

    def get_raw_data(self) -> dict:
        """
        Outil : retourne les données brutes déjà collectées.
        Permission requise : READ_DATA
        """
        start = time.time()
        try:
            self.pm.require(self.SERVER_ID, Permission.READ_DATA, "get_raw_data")
        except PermissionError as e:
            self.log.log(self.SERVER_ID, "get_raw_data", "read",
                         ActionStatus.DENIED, error=str(e))
            return {"success": False, "error": str(e)}

        path = Path("data/raw/products.csv")
        if not path.exists():
            return {"success": False, "error": "Données non disponibles"}

        df = pd.read_csv(path)
        duration = (time.time() - start) * 1000
        self.log.log(
            self.SERVER_ID, "get_raw_data", "read CSV",
            ActionStatus.SUCCESS, duration,
            details={"rows": len(df), "columns": list(df.columns)}
        )
        return {"success": True, "nb_products": len(df), "data": df}

    def get_scraping_status(self) -> dict:
        """Outil : vérifie si les données scrappées sont disponibles."""
        self.pm.require(self.SERVER_ID, Permission.READ_DATA, "get_scraping_status")

        raw_path      = Path("data/raw/products.csv")
        enriched_path = Path("data/processed/products_enriched.csv")

        status = {
            "raw_data_available":      raw_path.exists(),
            "enriched_data_available": enriched_path.exists(),
            "raw_products":            0,
            "enriched_products":       0,
        }
        if raw_path.exists():
            status["raw_products"] = len(pd.read_csv(raw_path))
        if enriched_path.exists():
            status["enriched_products"] = len(pd.read_csv(enriched_path))

        self.log.log(self.SERVER_ID, "get_scraping_status", "status check",
                     ActionStatus.SUCCESS, details=status)
        return status
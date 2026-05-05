"""
MCP Client : orchestre les serveurs, valide les permissions,
et fournit une interface unifiée à l'application hôte.
"""
import logging
import pandas as pd
from mcp.permissions import permission_manager
from mcp.audit_log import audit_logger
from mcp.servers.scraping_server import ScrapingServer
from mcp.servers.ml_server import MLServer
from mcp.servers.llm_server import LLMServer

logger = logging.getLogger("MCP.Client")


class MCPClient:
    """
    Point d'entrée unique pour toutes les opérations MCP.
    L'application (Host) n'interagit jamais directement
    avec les serveurs — tout passe par le Client.
    """

    def __init__(self):
        self.scraping = ScrapingServer()
        self.ml       = MLServer()
        self.llm      = LLMServer()
        logger.info("[MCPClient] Initialisé avec 3 serveurs")

    # ── Scraping ─────────────────────────────────────────────────

    def run_scraping(self, urls: list[str], max_pages: int = 3) -> dict:
        logger.info(f"[MCPClient] → ScrapingServer.scrape_shopify ({len(urls)} URLs)")
        return self.scraping.scrape_shopify(urls, max_pages)

    def get_data_status(self) -> dict:
        return self.scraping.get_scraping_status()

    # ── ML ───────────────────────────────────────────────────────

    def run_ml(self, k: int = 20) -> dict:
        logger.info(f"[MCPClient] → MLServer.run_pipeline (k={k})")
        return self.ml.run_pipeline(k)

    def get_top_k(self, k: int = 20, categorie: str = None) -> dict:
        return self.ml.get_top_k(k, categorie)

    def get_segments(self) -> dict:
        return self.ml.get_segments()

    # ── LLM ──────────────────────────────────────────────────────

    def generate_report(self) -> dict:
        logger.info("[MCPClient] → LLMServer.generate_report")
        return self.llm.generate_report()

    def chat(self, question: str, df: pd.DataFrame = None) -> dict:
        return self.llm.chat(question, df)

    # ── Audit & Monitoring ────────────────────────────────────────

    def get_audit_stats(self) -> dict:
        return audit_logger.get_stats()

    def get_recent_actions(self, n: int = 10) -> list[dict]:
        return audit_logger.get_recent(n)

    def get_permission_summary(self) -> dict:
        return permission_manager.summary()

    def get_denied_actions(self) -> list[dict]:
        return permission_manager.get_denied_log()
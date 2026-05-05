"""
MCP Server — Intelligence LLM.
Expose les outils LLM avec contrôle strict des appels externes.
"""
import time
import logging
import pandas as pd
from pathlib import Path

from mcp.permissions import (
    ServerManifest, Permission,
    permission_manager
)
from mcp.audit_log import ActionStatus, audit_logger

logger = logging.getLogger("MCP.LLMServer")

MANIFEST = ServerManifest(
    server_id    = "llm-server",
    description  = "Génère insights, résumés et recommandations via LLM Groq",
    permissions  = [
        Permission.READ_DATA,
        Permission.CALL_LLM,
        Permission.WRITE_DATA,
        Permission.EXPORT_DATA,
    ],
    tools        = ["generate_report", "summarize_products",
                    "generate_marketing", "chat"],
    data_sources = ["data/processed/products_enriched.csv"],
    approved     = True,
)

permission_manager.register_server(MANIFEST)


class LLMServer:
    SERVER_ID = "llm-server"

    def __init__(self):
        self.pm  = permission_manager
        self.log = audit_logger

    def generate_report(self) -> dict:
        """Outil : génère le rapport de tendances."""
        start = time.time()
        try:
            self.pm.require(self.SERVER_ID, Permission.CALL_LLM,   "generate_report")
            self.pm.require(self.SERVER_ID, Permission.READ_DATA,  "generate_report")
            self.pm.require(self.SERVER_ID, Permission.WRITE_DATA, "generate_report")
        except PermissionError as e:
            self.log.log(self.SERVER_ID, "generate_report", "LLM report",
                         ActionStatus.DENIED, error=str(e))
            return {"success": False, "error": str(e)}

        logger.info("[LLMServer] INTENTION : générer rapport de tendances via Groq")

        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from llm.analyst import generate_trend_report, save_report

            path = Path("data/processed/products_enriched.csv")
            if not path.exists():
                return {"success": False, "error": "Dataset enrichi non trouvé"}

            df      = pd.read_csv(path)
            report  = generate_trend_report(df)
            rpath   = save_report(report)
            duration = (time.time() - start) * 1000

            self.log.log(
                self.SERVER_ID, "generate_report", "trend report",
                ActionStatus.SUCCESS, duration,
                details={"report_path": rpath, "nb_chars": len(report)}
            )
            return {"success": True, "report": report, "path": rpath}

        except Exception as e:
            duration = (time.time() - start) * 1000
            self.log.log(self.SERVER_ID, "generate_report", "trend report",
                         ActionStatus.FAILURE, duration, error=str(e))
            return {"success": False, "error": str(e)}

    def chat(self, question: str, df: pd.DataFrame = None) -> dict:
        """Outil : chatbot Q&A sur le dataset."""
        start = time.time()
        try:
            self.pm.require(self.SERVER_ID, Permission.CALL_LLM,  "chat")
            self.pm.require(self.SERVER_ID, Permission.READ_DATA, "chat")
        except PermissionError as e:
            self.log.log(self.SERVER_ID, "chat", question,
                         ActionStatus.DENIED, error=str(e))
            return {"success": False, "error": str(e)}

        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from llm.chatbot import EcommerceChatbot

            if df is None:
                path = Path("data/processed/products_enriched.csv")
                df   = pd.read_csv(path) if path.exists() else pd.DataFrame()

            bot      = EcommerceChatbot(df)
            response = bot.chat(question)
            duration = (time.time() - start) * 1000

            self.log.log(
                self.SERVER_ID, "chat", question[:80],
                ActionStatus.SUCCESS, duration,
                details={"question_len": len(question), "response_len": len(response)}
            )
            return {"success": True, "response": response}

        except Exception as e:
            duration = (time.time() - start) * 1000
            self.log.log(self.SERVER_ID, "chat", question[:80],
                         ActionStatus.FAILURE, duration, error=str(e))
            return {"success": False, "error": str(e)}
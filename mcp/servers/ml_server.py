"""
MCP Server — Pipeline ML.
Expose les outils d'analyse ML de façon contrôlée.
"""
import time
import logging
import pandas as pd
from pathlib import Path

from mcp.permissions import (
    ServerManifest, Permission,
    permission_manager, PermissionManager
)
from mcp.audit_log import ActionStatus, audit_logger

logger = logging.getLogger("MCP.MLServer")

MANIFEST = ServerManifest(
    server_id    = "ml-server",
    description  = "Exécute le pipeline ML : clustering, classification, scoring Top-K",
    permissions  = [
        Permission.READ_DATA,
        Permission.RUN_ML,
        Permission.WRITE_DATA,
        Permission.EXPORT_DATA,
    ],
    tools        = ["run_pipeline", "get_top_k", "get_segments", "get_metrics"],
    data_sources = ["data/raw/products.csv", "data/processed/"],
    approved     = True,
)

permission_manager.register_server(MANIFEST)


class MLServer:
    SERVER_ID = "ml-server"

    def __init__(self):
        self.pm  = permission_manager
        self.log = audit_logger

    def run_pipeline(self, k: int = 20) -> dict:
        """Outil : exécute le pipeline ML complet."""
        start = time.time()
        try:
            self.pm.require(self.SERVER_ID, Permission.RUN_ML,     "run_pipeline")
            self.pm.require(self.SERVER_ID, Permission.WRITE_DATA, "run_pipeline")
        except PermissionError as e:
            self.log.log(self.SERVER_ID, "run_pipeline", "ML pipeline",
                         ActionStatus.DENIED, error=str(e))
            return {"success": False, "error": str(e)}

        logger.info(f"[MLServer] INTENTION : exécuter pipeline ML (Top-{k})")

        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from ml.pipeline import run_ml_pipeline

            df_ml, top_k, metrics = run_ml_pipeline(k=k)
            duration = (time.time() - start) * 1000

            self.log.log(
                self.SERVER_ID, "run_pipeline", f"ML pipeline Top-{k}",
                ActionStatus.SUCCESS, duration,
                details={
                    "nb_products":    len(df_ml),
                    "nb_top_k":       len(top_k),
                    "accuracy_xgb":   metrics.get("xgboost", {}).get("accuracy", 0),
                    "f1_xgb":         metrics.get("xgboost", {}).get("f1", 0),
                }
            )
            return {
                "success":  True,
                "nb_products": len(df_ml),
                "top_k":    top_k,
                "metrics":  metrics,
            }

        except Exception as e:
            duration = (time.time() - start) * 1000
            self.log.log(self.SERVER_ID, "run_pipeline", "ML pipeline",
                         ActionStatus.FAILURE, duration, error=str(e))
            return {"success": False, "error": str(e)}

    def get_top_k(self, k: int = 20, categorie: str = None) -> dict:
        """Outil : retourne le Top-K depuis le fichier sauvegardé."""
        start = time.time()
        self.pm.require(self.SERVER_ID, Permission.READ_DATA, "get_top_k")

        path = Path(f"data/processed/top_{k}_produits.csv")
        if not path.exists():
            path = Path("data/processed/products_enriched.csv")
            if not path.exists():
                return {"success": False, "error": "Fichier Top-K non trouvé"}

        df = pd.read_csv(path)
        if categorie:
            df = df[df["categorie"].str.contains(categorie, case=False, na=False)]

        top = df.nlargest(k, "score_composite") \
              if "score_composite" in df.columns else df.head(k)

        duration = (time.time() - start) * 1000
        self.log.log(
            self.SERVER_ID, "get_top_k", f"Top-{k}",
            ActionStatus.SUCCESS, duration,
            details={"k": k, "categorie": categorie, "found": len(top)}
        )
        return {"success": True, "top_k": top, "count": len(top)}

    def get_segments(self) -> dict:
        """Outil : retourne la distribution des segments KMeans."""
        self.pm.require(self.SERVER_ID, Permission.READ_DATA, "get_segments")

        path = Path("data/processed/products_enriched.csv")
        if not path.exists():
            return {"success": False, "error": "Pipeline ML non exécuté"}

        df = pd.read_csv(path)
        if "segment" not in df.columns:
            return {"success": False, "error": "Colonne 'segment' absente"}

        segments = df["segment"].value_counts().to_dict()
        self.log.log(self.SERVER_ID, "get_segments", "segments",
                     ActionStatus.SUCCESS, details=segments)
        return {"success": True, "segments": segments}

    def get_metrics(self) -> dict:
        """Outil : retourne les métriques ML depuis les fichiers sauvegardés."""
        self.pm.require(self.SERVER_ID, Permission.READ_DATA, "get_metrics")

        metrics = {}
        for img in ["feature_importance.png",
                    "confusion_matrix_xgboost.png",
                    "elbow_silhouette.png",
                    "pca_plot.png"]:
            p = Path(f"data/processed/{img}")
            metrics[img] = p.exists()

        self.log.log(self.SERVER_ID, "get_metrics", "metrics check",
                     ActionStatus.SUCCESS, details=metrics)
        return {"success": True, "artifacts": metrics}
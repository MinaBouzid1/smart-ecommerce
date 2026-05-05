"""
Système de permissions MCP.
Chaque serveur déclare ce qu'il peut faire.
Le client valide avant d'exécuter.
"""
from enum import Enum
from dataclasses import dataclass, field
from typing import Any
import logging

logger = logging.getLogger("MCP.Permissions")


class Permission(Enum):
    """Permissions disponibles dans le système MCP."""
    READ_DATA       = "read_data"        # lire les données produits
    WRITE_DATA      = "write_data"       # écrire/modifier des données
    CALL_EXTERNAL   = "call_external"    # appeler des APIs externes
    CALL_LLM        = "call_llm"         # appeler un LLM
    RUN_ML          = "run_ml"           # exécuter des modèles ML
    EXPORT_DATA     = "export_data"      # exporter des fichiers
    ACCESS_CONFIG   = "access_config"    # lire la configuration


@dataclass
class ServerManifest:
    """
    Manifeste déclaratif d'un serveur MCP.
    Chaque serveur DOIT déclarer ses permissions et capacités
    avant d'être autorisé à opérer.
    """
    server_id:    str
    description:  str
    permissions:  list[Permission]
    tools:        list[str]              # liste des outils exposés
    data_sources: list[str]             # sources de données accédées
    author:       str = "smart-ecommerce-team"
    version:      str = "1.0.0"
    approved:     bool = False           # validé manuellement


class PermissionManager:
    """
    Gestionnaire central des permissions MCP.
    Valide chaque requête avant exécution.
    Principe : moindre privilège — chaque serveur n'a accès
    qu'à ce dont il a strictement besoin.
    """

    def __init__(self):
        self._manifests: dict[str, ServerManifest] = {}
        self._denied_log: list[dict] = []

    def register_server(self, manifest: ServerManifest) -> bool:
        """
        Enregistre un serveur avec son manifeste.
        Un serveur non enregistré ne peut pas opérer.
        """
        self._manifests[manifest.server_id] = manifest
        status = "APPROUVÉ" if manifest.approved else "EN ATTENTE"
        logger.info(
            f"[Permissions] Serveur '{manifest.server_id}' enregistré "
            f"({len(manifest.permissions)} permissions) — {status}"
        )
        return manifest.approved

    def check(self,
              server_id: str,
              permission: Permission,
              context: str = "") -> bool:
        """
        Vérifie si un serveur a la permission demandée.
        Journalise les refus pour audit.
        """
        manifest = self._manifests.get(server_id)

        # Serveur non enregistré → refus systématique
        if not manifest:
            self._log_denial(server_id, permission, "Serveur non enregistré", context)
            return False

        # Serveur non approuvé → refus
        if not manifest.approved:
            self._log_denial(server_id, permission, "Serveur non approuvé", context)
            return False

        # Permission non déclarée → refus
        if permission not in manifest.permissions:
            self._log_denial(
                server_id, permission,
                f"Permission '{permission.value}' non déclarée dans le manifeste",
                context
            )
            return False

        logger.debug(
            f"[Permissions] ✓ {server_id} → {permission.value}"
            + (f" ({context})" if context else "")
        )
        return True

    def require(self,
                server_id: str,
                permission: Permission,
                context: str = "") -> None:
        """
        Vérifie une permission et lève une exception si refusée.
        À utiliser dans les serveurs pour un contrôle strict.
        """
        if not self.check(server_id, permission, context):
            raise PermissionError(
                f"[MCP] Accès refusé : '{server_id}' n'a pas "
                f"la permission '{permission.value}'"
                + (f" — contexte : {context}" if context else "")
            )

    def _log_denial(self, server_id: str, permission: Permission,
                    reason: str, context: str):
        entry = {
            "server_id":  server_id,
            "permission": permission.value,
            "reason":     reason,
            "context":    context,
        }
        self._denied_log.append(entry)
        logger.warning(
            f"[Permissions] ✗ REFUS — {server_id} → "
            f"{permission.value} : {reason}"
        )

    def get_denied_log(self) -> list[dict]:
        return self._denied_log.copy()

    def summary(self) -> dict:
        return {
            "servers_registered": len(self._manifests),
            "servers_approved":   sum(1 for m in self._manifests.values() if m.approved),
            "total_denials":      len(self._denied_log),
        }


# Instance globale partagée
permission_manager = PermissionManager()
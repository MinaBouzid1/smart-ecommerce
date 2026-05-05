"""
Audit Log MCP : journalise toutes les actions des agents.
Principe de responsabilité : chaque action est traçable.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger("MCP.AuditLog")


class ActionStatus(Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    DENIED  = "DENIED"
    WARNING = "WARNING"


@dataclass
class AuditEntry:
    """Une entrée d'audit — immuable une fois créée."""
    timestamp:   str
    server_id:   str
    tool_name:   str
    action:      str
    status:      str
    duration_ms: float
    details:     dict = field(default_factory=dict)
    error:       str  = ""


class AuditLogger:
    """
    Journalise toutes les actions MCP dans un fichier JSONL
    (une entrée JSON par ligne — format standard pour les logs).

    Chaque action est enregistrée avec :
    - qui a agi (server_id)
    - quoi (tool_name + action)
    - quand (timestamp ISO)
    - résultat (SUCCESS / FAILURE / DENIED)
    - combien de temps (duration_ms)
    - détails contextuels
    """

    def __init__(self, log_dir: str = "data/logs"):
        self.log_dir  = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / "mcp_audit.jsonl"
        self._entries: list[AuditEntry] = []
        logger.info(f"[AuditLog] Log MCP : {self.log_path}")

    def log(self,
            server_id:   str,
            tool_name:   str,
            action:      str,
            status:      ActionStatus,
            duration_ms: float = 0.0,
            details:     dict  = None,
            error:       str   = "") -> AuditEntry:
        """Enregistre une action dans le log."""
        entry = AuditEntry(
            timestamp=datetime.now().isoformat(),
            server_id=server_id,
            tool_name=tool_name,
            action=action,
            status=status.value,
            duration_ms=round(duration_ms, 2),
            details=details or {},
            error=error,
        )
        self._entries.append(entry)

        # Écriture immédiate dans le fichier JSONL
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")

        # Log console selon le statut
        msg = (f"[{entry.status}] {server_id}.{tool_name} "
               f"— {action} ({duration_ms:.0f}ms)")
        if status == ActionStatus.SUCCESS:
            logger.info(msg)
        elif status == ActionStatus.FAILURE:
            logger.error(msg + (f" — {error}" if error else ""))
        elif status == ActionStatus.DENIED:
            logger.warning(msg)

        return entry

    def get_stats(self) -> dict:
        """Statistiques du log pour le rapport."""
        if not self._entries:
            return {"total": 0}

        by_status = {}
        by_server = {}
        total_ms  = 0.0

        for e in self._entries:
            by_status[e.status]    = by_status.get(e.status, 0) + 1
            by_server[e.server_id] = by_server.get(e.server_id, 0) + 1
            total_ms += e.duration_ms

        return {
            "total":          len(self._entries),
            "by_status":      by_status,
            "by_server":      by_server,
            "avg_duration_ms": round(total_ms / len(self._entries), 2),
            "log_file":       str(self.log_path),
        }

    def get_recent(self, n: int = 20) -> list[dict]:
        """Retourne les N dernières entrées."""
        return [asdict(e) for e in self._entries[-n:]]

    def load_from_file(self) -> list[dict]:
        """Charge le log depuis le fichier JSONL."""
        if not self.log_path.exists():
            return []
        entries = []
        with open(self.log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return entries


# Instance globale partagée
audit_logger = AuditLogger()
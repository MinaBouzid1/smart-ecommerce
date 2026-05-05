import time
import logging
import requests
from abc import ABC, abstractmethod
from fake_useragent import UserAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

class BaseAgent(ABC):
    """
    Classe mère pour tous les agents A2A.
    Chaque agent hérite de cette classe et implémente scrape().
    """

    def __init__(self, name: str, delay: float = 1.5):
        self.name = name
        self.delay = delay          # délai entre requêtes (politeness)
        self.logger = logging.getLogger(name)
        self.ua = UserAgent()
        self.session = requests.Session()
        self.results = []

    def _get_headers(self) -> dict:
        """Headers dynamiques pour éviter le blocage."""
        return {
            "User-Agent": self.ua.random,
            "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
            "Accept": "application/json, text/html",
        }

    def _safe_get(self, url: str, params: dict = None) -> requests.Response | None:
        """Requête GET sécurisée avec retry automatique."""
        for attempt in range(3):
            try:
                resp = self.session.get(
                    url,
                    headers=self._get_headers(),
                    params=params,
                    timeout=10
                )
                resp.raise_for_status()
                time.sleep(self.delay)
                return resp
            except requests.RequestException as e:
                self.logger.warning(f"Tentative {attempt+1}/3 échouée : {e}")
                time.sleep(2 ** attempt)   # backoff exponentiel
        self.logger.error(f"Echec total pour : {url}")
        return None

    @abstractmethod
    def scrape(self, **kwargs) -> list[dict]:
        """Méthode principale à implémenter dans chaque agent."""
        pass

    def run(self, **kwargs) -> list[dict]:
        self.logger.info(f"Agent '{self.name}' démarré.")
        self.results = self.scrape(**kwargs)
        self.logger.info(f"Agent '{self.name}' terminé — {len(self.results)} produits collectés.")
        return self.results
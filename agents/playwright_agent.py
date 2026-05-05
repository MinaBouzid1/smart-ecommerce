"""
Agent Playwright — scrape les sites e-commerce à rendu JavaScript.
Supporte : Nike, ASOS, H&M, Zalando, et sites génériques.
"""
import asyncio
import re
import logging
from playwright.async_api import async_playwright, TimeoutError as PWTimeout
from agents.base_agent import BaseAgent

logger = logging.getLogger("PlaywrightAgent")


class PlaywrightAgent(BaseAgent):
    """
    Agent qui scrape les sites JavaScript dynamiques.
    Lance un vrai navigateur Chromium headless.
    """

    # Sélecteurs CSS spécifiques par site
    SITE_CONFIGS = {
        "nike.com": {
            "product_card":  ".product-card__body, .product-card",
            "name":          ".product-card__title, .product-card__subtitle",
            "price":         ".product-price__wrapper, .css-b9fpep",
            "image":         ".product-card__hero-image img",
            "link":          ".product-card__link-overlay",
            "scroll_times":  4,
            "wait_ms":       2000,
        },
        "asos.com": {
            "product_card":  "[data-auto-id='productTile'], ._2qG5x",
            "name":          "[class*='productDescription'], ._3_FN3",
            "price":         "[class*='currentPrice'], .cPFAd",
            "image":         "img[class*='image']",
            "link":          "a[href*='/prd/']",
            "scroll_times":  5,
            "wait_ms":       2500,
        },
        "hm.com": {
            "product_card":  ".product-item, li.product-item",
            "name":          ".item-heading, .product-item-headline",
            "price":         ".price, .item-price",
            "image":         ".product-item-image img",
            "link":          "a.product-item-link",
            "scroll_times":  3,
            "wait_ms":       2000,
        },
        "zalando": {
            "product_card":  "article[class*='card'], ._5qdMrS",
            "name":          "[class*='title'], h3",
            "price":         "[class*='price'], ._1NvV9A",
            "image":         "img[class*='image']",
            "link":          "a[href*='/']",
            "scroll_times":  4,
            "wait_ms":       2000,
        },
        # Config générique pour tous les autres sites
        "default": {
            "product_card":  (
                "[data-product-id], .product-card, .product-item, "
                ".product-tile, article.product, .item-card, "
                "[class*='ProductCard'], [class*='product_card']"
            ),
            "name":          (
                "h2, h3, .product-title, .product-name, "
                "[class*='title'], [class*='name']"
            ),
            "price":         (
                ".price, [class*='price'], [data-price], "
                "[class*='Price'], span[itemprop='price']"
            ),
            "image":         "img",
            "link":          "a",
            "scroll_times":  3,
            "wait_ms":       1500,
        }
    }

    def __init__(self, urls: list[str], headless: bool = True):
        super().__init__(name="PlaywrightAgent")
        self.urls     = urls
        self.headless = headless

    def _get_config(self, url: str) -> dict:
        """Retourne la config du site selon l'URL."""
        for key, cfg in self.SITE_CONFIGS.items():
            if key in url:
                return cfg
        return self.SITE_CONFIGS["default"]

    def scrape(self, **kwargs) -> list[dict]:
        """Point d'entrée synchrone."""
        return asyncio.run(self._scrape_all())

    async def _scrape_all(self) -> list[dict]:
        results = []
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(
                headless=self.headless,
                args=[
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-blink-features=AutomationControlled",
                ]
            )
            context = await browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1920, "height": 1080},
                locale="fr-FR",
            )

            # Anti-détection
            await context.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                Object.defineProperty(navigator, 'plugins', {get: () => [1,2,3]});
            """)

            page = await context.new_page()

            for url in self.urls:
                try:
                    logger.info(f"[Playwright] Scraping : {url}")
                    products = await self._scrape_url(page, url)
                    results.extend(products)
                    logger.info(f"[Playwright] {url} → {len(products)} produits")
                except Exception as e:
                    logger.error(f"[Playwright] Erreur sur {url} : {e}")

            await browser.close()

        logger.info(f"[Playwright] Total : {len(results)} produits collectés")
        return results

    async def _scrape_url(self, page, url: str) -> list[dict]:
        """Scrape une URL avec gestion du scroll et du lazy loading."""
        cfg = self._get_config(url)

        # Navigation
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        except PWTimeout:
            logger.warning(f"Timeout navigation : {url}")
            return []

        # Fermeture popups/cookies courants
        await self._close_popups(page)

        # Scroll progressif pour déclencher le lazy loading
        for _ in range(cfg["scroll_times"]):
            await page.evaluate("""
                window.scrollBy({top: window.innerHeight * 0.8, behavior: 'smooth'});
            """)
            await page.wait_for_timeout(cfg["wait_ms"])

        # Retour en haut
        await page.evaluate("window.scrollTo(0, 0)")
        await page.wait_for_timeout(500)

        # Extraction JavaScript côté navigateur
        products_raw = await page.evaluate(f"""
            () => {{
                const results = [];
                const cards = document.querySelectorAll('{cfg["product_card"]}');

                cards.forEach(card => {{
                    // Nom du produit
                    const nameEl = card.querySelector('{cfg["name"]}');
                    const nom = nameEl?.innerText?.trim() || nameEl?.textContent?.trim() || '';

                    // Prix
                    const priceEl = card.querySelector('{cfg["price"]}');
                    const prixRaw = priceEl?.innerText?.trim() || priceEl?.getAttribute('data-price') || '0';

                    // Image
                    const imgEl = card.querySelector('{cfg["image"]}');
                    const image = imgEl?.src || imgEl?.getAttribute('data-src') || '';

                    // Lien
                    const linkEl = card.querySelector('{cfg["link"]}') || card.closest('a');
                    const link = linkEl?.href || '';

                    if (nom && nom.length > 2) {{
                        results.push({{
                            nom:         nom,
                            prix_raw:    prixRaw,
                            image:       image,
                            url_produit: link,
                        }});
                    }}
                }});

                return results;
            }}
        """)

        # Nettoyage et enrichissement
        products = []
        for p in products_raw:
            prix = self._parse_price(p.get("prix_raw", "0"))

            products.append({
                "source":       "playwright",
                "shop_url":     url,
                "product_id":   self._make_id(p.get("nom", ""), url),
                "nom":          p.get("nom", "").strip()[:200],
                "description":  "",
                "categorie":    self._guess_category(url),
                "marque":       self._extract_brand(url),
                "tags":         "",
                "prix":         prix,
                "prix_compare": 0.0,
                "remise_pct":   0.0,
                "disponible":   True,
                "stock":        0,
                "rating":       0.0,
                "nb_reviews":   0,
                "nb_images":    1 if p.get("image") else 0,
                "nb_variants":  1,
                "date_creation": "",
                "url_produit":  p.get("url_produit", ""),
                "image":        p.get("image", ""),
            })

        # Déduplication par nom
        seen = set()
        unique = []
        for p in products:
            key = p["nom"].lower()[:50]
            if key not in seen and p["prix"] >= 0:
                seen.add(key)
                unique.append(p)

        return unique

    async def _close_popups(self, page):
        """Ferme les popups courants (cookies, newsletters)."""
        popup_selectors = [
            # Cookies
            "button[id*='accept'], button[class*='accept']",
            "button[id*='cookie'], button[class*='cookie']",
            "#onetrust-accept-btn-handler",
            ".cc-btn.cc-allow",
            # Newsletters / modals
            "button[class*='close'], button[aria-label*='close']",
            "button[aria-label*='Close'], button[aria-label*='Fermer']",
            ".modal-close, .popup-close, .overlay-close",
        ]
        for sel in popup_selectors:
            try:
                btn = page.locator(sel).first
                if await btn.is_visible(timeout=1500):
                    await btn.click()
                    await page.wait_for_timeout(500)
            except Exception:
                pass

    def _parse_price(self, raw: str) -> float:
        """Extrait le prix numérique d'une chaîne."""
        if not raw:
            return 0.0
        # Supprimer tout sauf chiffres, virgule, point
        cleaned = re.sub(r"[^\d.,]", "", str(raw))
        if not cleaned:
            return 0.0
        # Gérer virgule décimale
        cleaned = cleaned.replace(",", ".")
        # Si plusieurs points → garder le dernier comme décimal
        parts = cleaned.split(".")
        if len(parts) > 2:
            cleaned = "".join(parts[:-1]) + "." + parts[-1]
        try:
            return round(float(cleaned), 2)
        except ValueError:
            return 0.0

    def _make_id(self, nom: str, url: str) -> str:
        """Génère un ID unique depuis le nom et l'URL."""
        import hashlib
        raw = f"{url}_{nom}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def _guess_category(self, url: str) -> str:
        """Devine la catégorie depuis l'URL."""
        url_lower = url.lower()
        mapping = {
            "shoe": "Chaussures", "sneaker": "Chaussures", "running": "Running",
            "men": "Homme",       "women": "Femme",        "kids": "Enfants",
            "jacket": "Vestes",   "shirt": "Hauts",        "pant": "Bas",
            "sport": "Sport",     "fitness": "Fitness",    "gym": "Gym",
            "dress": "Robes",     "bag": "Sacs",           "watch": "Montres",
        }
        for key, cat in mapping.items():
            if key in url_lower:
                return cat
        return "Mode"

    def _extract_brand(self, url: str) -> str:
        """Extrait le nom de la marque depuis le domaine."""
        import re
        domain = re.sub(r"https?://(www\.)?", "", url).split("/")[0]
        brand  = domain.split(".")[0].capitalize()
        return brand
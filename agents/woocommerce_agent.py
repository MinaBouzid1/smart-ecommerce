from agents.base_agent import BaseAgent

class WooCommerceAgent(BaseAgent):
    """
    Agent qui utilise la REST API WooCommerce (v3).
    Requiert consumer_key et consumer_secret (lecture seule).
    
    Docs : https://woocommerce.github.io/woocommerce-rest-api-docs/
    """

    def __init__(self, shop_url: str, consumer_key: str, consumer_secret: str):
        super().__init__(name="WooCommerceAgent")
        self.shop_url   = shop_url.rstrip("/")
        self.api_url    = f"{shop_url}/wp-json/wc/v3"
        self.auth       = (consumer_key, consumer_secret)

    def scrape(self, max_pages: int = 10, per_page: int = 100, **kwargs) -> list[dict]:
        all_products = []

        for page in range(1, max_pages + 1):
            url = f"{self.api_url}/products"
            params = {
                "per_page": per_page,
                "page":     page,
                "status":   "publish",
                "orderby":  "popularity",
            }

            try:
                resp = self.session.get(
                    url, auth=self.auth, params=params, timeout=15
                )
                resp.raise_for_status()
            except Exception as e:
                self.logger.error(f"Erreur WooCommerce page {page}: {e}")
                break

            products = resp.json()
            if not products:
                break

            for p in products:
                # Prix
                prix = float(p.get("price", 0) or 0)
                prix_regular = float(p.get("regular_price", 0) or 0)
                prix_sale = float(p.get("sale_price", 0) or 0)

                # Note et avis
                rating = float(p.get("average_rating", 0) or 0)
                nb_reviews = int(p.get("rating_count", 0) or 0)

                product = {
                    "source":        "woocommerce",
                    "shop_url":      self.shop_url,
                    "product_id":    str(p.get("id", "")),
                    "nom":           p.get("name", "").strip(),
                    "description":   self._extract_text(p.get("description", "")),
                    "description_courte": self._extract_text(p.get("short_description", "")),
                    "categorie":     self._get_categories(p),
                    "tags":          self._get_tags(p),
                    "prix":          prix,
                    "prix_regular":  prix_regular,
                    "prix_promo":    prix_sale,
                    "remise_pct":    self._calc_remise(prix_regular, prix_sale),
                    "rating":        rating,
                    "nb_reviews":    nb_reviews,
                    "disponible":    p.get("in_stock", False),
                    "stock":         p.get("stock_quantity", 0) or 0,
                    "nb_ventes":     p.get("total_sales", 0),
                    "poids":         p.get("weight", ""),
                    "date_creation": p.get("date_created", ""),
                    "nb_images":     len(p.get("images", [])),
                }

                all_products.append(product)

            self.logger.info(f"Page {page} : {len(products)} produits")

        return all_products

    def _extract_text(self, html: str) -> str:
        from bs4 import BeautifulSoup
        if not html:
            return ""
        return BeautifulSoup(html, "html.parser").get_text(" ").strip()[:500]

    def _get_categories(self, p: dict) -> str:
        cats = p.get("categories", [])
        return ", ".join(c.get("name", "") for c in cats)

    def _get_tags(self, p: dict) -> str:
        tags = p.get("tags", [])
        return ", ".join(t.get("name", "") for t in tags)

    def _calc_remise(self, regular: float, sale: float) -> float:
        if regular > 0 and sale > 0 and sale < regular:
            return round((1 - sale / regular) * 100, 1)
        return 0.0
from agents.base_agent import BaseAgent

class ShopifyAgent(BaseAgent):
    """
    Agent qui exploite l'API JSON publique de Shopify.
    Tous les shops Shopify exposent /products.json sans authentification.
    
    Exemple : https://allbirds.com/products.json
    """

    def __init__(self, shop_urls: list[str], max_pages: int = 5):
        super().__init__(name="ShopifyAgent")
        self.shop_urls = shop_urls
        self.max_pages = max_pages

    def scrape(self, **kwargs) -> list[dict]:
        all_products = []

        for shop_url in self.shop_urls:
            shop_url = shop_url.rstrip("/")
            self.logger.info(f"Scraping Shopify : {shop_url}")

            for page in range(1, self.max_pages + 1):
                url = f"{shop_url}/products.json"
                resp = self._safe_get(url, params={"limit": 250, "page": page})

                if not resp:
                    break

                data = resp.json()
                products = data.get("products", [])

                if not products:
                    break   # plus de pages

                for p in products:
                    # On extrait le premier variant (le plus courant)
                    variant = p.get("variants", [{}])[0]

                    product = {
                        "source":       "shopify",
                        "shop_url":     shop_url,
                        "product_id":   str(p.get("id", "")),
                        "nom":          p.get("title", "").strip(),
                        "description":  self._clean_html(p.get("body_html", "")),
                        "categorie":    p.get("product_type", "").strip(),
                        "marque":       p.get("vendor", "").strip(),
                        "tags":         ", ".join(p.get("tags", [])),
                        "prix":         float(variant.get("price", 0) or 0),
                        "prix_compare": float(variant.get("compare_at_price") or 0),
                        "disponible":   variant.get("available", False),
                        "stock":        variant.get("inventory_quantity", 0),
                        "date_creation": p.get("created_at", ""),
                        "date_update":   p.get("updated_at", ""),
                        "nb_images":    len(p.get("images", [])),
                        "nb_variants":  len(p.get("variants", [])),
                    }

                    # Calcul de la remise si prix_compare renseigné
                    if product["prix_compare"] > 0 and product["prix"] > 0:
                        remise = (1 - product["prix"] / product["prix_compare"]) * 100
                        product["remise_pct"] = round(remise, 1)
                    else:
                        product["remise_pct"] = 0.0

                    all_products.append(product)

                self.logger.info(f"  Page {page} : {len(products)} produits")

        return all_products

    def _clean_html(self, html: str) -> str:
        """Supprime les balises HTML de la description."""
        from bs4 import BeautifulSoup
        if not html:
            return ""
        return BeautifulSoup(html, "html.parser").get_text(separator=" ").strip()[:500]
from agents.orchestrator import run_scraping_pipeline

if __name__ == "__main__":

    config = {
        "shopify_urls": [
        # Chaussures & sport
        "https://allbirds.com",
        "https://gymshark.com",
        "https://kith.com",
        "https://taylorstitch.com",
        "https://chubbiesshorts.com",

        # Mode & lifestyle
        "https://tentree.com",
        "https://outerknown.com",
        "https://tracksmith.com",

        # Montres & accessoires
        "https://mvmtwatches.com",
        "https://kapten-son.com",

        # Cosmétiques
        "https://jeffreestarcosmetics.com",
        "https://colourpop.com",

        # Tech & gadgets
        "https://nomadgoods.com",
        "https://峰.com",
    ],

        # WooCommerce (nécessite des clés API en lecture seule)
        "woocommerce": [
            # {
            #     "url": "https://monshop.com",
            #     "key": "ck_XXXX",
            #     "secret": "cs_XXXX"
            # }
        ],

         # Playwright lance un vrai navigateur Chromium headless
        "playwright_urls": [
            # Nike — page running homme
            "https://www.nike.com/fr/w/chaussures-de-running-homme-37v7jznik1zy7ok",
            # H&M — section homme
            "https://www2.hm.com/fr_fr/homme/nouveautes/voir-tout.html",
            # Pull&Bear — nouveautés
            "https://www.pullandbear.com/fr/homme/nouveautes-n5584",
            # ASOS — page produits
            "https://www.asos.com/fr/homme/hauts/cat/?cid=1614&currentpricerange=5-250",
        ],
    }

    df = run_scraping_pipeline(config)

    print("\n=== APERÇU DES DONNÉES ===")
    print(df[["source", "nom", "categorie", "prix", "rating", "disponible"]].head(10))
    print(f"\nTotal : {len(df)} produits | Colonnes : {list(df.columns)}")
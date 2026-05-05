import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder


def run_association_rules(df: pd.DataFrame,
                           min_support: float = 0.02,
                           min_confidence: float = 0.4,
                           min_lift: float = 1.1) -> pd.DataFrame:
    """
    Règles d'association entre catégories de produits.
    Robuste aux données pauvres et aux erreurs mémoire.
    """
    df = df.copy()

    # ── Nettoyage catégorie ────────────────────────────────────────
    df["categorie"] = (
        df["categorie"]
        .fillna("inconnu")
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"": "inconnu", "nan": "inconnu", "none": "inconnu"})
    )

    # ── Transactions par shop ──────────────────────────────────────
    transactions = (
        df.groupby("shop_url")["categorie"]
        .apply(lambda cats: sorted({
            c for c in cats.unique()
            if c not in ("inconnu", "nan", "none", "")
        }))
        .tolist()
    )
    transactions = [list(t) for t in transactions if len(t) >= 2]
    print(f"  [Association] {len(transactions)} shops multi-catégories")

    # ── Plan B : regrouper par product_type / marque ──────────────
    if len(transactions) < 5:
        print("  [Association] Trop peu de shops → regroupement par marque")
        transactions = _transactions_by_brand(df)

    # ── Plan C : transactions synthétiques par segment de prix ────
    if len(transactions) < 5:
        print("  [Association] Plan C → transactions par segment de prix")
        transactions = _transactions_by_price_segment(df)

    if len(transactions) < 3:
        print("  [Association] Données vraiment insuffisantes → skip.")
        return pd.DataFrame()

    # ── Limite le vocabulaire pour éviter MemoryError ─────────────
    transactions = _limit_vocabulary(transactions, max_items=50)

    print(f"  [Association] {len(transactions)} transactions | "
          f"vocab={len({i for t in transactions for i in t})} items")

    # ── Encodage binaire ───────────────────────────────────────────
    te       = TransactionEncoder()
    te_array = te.fit_transform(transactions)
    df_enc   = pd.DataFrame(te_array, columns=te.columns_)

    # ── Apriori avec support progressif ───────────────────────────
    frequent_items = pd.DataFrame()
    support_try    = min_support

    while frequent_items.empty and support_try <= 0.5:
        try:
            frequent_items = apriori(
                df_enc,
                min_support=support_try,
                use_colnames=True,
                max_len=2          # limité à paires pour éviter explosion mémoire
            )
        except MemoryError:
            print(f"  [Association] MemoryError à support={support_try} → augmentation")
            support_try = round(support_try * 2, 3)
            continue

        if frequent_items.empty:
            support_try = round(support_try * 1.5, 3)
            print(f"  [Association] Vide → support={support_try}")

    if frequent_items.empty:
        print("  [Association] Aucun ensemble fréquent trouvé → skip.")
        return pd.DataFrame()

    # ── Règles ────────────────────────────────────────────────────
    try:
        rules = association_rules(
            frequent_items,
            metric="confidence",
            min_threshold=min_confidence
        )
    except Exception as e:
        print(f"  [Association] Erreur règles : {e} → skip.")
        return pd.DataFrame()

    if rules.empty:
        print("  [Association] Aucune règle trouvée.")
        return pd.DataFrame()

    rules = rules[rules["lift"] >= min_lift].sort_values("lift", ascending=False)

    print(f"\n  [Association] {len(rules)} règles trouvées")

    if not rules.empty:
        display = rules[["antecedents", "consequents",
                          "support", "confidence", "lift"]].head(10).copy()
        display["antecedents"] = display["antecedents"].apply(
            lambda x: ", ".join(sorted(x)))
        display["consequents"] = display["consequents"].apply(
            lambda x: ", ".join(sorted(x)))
        for col in ["support", "confidence", "lift"]:
            display[col] = display[col].round(3)
        print(display.to_string(index=False))
        rules.to_csv("data/processed/association_rules.csv", index=False)
        print("  [Association] Sauvegardé : data/processed/association_rules.csv")

    return rules


# ── Helpers ────────────────────────────────────────────────────────

def _transactions_by_brand(df: pd.DataFrame) -> list:
    """Un groupe de marque = une transaction de catégories."""
    if "marque" not in df.columns:
        return []
    t = (
        df.groupby("marque")["categorie"]
        .apply(lambda c: sorted({
            x for x in c.unique()
            if str(x) not in ("inconnu", "nan", "")
        }))
        .tolist()
    )
    return [list(x) for x in t if len(x) >= 2]


def _transactions_by_price_segment(df: pd.DataFrame) -> list:
    """Regroupe les catégories par segment de prix — garanti non-vide."""
    if "prix" not in df.columns or df["prix"].sum() == 0:
        return []

    df = df.copy()
    df["_seg"] = pd.qcut(df["prix"], q=5, labels=False, duplicates="drop")
    t = (
        df.groupby("_seg")["categorie"]
        .apply(lambda c: sorted({
            x for x in c.unique()
            if str(x) not in ("inconnu", "nan", "")
        }))
        .tolist()
    )
    return [list(x) for x in t if len(x) >= 2]


def _limit_vocabulary(transactions: list, max_items: int = 50) -> list:
    """
    Garde seulement les max_items items les plus fréquents.
    Évite l'explosion mémoire d'Apriori sur un grand vocabulaire.
    """
    from collections import Counter
    freq  = Counter(item for t in transactions for item in t)
    top   = {item for item, _ in freq.most_common(max_items)}
    filtered = [
        [item for item in t if item in top]
        for t in transactions
    ]
    return [t for t in filtered if len(t) >= 2]
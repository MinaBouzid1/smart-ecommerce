import streamlit as st
import pandas as pd


def render_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Affiche les filtres dans la sidebar et retourne le DataFrame filtré.
    Compatible avec toutes les pages du dashboard.
    """
    with st.sidebar:
        st.markdown("## Filtres")

        # ── Source ────────────────────────────────────────────────
        sources = ["Toutes"] + sorted(df["source"].dropna().unique().tolist())
        source_sel = st.selectbox("Plateforme", sources)

        # ── Catégorie ─────────────────────────────────────────────
        cats = ["Toutes"] + sorted(df["categorie"].dropna().unique().tolist())
        cat_sel = st.selectbox("Catégorie", cats)

        # ── Prix ──────────────────────────────────────────────────
        prix_min = float(df["prix"].min())
        prix_max = float(df["prix"].max())
        if prix_min == prix_max:
            prix_max = prix_min + 1.0

        prix_range = st.slider(
            "Fourchette de prix (€)",
            min_value=prix_min,
            max_value=prix_max,
            value=(prix_min, prix_max),
            step=max(1.0, round((prix_max - prix_min) / 100, 1))
        )

        # ── Disponibilité ─────────────────────────────────────────
        dispo_only = st.checkbox("Produits disponibles uniquement", value=False)

        # ── En promo ──────────────────────────────────────────────
        promo_only = st.checkbox("En promotion uniquement", value=False)

        # ── Top produits ──────────────────────────────────────────
        top_only = st.checkbox("Top produits uniquement", value=False)

        st.markdown("---")
        st.markdown(
            f"<div style='color:#94a3b8;font-size:0.75rem;'>"
            f"Dataset : <b style='color:#6366f1'>{len(df):,}</b> produits</div>",
            unsafe_allow_html=True
        )

    # ── Application des filtres ────────────────────────────────────
    filtered = df.copy()

    if source_sel != "Toutes":
        filtered = filtered[filtered["source"] == source_sel]

    if cat_sel != "Toutes":
        filtered = filtered[filtered["categorie"] == cat_sel]

    filtered = filtered[
        (filtered["prix"] >= prix_range[0]) &
        (filtered["prix"] <= prix_range[1])
    ]

    if dispo_only and "disponible" in filtered.columns:
        filtered = filtered[filtered["disponible"] == True]

    if promo_only and "remise_pct" in filtered.columns:
        filtered = filtered[filtered["remise_pct"] > 0]

    if top_only and "top_produit" in filtered.columns:
        filtered = filtered[filtered["top_produit"] == 1]

    return filtered
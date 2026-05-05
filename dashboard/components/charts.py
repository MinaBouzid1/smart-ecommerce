import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Palette cohérente avec le thème sombre
PALETTE   = ["#6366f1", "#10b981", "#f59e0b", "#ef4444", "#06b6d4",
             "#8b5cf6", "#ec4899", "#14b8a6"]
PLOT_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,23,42,0.6)",
    font=dict(family="IBM Plex Sans", color="#94a3b8", size=12),
    margin=dict(l=20, r=20, t=40, b=20),
)


def chart_top_k_bar(df: pd.DataFrame, k: int = 20) -> go.Figure:
    """Barres horizontales des Top-K produits par score composite."""
    top = df.nlargest(k, "score_composite")[["nom", "score_composite", "categorie", "prix"]].copy()
    top["nom_court"] = top["nom"].str[:35] + "…"

    fig = px.bar(
        top, x="score_composite", y="nom_court",
        orientation="h",
        color="score_composite",
        color_continuous_scale=["#312e81", "#6366f1", "#a5b4fc"],
        hover_data={"categorie": True, "prix": True, "score_composite": ":.2f"},
        labels={"score_composite": "Score", "nom_court": ""},
        title=f"Top {k} produits — Score composite"
    )
    fig.update_layout(**PLOT_THEME)
    fig.update_coloraxes(showscale=False)
    fig.update_yaxes(autorange="reversed")
    return fig


def chart_price_distribution(df: pd.DataFrame) -> go.Figure:
    """Histogramme de distribution des prix avec KDE."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df["prix"].clip(upper=df["prix"].quantile(0.95)),
        nbinsx=50,
        name="Distribution prix",
        marker_color="#6366f1",
        opacity=0.75,
    ))
    fig.update_layout(
        **PLOT_THEME,
        title="Distribution des prix",
        xaxis_title="Prix (€)",
        yaxis_title="Nombre de produits",
        bargap=0.05
    )
    return fig


def chart_rating_vs_price(df: pd.DataFrame) -> go.Figure:
    """Scatter : Rating vs Prix coloré par segment."""
    plot_df = df.dropna(subset=["prix", "rating"]).copy()
    plot_df = plot_df[plot_df["prix"] > 0]

    seg_col = "segment" if "segment" in plot_df.columns else "categorie"

    fig = px.scatter(
        plot_df,
        x="prix",
        y="rating",
        color=seg_col,
        size="score_composite" if "score_composite" in plot_df.columns else None,
        size_max=18,
        opacity=0.7,
        hover_data=["nom", "prix", "rating", "nb_reviews"],
        color_discrete_sequence=PALETTE,
        title="Rating vs Prix par segment",
        labels={"prix": "Prix (€)", "rating": "Note moyenne"}
    )
    fig.update_layout(**PLOT_THEME)
    return fig


def chart_cluster_scatter(df: pd.DataFrame) -> go.Figure:
    """PCA 2D des clusters (utilise prix_norm et rating_norm comme proxy)."""
    if "prix_norm" not in df.columns or "remise_pct_norm" not in df.columns:
        return go.Figure()

    seg_col = "segment" if "segment" in df.columns else "cluster_kmeans"
    if seg_col not in df.columns:
        return go.Figure()

    fig = px.scatter(
        df.sample(min(1000, len(df)), random_state=42),
        x="prix_norm",
        y="remise_pct_norm",
        color=seg_col,
        opacity=0.6,
        color_discrete_sequence=PALETTE,
        hover_data=["nom", "prix", "remise_pct"],
        title="Segmentation des produits (espace normalisé)",
        labels={
            "prix_norm": "Prix normalisé",
            "remise_pct_norm": "Remise normalisée"
        }
    )
    fig.update_traces(marker=dict(size=6))
    fig.update_layout(**PLOT_THEME)
    return fig


def chart_category_treemap(df: pd.DataFrame) -> go.Figure:
    """Treemap des catégories par nombre de produits."""
    cat_counts = (
        df.groupby("categorie")
        .agg(nb_produits=("nom", "count"), prix_moyen=("prix", "mean"))
        .reset_index()
        .sort_values("nb_produits", ascending=False)
        .head(20)
    )
    fig = px.treemap(
        cat_counts,
        path=["categorie"],
        values="nb_produits",
        color="prix_moyen",
        color_continuous_scale=["#312e81", "#6366f1", "#a5b4fc"],
        title="Répartition par catégorie (taille = nb produits, couleur = prix moyen)"
    )
    fig.update_layout(**PLOT_THEME)
    return fig


def chart_availability_pie(df: pd.DataFrame) -> go.Figure:
    """Camembert disponibilité produits."""
    if "disponible" not in df.columns:
        return go.Figure()
    counts = df["disponible"].value_counts().reset_index()
    counts.columns = ["disponible", "count"]
    counts["label"] = counts["disponible"].map({True: "Disponible", False: "Rupture"})

    fig = px.pie(
        counts,
        names="label",
        values="count",
        color_discrete_sequence=["#10b981", "#ef4444"],
        title="Disponibilité des produits",
        hole=0.45
    )
    fig.update_layout(**PLOT_THEME)
    return fig


def chart_score_histogram(df: pd.DataFrame) -> go.Figure:
    """Histogramme des scores composites."""
    if "score_composite" not in df.columns:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df["score_composite"],
        nbinsx=40,
        marker_color="#10b981",
        opacity=0.8,
        name="Score composite"
    ))
    fig.update_layout(
        **PLOT_THEME,
        title="Distribution des scores composites",
        xaxis_title="Score composite (0-100)",
        yaxis_title="Nombre de produits"
    )
    return fig


def chart_promo_analysis(df: pd.DataFrame) -> go.Figure:
    """Box plot des remises par catégorie."""
    df_promo = df[df["remise_pct"] > 0].copy()
    if df_promo.empty:
        return go.Figure()

    top_cats = df_promo["categorie"].value_counts().head(8).index.tolist()
    df_promo = df_promo[df_promo["categorie"].isin(top_cats)]

    fig = px.box(
        df_promo,
        x="categorie",
        y="remise_pct",
        color="categorie",
        color_discrete_sequence=PALETTE,
        title="Distribution des remises par catégorie",
        labels={"remise_pct": "Remise (%)", "categorie": ""}
    )
    fig.update_layout(**PLOT_THEME)
    fig.update_layout(showlegend=False)
    return fig
"""
Smart eCommerce Intelligence — Dashboard Premium v3
Architecture : fichier unique, pas de pages multiples
Chatbot Groq intégré, zéro HTML div dans le contenu
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import os

# ── DOIT être la première commande Streamlit ────────────────────────
st.set_page_config(
    page_title="SmartCommerce Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Chemin racine projet ────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── CSS (encodage UTF-8 explicite — fix Windows) ────────────────────
css_path = Path(__file__).parent / "assets" / "style.css"
if css_path.exists():
    css_text = css_path.read_text(encoding="utf-8")
    st.markdown(f"<style>{css_text}</style>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  PLOTLY THEME UNIFIÉ
# ══════════════════════════════════════════════════════════════════════
PT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(19,28,46,0.5)",
    font=dict(family="DM Mono, monospace", color="#7a90b0", size=10),
    margin=dict(l=10, r=10, t=36, b=10),
    colorway=["#00d4ff","#8b5cf6","#10b981","#f59e0b","#ef4444","#06b6d4","#a78bfa"],
)
PAL = ["#00d4ff","#8b5cf6","#10b981","#f59e0b","#ef4444","#06b6d4","#a78bfa"]


# ══════════════════════════════════════════════════════════════════════
#  CHARGEMENT DES DONNÉES
# ══════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=300)
def load_data():
    base = ROOT / "data" / "processed"
    raw  = ROOT / "data" / "raw"

    # Priorité : LLM enrichi > enrichi > brut
    for p in [base/"products_llm_enriched.csv",
               base/"products_enriched.csv",
               raw/"products.csv"]:
        if p.exists():
            df = pd.read_csv(p)
            break
    else:
        # Données démo si rien n'existe
        np.random.seed(42)
        n = 200
        cats = ["Footwear","Apparel","Accessories","Socks","Care","Kids"]
        df = pd.DataFrame({
            "source":      ["shopify"] * n,
            "shop_url":    np.random.choice(["allbirds.com","gymshark.com"], n),
            "product_id":  [str(i) for i in range(n)],
            "nom":         [f"Produit Demo {i}" for i in range(n)],
            "description": ["Description exemple"] * n,
            "categorie":   np.random.choice(cats, n),
            "marque":      np.random.choice(["BrandA","BrandB","BrandC"], n),
            "prix":        np.random.uniform(15, 300, n).round(2),
            "remise_pct":  np.random.choice([0,0,0,10,15,20,25,30,40], n).astype(float),
            "rating":      np.random.uniform(3.5, 5.0, n).round(1),
            "nb_reviews":  np.random.randint(0, 2000, n),
            "stock":       np.random.randint(0, 150, n),
            "disponible":  np.random.choice([True, False], n, p=[0.8, 0.2]),
            "nb_images":   np.random.randint(1, 8, n),
            "nb_variants": np.random.randint(1, 5, n),
            "uid":         [f"shopify_{i}" for i in range(n)],
        })

    # Colonnes garanties
    num_cols = ["prix","remise_pct","rating","nb_reviews","stock","score_composite"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if "score_composite" not in df.columns or df["score_composite"].sum() == 0:
        px_n = (df["prix"] - df["prix"].min()) / (df["prix"].max() - df["prix"].min() + 1e-9)
        df["score_composite"] = (
            (1 - px_n) * 35 +
            (df.get("remise_pct", pd.Series(0, index=df.index)) /
             (df.get("remise_pct", pd.Series(1, index=df.index)).max() + 1e-9)) * 30 +
            (df.get("rating", pd.Series(0, index=df.index)) / 5) * 25 +
            (df.get("disponible", pd.Series(True, index=df.index)).astype(float)) * 10
        ).round(2)

    if "rang" not in df.columns:
        df["rang"] = df["score_composite"].rank(ascending=False, method="min").astype(int)

    if "top_produit" not in df.columns:
        seuil = df["score_composite"].quantile(0.70)
        df["top_produit"] = (df["score_composite"] >= seuil).astype(int)

    if "segment" not in df.columns:
        df["segment"] = pd.cut(
            df["score_composite"],
            bins=[0, 30, 50, 70, 100],
            labels=["Entrée gamme", "Standard", "Bestseller", "Premium"],
            include_lowest=True
        ).astype(str)

    if "ratio_qualite_prix" not in df.columns:
        df["ratio_qualite_prix"] = np.where(
            df["prix"] > 0,
            df.get("rating", pd.Series(4.0, index=df.index)) / np.log1p(df["prix"]),
            0
        ).round(4)

    if "en_promo" not in df.columns:
        df["en_promo"] = (df["remise_pct"] > 0).astype(int)

    if "prix_norm" not in df.columns:
        mn, mx = df["prix"].min(), df["prix"].max()
        df["prix_norm"] = ((df["prix"] - mn) / (mx - mn + 1e-9)).round(4)

    if "remise_pct_norm" not in df.columns:
        mn, mx = df["remise_pct"].min(), df["remise_pct"].max()
        df["remise_pct_norm"] = ((df["remise_pct"] - mn) / (mx - mn + 1e-9)).round(4)

    return df

df = load_data()


# ══════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:

    # Logo
    st.markdown("""
    <div style="padding:22px 16px 16px;border-bottom:1px solid rgba(0,212,255,0.07);margin-bottom:10px">
      <div style="font-family:'Syne',sans-serif;font-size:17px;font-weight:800;
                  background:linear-gradient(135deg,#00d4ff,#8b5cf6);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                  background-clip:text">◈ SmartCommerce</div>
      <div style="font-family:'DM Mono',monospace;font-size:9px;color:#3d5070;
                  letter-spacing:0.15em;text-transform:uppercase;margin-top:3px">
        Intelligence · Shopify ML</div>
    </div>
    """, unsafe_allow_html=True)

    # Navigation
    st.markdown(
        "<div style='font-family:DM Mono,monospace;font-size:9px;color:#3d5070;"
        "text-transform:uppercase;letter-spacing:0.18em;padding:4px 14px 8px'>"
        "Navigation</div>",
        unsafe_allow_html=True
    )

    page = st.radio(
        "nav",
        ["Vue Globale", "Top Produits", "Clustering", "Analyse Prix",
         "Chatbot IA", "Journal MCP", "Données"],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:rgba(0,212,255,0.07);margin:12px 0'/>",
                unsafe_allow_html=True)

    # Filtres
    st.markdown(
        "<div style='font-family:DM Mono,monospace;font-size:9px;color:#3d5070;"
        "text-transform:uppercase;letter-spacing:0.18em;padding:0 14px 8px'>"
        "Filtres</div>",
        unsafe_allow_html=True
    )

    src_opts = ["Toutes"] + sorted(df["source"].dropna().unique().tolist())
    src_sel  = st.selectbox("Plateforme", src_opts)

    cat_opts = ["Toutes"] + sorted(df["categorie"].dropna().unique().tolist())
    cat_sel  = st.selectbox("Catégorie", cat_opts)

    px_min = float(df["prix"].min())
    px_max = float(df["prix"].max())
    if px_min >= px_max:
        px_max = px_min + 1.0
    prix_range = st.slider(
        "Prix (€)",
        min_value=px_min, max_value=px_max,
        value=(px_min, px_max),
        step=max(0.5, round((px_max - px_min) / 100, 1))
    )

    promo_only = st.checkbox("En promotion uniquement")
    top_only   = st.checkbox("Top produits ML uniquement")

    st.markdown("<hr style='border-color:rgba(0,212,255,0.07);margin:12px 0'/>",
                unsafe_allow_html=True)

    # Stats live
    st.markdown(f"""
    <div style="padding:0 4px">
      <div style="display:flex;justify-content:space-between;align-items:center;
                  padding:5px 10px;margin-bottom:3px">
        <span style="font-family:DM Mono,monospace;font-size:9px;color:#3d5070;
                     text-transform:uppercase;letter-spacing:0.1em">Produits</span>
        <span style="font-family:DM Mono,monospace;font-size:11px;color:#00d4ff;
                     font-weight:500">{len(df):,}</span>
      </div>
      <div style="display:flex;justify-content:space-between;align-items:center;
                  padding:5px 10px;margin-bottom:3px">
        <span style="font-family:DM Mono,monospace;font-size:9px;color:#3d5070;
                     text-transform:uppercase;letter-spacing:0.1em">Catégories</span>
        <span style="font-family:DM Mono,monospace;font-size:11px;color:#00d4ff;
                     font-weight:500">{df['categorie'].nunique()}</span>
      </div>
      <div style="display:flex;justify-content:space-between;align-items:center;
                  padding:5px 10px;margin-bottom:3px">
        <span style="font-family:DM Mono,monospace;font-size:9px;color:#3d5070;
                     text-transform:uppercase;letter-spacing:0.1em">Top ML</span>
        <span style="font-family:DM Mono,monospace;font-size:11px;color:#10b981;
                     font-weight:500">{int(df['top_produit'].sum()):,}</span>
      </div>
      <div style="display:flex;align-items:center;gap:6px;padding:10px 10px 4px">
        <div style="width:6px;height:6px;border-radius:50%;background:#10b981;
                    box-shadow:0 0 6px #10b981;animation:none"></div>
        <span style="font-family:DM Mono,monospace;font-size:9px;color:#3d5070;
                     text-transform:uppercase;letter-spacing:0.1em">Pipeline actif</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  FILTRAGE
# ══════════════════════════════════════════════════════════════════════
df_f = df.copy()
if src_sel != "Toutes":
    df_f = df_f[df_f["source"] == src_sel]
if cat_sel != "Toutes":
    df_f = df_f[df_f["categorie"] == cat_sel]
df_f = df_f[
    (df_f["prix"] >= prix_range[0]) &
    (df_f["prix"] <= prix_range[1])
]
if promo_only:
    df_f = df_f[df_f["remise_pct"] > 0]
if top_only and "top_produit" in df_f.columns:
    df_f = df_f[df_f["top_produit"] == 1]
df_f = df_f.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════
#  UTILITAIRES
# ══════════════════════════════════════════════════════════════════════
def section_header(title: str, subtitle: str = "", badge: str = ""):
    badge_html = ""
    if badge:
        badge_html = (
            f"<span style='background:rgba(0,212,255,0.08);border:1px solid "
            f"rgba(0,212,255,0.18);color:#00d4ff;font-family:DM Mono,monospace;"
            f"font-size:9px;padding:3px 10px;border-radius:20px;letter-spacing:0.08em'>"
            f"{badge}</span>"
        )
    sub_html = (
        f"<div style='font-family:DM Mono,monospace;font-size:9px;color:#3d5070;"
        f"margin-top:4px;letter-spacing:0.08em'>{subtitle}</div>"
    ) if subtitle else ""
    st.markdown(f"""
    <div style="display:flex;align-items:flex-start;justify-content:space-between;
                margin-bottom:16px;padding-bottom:14px;
                border-bottom:1px solid rgba(0,212,255,0.07)">
      <div>
        <div style="font-family:Syne,sans-serif;font-size:20px;font-weight:800;
                    color:#eef2f8;letter-spacing:-0.01em">{title}</div>
        {sub_html}
      </div>
      {badge_html}
    </div>
    """, unsafe_allow_html=True)


def card(content_fn, title="", subtitle="", badge="", badge_color="cyan"):
    colors = {
        "cyan":   ("rgba(0,212,255,0.08)",   "rgba(0,212,255,0.18)",   "#00d4ff"),
        "green":  ("rgba(16,185,129,0.08)",  "rgba(16,185,129,0.18)",  "#10b981"),
        "violet": ("rgba(139,92,246,0.08)",  "rgba(139,92,246,0.18)",  "#8b5cf6"),
        "amber":  ("rgba(245,158,11,0.08)",  "rgba(245,158,11,0.18)",  "#f59e0b"),
    }
    bg, bdr, col = colors.get(badge_color, colors["cyan"])

    hdr = ""
    if title:
        badge_html = ""
        if badge:
            badge_html = (
                f"<span style='background:{bg};border:1px solid {bdr};"
                f"color:{col};font-family:DM Mono,monospace;font-size:9px;"
                f"padding:3px 10px;border-radius:20px'>{badge}</span>"
            )
        sub_html = (
            f"<div style='font-family:DM Mono,monospace;font-size:9px;"
            f"color:#3d5070;margin-top:3px'>{subtitle}</div>"
        ) if subtitle else ""
        hdr = f"""
        <div style="display:flex;align-items:flex-start;justify-content:space-between;
                    margin-bottom:14px">
          <div>
            <div style="font-family:Syne,sans-serif;font-size:13px;font-weight:700;
                        color:#eef2f8">{title}</div>
            {sub_html}
          </div>
          {badge_html}
        </div>"""

    st.markdown(f"""
    <div style="background:#131c2e;border:1px solid rgba(0,212,255,0.07);
                border-radius:12px;padding:20px 22px;
                transition:border-color 0.25s">
      {hdr}
    """, unsafe_allow_html=True)
    content_fn()
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  PAGE — VUE GLOBALE
# ══════════════════════════════════════════════════════════════════════
if page == "Vue Globale":
    section_header(
        "Vue Globale",
        f"{len(df_f):,} produits · {df_f['source'].nunique()} source(s) · "
        f"{df_f['categorie'].nunique()} catégories",
        "LIVE"
    )

    # ── KPI row ──────────────────────────────────────────────────────
    top_n    = int(df_f.get("top_produit", pd.Series(0, index=df_f.index)).sum())
    promo_n  = int((df_f["remise_pct"] > 0).sum())
    score_m  = float(df_f["score_composite"].mean())
    prix_m   = float(df_f["prix"].mean())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Produits analysés",  f"{len(df_f):,}",
              f"Dataset filtré")
    c2.metric("Top produits ML",    f"{top_n:,}",
              f"{top_n/max(len(df_f),1)*100:.1f}% du total")
    c3.metric("En promotion",       f"{promo_n:,}",
              f"Remise moy. {df_f[df_f['remise_pct']>0]['remise_pct'].mean():.1f}%"
              if promo_n > 0 else "Aucune promo")
    c4.metric("Prix moyen",         f"€{prix_m:.0f}",
              f"Médiane €{df_f['prix'].median():.0f}")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Charts row 1 ─────────────────────────────────────────────────
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown(
            "<div style='background:#131c2e;border:1px solid rgba(0,212,255,0.07);"
            "border-radius:12px;padding:20px 22px'>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<div style='font-family:Syne,sans-serif;font-size:13px;font-weight:700;"
            "color:#eef2f8;margin-bottom:4px'>Top 15 produits</div>"
            "<div style='font-family:DM Mono,monospace;font-size:9px;color:#3d5070;"
            "margin-bottom:14px'>Classement par score composite ML</div>",
            unsafe_allow_html=True
        )
        top15 = df_f.nlargest(15, "score_composite").copy()
        top15["nom_c"] = top15["nom"].str[:42]
        fig = go.Figure(go.Bar(
            y=top15["nom_c"],
            x=top15["score_composite"],
            orientation="h",
            marker=dict(
                color=top15["score_composite"],
                colorscale=[[0,"#1a2540"],[0.45,"#8b5cf6"],[1,"#00d4ff"]],
                line=dict(width=0),
            ),
            text=top15["score_composite"].round(1),
            textposition="outside",
            textfont=dict(family="DM Mono", size=9, color="#7a90b0"),
            hovertemplate="<b>%{y}</b><br>Score: %{x:.2f}<extra></extra>",
        ))
        fig.update_layout(**PT, height=310)
        fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
        fig.update_xaxes(gridcolor="rgba(255,255,255,0.04)", range=[0, top15["score_composite"].max()*1.15])
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        # Donut segments
        st.markdown(
            "<div style='background:#131c2e;border:1px solid rgba(0,212,255,0.07);"
            "border-radius:12px;padding:20px 22px;margin-bottom:14px'>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<div style='font-family:Syne,sans-serif;font-size:13px;font-weight:700;"
            "color:#eef2f8;margin-bottom:4px'>Segments ML</div>"
            "<div style='font-family:DM Mono,monospace;font-size:9px;color:#3d5070;"
            "margin-bottom:14px'>Distribution KMeans</div>",
            unsafe_allow_html=True
        )
        if "segment" in df_f.columns:
            sc = df_f["segment"].value_counts().reset_index()
            sc.columns = ["Segment","Count"]
            fig2 = go.Figure(go.Pie(
                labels=sc["Segment"], values=sc["Count"],
                hole=0.62,
                marker=dict(colors=PAL, line=dict(color="#07090f", width=2)),
                textinfo="none",
                hovertemplate="<b>%{label}</b><br>%{value} produits (%{percent})<extra></extra>",
            ))
            fig2.update_layout(**PT, height=180, showlegend=True,
                               legend=dict(font=dict(size=9, family="DM Mono"),
                                           x=1.0, y=0.5, xanchor="left",
                                           bgcolor="rgba(0,0,0,0)"))
            fig2.add_annotation(
                text=f"<b>{len(df_f):,}</b>",
                showarrow=False,
                font=dict(family="Syne", size=15, color="#eef2f8")
            )
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

        # Disponibilité
        st.markdown(
            "<div style='background:#131c2e;border:1px solid rgba(0,212,255,0.07);"
            "border-radius:12px;padding:20px 22px'>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<div style='font-family:Syne,sans-serif;font-size:13px;font-weight:700;"
            "color:#eef2f8;margin-bottom:14px'>Disponibilité</div>",
            unsafe_allow_html=True
        )
        if "disponible" in df_f.columns:
            dispo = df_f["disponible"].value_counts()
            n_ok  = int(dispo.get(True,  0))
            n_ko  = int(dispo.get(False, 0))
            pct   = n_ok / max(n_ok + n_ko, 1) * 100
            fig3 = go.Figure(go.Bar(
                x=["Disponible","Rupture"],
                y=[n_ok, n_ko],
                marker_color=["#10b981","#ef4444"],
                width=0.45,
                text=[f"{n_ok:,}", f"{n_ko:,}"],
                textposition="outside",
                textfont=dict(family="DM Mono", size=9, color="#7a90b0"),
            ))
            fig3.update_layout(**PT, height=140,
                               xaxis=dict(showgrid=False),
                               yaxis=dict(gridcolor="rgba(255,255,255,0.04)", showticklabels=False))
            st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Charts row 2 ─────────────────────────────────────────────────
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)

    with col3:
        st.markdown(
            "<div style='background:#131c2e;border:1px solid rgba(0,212,255,0.07);"
            "border-radius:12px;padding:20px 22px'>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<div style='font-family:Syne,sans-serif;font-size:13px;font-weight:700;"
            "color:#eef2f8;margin-bottom:14px'>Catégories — volume</div>",
            unsafe_allow_html=True
        )
        cat_df = (df_f.groupby("categorie")
                  .agg(n=("nom","count"), prix_moy=("prix","mean"))
                  .reset_index().nlargest(16,"n"))
        fig4 = px.treemap(
            cat_df, path=["categorie"], values="n", color="prix_moy",
            color_continuous_scale=["#0c1019","#8b5cf6","#00d4ff"],
            hover_data={"n": True, "prix_moy": ":.0f"}
        )
        fig4.update_layout(**PT, height=230)
        fig4.update_traces(
            textfont=dict(family="DM Mono", size=10),
            hovertemplate="<b>%{label}</b><br>Produits : %{value}<extra></extra>"
        )
        st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with col4:
        st.markdown(
            "<div style='background:#131c2e;border:1px solid rgba(0,212,255,0.07);"
            "border-radius:12px;padding:20px 22px'>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<div style='font-family:Syne,sans-serif;font-size:13px;font-weight:700;"
            "color:#eef2f8;margin-bottom:14px'>Distribution scores composites</div>",
            unsafe_allow_html=True
        )
        fig5 = go.Figure(go.Histogram(
            x=df_f["score_composite"], nbinsx=40,
            marker=dict(color="#00d4ff", opacity=0.65,
                        line=dict(color="#07090f", width=0.5)),
            hovertemplate="Score : %{x:.1f}<br>Produits : %{y}<extra></extra>",
        ))
        fig5.update_layout(**PT, height=230,
                           xaxis=dict(gridcolor="rgba(255,255,255,0.04)", title="Score"),
                           yaxis=dict(gridcolor="rgba(255,255,255,0.04)", title=""))
        st.plotly_chart(fig5, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  PAGE — TOP PRODUITS
# ══════════════════════════════════════════════════════════════════════
elif page == "Top Produits":
    section_header(
        "Top Produits",
        "Classement ML · XGBoost + Score composite pondéré",
        "Scoring actif"
    )

    c1, c2, c3 = st.columns([1, 2, 2])
    with c1:
        k = st.slider("Top K", 5, 100, 20, step=5)
    with c2:
        cat_f2 = ["Toutes"] + sorted(df_f["categorie"].dropna().unique().tolist())
        cat_2  = st.selectbox("Catégorie", cat_f2, key="tp_cat")
    with c3:
        sort_opts = [c for c in ["score_composite","prix","remise_pct","rating","nb_reviews"]
                     if c in df_f.columns]
        sort_by = st.selectbox("Trier par", sort_opts)

    df_view = df_f.copy()
    if cat_2 != "Toutes":
        df_view = df_view[df_view["categorie"] == cat_2]

    sort_col = sort_by if sort_by in df_view.columns else "score_composite"
    top = df_view.nlargest(k, sort_col)

    # Chart
    st.markdown(
        "<div style='background:#131c2e;border:1px solid rgba(0,212,255,0.07);"
        "border-radius:12px;padding:20px 22px;margin-bottom:14px'>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div style='font-family:Syne,sans-serif;font-size:13px;font-weight:700;"
        f"color:#eef2f8;margin-bottom:14px'>Top {k} — {sort_col}</div>",
        unsafe_allow_html=True
    )
    top_plot = top.head(25).copy()
    top_plot["nom_c"] = top_plot["nom"].str[:44]
    fig = go.Figure(go.Bar(
        y=top_plot["nom_c"],
        x=top_plot[sort_col],
        orientation="h",
        marker=dict(
            color=top_plot[sort_col],
            colorscale=[[0,"#1a2540"],[0.5,"#8b5cf6"],[1,"#00d4ff"]],
            line=dict(width=0),
        ),
        text=top_plot[sort_col].round(1),
        textposition="outside",
        textfont=dict(family="DM Mono", size=9, color="#7a90b0"),
        hovertemplate="<b>%{y}</b><br>%{x:.2f}<extra></extra>",
    ))
    fig.update_layout(**PT, height=min(40 * len(top_plot) + 60, 480))
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.04)")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

    # Tableau
    st.markdown(
        "<div style='background:#131c2e;border:1px solid rgba(0,212,255,0.07);"
        "border-radius:12px;padding:20px 22px'>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div style='font-family:Syne,sans-serif;font-size:13px;font-weight:700;"
        f"color:#eef2f8;margin-bottom:14px'>Détail · {len(top)} produits</div>",
        unsafe_allow_html=True
    )
    display_cols = [c for c in
                    ["rang","nom","categorie","source","prix","remise_pct",
                     "score_composite","rating","disponible"]
                    if c in top.columns]
    df_d = top[display_cols].copy()
    if "prix"             in df_d: df_d["prix"]             = df_d["prix"].map("€{:.2f}".format)
    if "remise_pct"       in df_d: df_d["remise_pct"]       = df_d["remise_pct"].map("{:.1f}%".format)
    if "score_composite"  in df_d: df_d["score_composite"]  = df_d["score_composite"].map("{:.2f}".format)
    if "rating"           in df_d: df_d["rating"]           = df_d["rating"].map("{:.1f}".format)
    rename = {"rang":"#","nom":"Produit","categorie":"Catégorie","source":"Source",
              "prix":"Prix","remise_pct":"Remise","score_composite":"Score",
              "rating":"Note","disponible":"Dispo"}
    df_d = df_d.rename(columns={k:v for k,v in rename.items() if k in df_d.columns})
    st.dataframe(df_d.reset_index(drop=True), use_container_width=True, height=380)

    csv = top.to_csv(index=False).encode("utf-8-sig")
    st.download_button(f"↓ Exporter Top-{k} (CSV)", csv,
                       f"top_{k}_produits.csv", "text/csv")
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  PAGE — CLUSTERING
# ══════════════════════════════════════════════════════════════════════
elif page == "Clustering":
    section_header(
        "Analyse des Segments",
        "KMeans · DBSCAN anomalies · visualisation PCA",
        "4 clusters"
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(
            "<div style='background:#131c2e;border:1px solid rgba(0,212,255,0.07);"
            "border-radius:12px;padding:20px 22px'>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<div style='font-family:Syne,sans-serif;font-size:13px;font-weight:700;"
            "color:#eef2f8;margin-bottom:14px'>Segments détectés</div>",
            unsafe_allow_html=True
        )
        if "segment" in df_f.columns:
            seg_colors = {
                "Premium":      "#8b5cf6",
                "Bestseller":   "#00d4ff",
                "Standard":     "#10b981",
                "Entrée gamme": "#f59e0b",
            }
            segs = df_f["segment"].value_counts()
            for seg, count in segs.items():
                pct   = count / max(len(df_f), 1) * 100
                color = seg_colors.get(str(seg), "#7a90b0")
                st.markdown(f"""
                <div style="background:#1a2540;border:1px solid rgba(0,212,255,0.07);
                            border-radius:8px;padding:12px 14px;margin-bottom:8px">
                  <div style="display:flex;justify-content:space-between;
                              align-items:center;margin-bottom:6px">
                    <span style="font-family:Syne,sans-serif;font-size:12px;
                                 font-weight:700;color:#eef2f8">{seg}</span>
                    <span style="font-family:DM Mono,monospace;font-size:10px;
                                 color:{color}">{pct:.1f}%</span>
                  </div>
                  <div style="height:3px;background:#07090f;border-radius:2px;overflow:hidden">
                    <div style="height:100%;width:{pct:.1f}%;background:{color};
                                border-radius:2px"></div>
                  </div>
                  <div style="font-family:DM Mono,monospace;font-size:9px;
                              color:#3d5070;margin-top:5px">{count:,} produits</div>
                </div>
                """, unsafe_allow_html=True)

            # Stats par segment
            st.markdown(
                "<div style='font-family:Syne,sans-serif;font-size:12px;font-weight:700;"
                "color:#eef2f8;margin:12px 0 8px'>Stats par segment</div>",
                unsafe_allow_html=True
            )
            seg_stats = df_f.groupby("segment")[
                [c for c in ["prix","score_composite","remise_pct"] if c in df_f.columns]
            ].mean().round(2)
            st.dataframe(seg_stats, use_container_width=True, height=160)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        # Scatter segmentation
        st.markdown(
            "<div style='background:#131c2e;border:1px solid rgba(0,212,255,0.07);"
            "border-radius:12px;padding:20px 22px;margin-bottom:14px'>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<div style='font-family:Syne,sans-serif;font-size:13px;font-weight:700;"
            "color:#eef2f8;margin-bottom:14px'>Espace de segmentation (normalisé)</div>",
            unsafe_allow_html=True
        )
        sample = df_f.sample(min(900, len(df_f)), random_state=42)
        color_col = "segment" if "segment" in sample.columns else "categorie"
        fig_sc = px.scatter(
            sample, x="prix_norm", y="remise_pct_norm",
            color=color_col, opacity=0.6,
            color_discrete_sequence=PAL,
            hover_data={"nom": True, "prix": True, "remise_pct": True},
            labels={"prix_norm":"Prix normalisé", "remise_pct_norm":"Remise normalisée"},
        )
        fig_sc.update_traces(marker=dict(size=5))
        fig_sc.update_layout(**PT, height=250)
        st.plotly_chart(fig_sc, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

        # Prix vs score
        st.markdown(
            "<div style='background:#131c2e;border:1px solid rgba(0,212,255,0.07);"
            "border-radius:12px;padding:20px 22px'>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<div style='font-family:Syne,sans-serif;font-size:13px;font-weight:700;"
            "color:#eef2f8;margin-bottom:14px'>Prix vs Score composite</div>",
            unsafe_allow_html=True
        )
        fig_pv = px.scatter(
            df_f.sample(min(600, len(df_f)), random_state=1),
            x="prix", y="score_composite",
            color=color_col, opacity=0.6,
            color_discrete_sequence=PAL,
            hover_data={"nom": True, "prix": True},
            labels={"prix":"Prix (€)", "score_composite":"Score ML"},
        )
        fig_pv.update_traces(marker=dict(size=5))
        fig_pv.update_layout(**PT, height=220)
        st.plotly_chart(fig_pv, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  PAGE — ANALYSE PRIX
# ══════════════════════════════════════════════════════════════════════
elif page == "Analyse Prix":
    section_header(
        "Analyse des Prix",
        "Distributions · remises · positionnement concurrentiel"
    )

    # KPIs prix
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prix minimum", f"€{df_f['prix'].min():.2f}")
    c2.metric("Prix médian",  f"€{df_f['prix'].median():.2f}")
    c3.metric("Prix moyen",   f"€{df_f['prix'].mean():.2f}")
    c4.metric("Prix maximum", f"€{df_f['prix'].max():.2f}")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            "<div style='background:#131c2e;border:1px solid rgba(0,212,255,0.07);"
            "border-radius:12px;padding:20px 22px'>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<div style='font-family:Syne,sans-serif;font-size:13px;font-weight:700;"
            "color:#eef2f8;margin-bottom:14px'>Distribution des prix</div>",
            unsafe_allow_html=True
        )
        fig_h = go.Figure(go.Histogram(
            x=df_f["prix"].clip(upper=df_f["prix"].quantile(0.95)),
            nbinsx=40,
            marker=dict(color="#00d4ff", opacity=0.72,
                        line=dict(color="#07090f", width=0.5)),
            hovertemplate="Prix : %{x:.0f}€<br>Produits : %{y}<extra></extra>",
        ))
        fig_h.update_layout(**PT, height=230,
                            xaxis=dict(gridcolor="rgba(255,255,255,0.04)", title="Prix (€)"),
                            yaxis=dict(gridcolor="rgba(255,255,255,0.04)"))
        st.plotly_chart(fig_h, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(
            "<div style='background:#131c2e;border:1px solid rgba(0,212,255,0.07);"
            "border-radius:12px;padding:20px 22px'>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<div style='font-family:Syne,sans-serif;font-size:13px;font-weight:700;"
            "color:#eef2f8;margin-bottom:14px'>Remises par catégorie</div>",
            unsafe_allow_html=True
        )
        df_promo = df_f[df_f["remise_pct"] > 0].copy()
        if not df_promo.empty:
            top_cats  = df_promo["categorie"].value_counts().head(7).index
            df_promo2 = df_promo[df_promo["categorie"].isin(top_cats)]
            fig_b = px.box(df_promo2, x="categorie", y="remise_pct",
                           color="categorie", color_discrete_sequence=PAL,
                           labels={"remise_pct":"Remise (%)","categorie":""})
            fig_b.update_layout(**PT, height=230, showlegend=False,
                                xaxis=dict(tickangle=-30))
            st.plotly_chart(fig_b, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Aucun produit en promotion dans la sélection.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Scatter rating vs prix
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='background:#131c2e;border:1px solid rgba(0,212,255,0.07);"
        "border-radius:12px;padding:20px 22px'>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<div style='font-family:Syne,sans-serif;font-size:13px;font-weight:700;"
        "color:#eef2f8;margin-bottom:14px'>Rating vs Prix · par segment</div>",
        unsafe_allow_html=True
    )
    color_c = "segment" if "segment" in df_f.columns else "categorie"
    sz_c    = "score_composite" if "score_composite" in df_f.columns else None
    fig_rv  = px.scatter(
        df_f.sample(min(800, len(df_f)), random_state=42),
        x="prix", y="rating", color=color_c,
        size=sz_c, size_max=14, opacity=0.65,
        color_discrete_sequence=PAL,
        hover_data={"nom": True, "prix": True, "rating": True},
        labels={"prix":"Prix (€)","rating":"Note"}
    )
    fig_rv.update_layout(**PT, height=300)
    st.plotly_chart(fig_rv, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

    # Top rapport qualité/prix
    if "ratio_qualite_prix" in df_f.columns:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='background:#131c2e;border:1px solid rgba(0,212,255,0.07);"
            "border-radius:12px;padding:20px 22px'>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<div style='font-family:Syne,sans-serif;font-size:13px;font-weight:700;"
            "color:#eef2f8;margin-bottom:14px'>Top 10 — Meilleur rapport qualité/prix</div>",
            unsafe_allow_html=True
        )
        qp = df_f.nlargest(10,"ratio_qualite_prix")[
            [c for c in ["nom","categorie","prix","rating","remise_pct","ratio_qualite_prix"]
             if c in df_f.columns]
        ].copy()
        if "prix"               in qp: qp["prix"]               = qp["prix"].map("€{:.2f}".format)
        if "remise_pct"         in qp: qp["remise_pct"]         = qp["remise_pct"].map("{:.1f}%".format)
        if "ratio_qualite_prix" in qp: qp["ratio_qualite_prix"] = qp["ratio_qualite_prix"].map("{:.4f}".format)
        if "rating"             in qp: qp["rating"]             = qp["rating"].map("{:.1f}".format)
        st.dataframe(qp.reset_index(drop=True), use_container_width=True, height=280)
        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  PAGE — CHATBOT IA
# ══════════════════════════════════════════════════════════════════════
elif page == "Chatbot IA":
    section_header(
        "Chatbot Intelligence",
        "Interroge le dataset en langage naturel · Groq LLaMA",
        "LLM actif"
    )

    # Initialisation session
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chatbot_ready" not in st.session_state:
        st.session_state.chatbot_ready = False
    if "chatbot_obj" not in st.session_state:
        st.session_state.chatbot_obj = None

    # Chargement du chatbot
    if not st.session_state.chatbot_ready:
        try:
            from llm.chatbot import EcommerceChatbot
            st.session_state.chatbot_obj   = EcommerceChatbot(df_f)
            st.session_state.chatbot_ready = True
        except EnvironmentError as e:
            st.markdown(
                "<div style='background:#131c2e;border:1px solid rgba(245,158,11,0.2);"
                "border-radius:12px;padding:20px 22px'>",
                unsafe_allow_html=True
            )
            st.warning(
                "**Clé Groq non configurée.**\n\n"
                "Crée un fichier `.env` à la racine du projet :\n"
                "```\nGROQ_API_KEY=gsk_ta_cle_ici\nLLM_MODEL=llama-3.1-8b-instant\n```\n\n"
                "Obtiens une clé gratuite sur [console.groq.com](https://console.groq.com/keys)"
            )
            st.markdown("</div>", unsafe_allow_html=True)
            st.stop()
        except ImportError:
            st.error("Module LLM non trouvé. Vérifie que `llm/chatbot.py` existe.")
            st.stop()
        except Exception as e:
            st.error(f"Erreur initialisation chatbot : {e}")
            st.stop()

    # Questions suggérées
    st.markdown(
        "<div style='font-family:DM Mono,monospace;font-size:9px;color:#3d5070;"
        "text-transform:uppercase;letter-spacing:0.15em;margin-bottom:10px'>"
        "Questions suggérées</div>",
        unsafe_allow_html=True
    )

    suggestions = [
        "Quels sont les 5 meilleurs produits ?",
        "Quelles tendances vois-tu dans ce dataset ?",
        "Quels produits ont le meilleur rapport qualité/prix ?",
        "Analyse les produits en promotion",
        "Quelle stratégie marketing recommandes-tu ?",
        "Compare les différents segments de produits",
    ]

    cols_s = st.columns(3)
    for i, q in enumerate(suggestions):
        if cols_s[i % 3].button(q, key=f"sug_{i}"):
            st.session_state.messages.append({"role":"user","content":q})
            with st.spinner("Analyse en cours..."):
                try:
                    resp = st.session_state.chatbot_obj.chat(q)
                except Exception as e:
                    resp = f"Erreur : {e}"
            st.session_state.messages.append({"role":"assistant","content":resp})
            st.rerun()

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # Historique
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    if prompt := st.chat_input("Pose ta question sur le dataset eCommerce…"):
        st.session_state.messages.append({"role":"user","content":prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Groq LLaMA analyse..."):
                try:
                    response = st.session_state.chatbot_obj.chat(prompt)
                except Exception as e:
                    response = f"Erreur : {e}"
            st.markdown(response)
        st.session_state.messages.append({"role":"assistant","content":response})

    # Reset
    if st.session_state.messages:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("↺ Réinitialiser la conversation"):
            st.session_state.messages = []
            if st.session_state.chatbot_obj:
                st.session_state.chatbot_obj.reset()
            st.rerun()


# ══════════════════════════════════════════════════════════════════════
#  PAGE — JOURNAL MCP
# ══════════════════════════════════════════════════════════════════════
elif page == "Journal MCP":
    section_header(
        "Journal MCP",
        "Audit Log · Actions agents · Permissions · Traçabilité",
        "Audit actif"
    )

    log_path = ROOT / "data" / "logs" / "mcp_audit.jsonl"
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            "<div style='background:#131c2e;border:1px solid rgba(0,212,255,0.07);"
            "border-radius:12px;padding:20px 22px'>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<div style='font-family:Syne,sans-serif;font-size:13px;font-weight:700;"
            "color:#eef2f8;margin-bottom:14px'>Journal des actions (15 dernières)</div>",
            unsafe_allow_html=True
        )

        if log_path.exists():
            import json
            entries = []
            with open(log_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try: entries.append(json.loads(line))
                        except: pass

            if entries:
                icons = {
                    "scraping-server": ("⬡","rgba(0,212,255,0.1)","rgba(0,212,255,0.2)"),
                    "ml-server":       ("◈","rgba(139,92,246,0.1)","rgba(139,92,246,0.2)"),
                    "llm-server":      ("✦","rgba(16,185,129,0.1)","rgba(16,185,129,0.2)"),
                }
                sc = {"SUCCESS":"#10b981","FAILURE":"#ef4444","DENIED":"#f59e0b","WARNING":"#f59e0b"}

                for e in reversed(entries[-15:]):
                    ico, bg, bdr = icons.get(e.get("server_id",""), ("○","rgba(255,255,255,0.05)","rgba(255,255,255,0.1)"))
                    status_c = sc.get(e.get("status",""), "#7a90b0")
                    st.markdown(f"""
                    <div style="display:flex;gap:12px;padding:10px 0;
                                border-bottom:1px solid rgba(255,255,255,0.03)">
                      <div style="width:30px;height:30px;border-radius:8px;
                                  background:{bg};border:1px solid {bdr};
                                  display:flex;align-items:center;justify-content:center;
                                  font-size:12px;flex-shrink:0">{ico}</div>
                      <div style="flex:1">
                        <div style="font-size:11px;color:#7a90b0;line-height:1.5">
                          <span style="color:#eef2f8;font-weight:500">{e.get('server_id','')}</span> ·
                          {e.get('tool_name','')} ·
                          <span style="color:{status_c}">{e.get('status','')}</span> ·
                          {e.get('action','')[:55]}
                        </div>
                        <div style="display:flex;gap:16px;margin-top:2px">
                          <span style="font-family:DM Mono,monospace;font-size:9px;color:#3d5070">
                            {e.get('timestamp','')[:19]}
                          </span>
                          <span style="font-family:DM Mono,monospace;font-size:9px;color:#3d5070">
                            {e.get('duration_ms',0):.0f}ms
                          </span>
                        </div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Graphe statuts
                st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
                from collections import Counter
                statuses = Counter(e.get("status","") for e in entries)
                sc_colors = {"SUCCESS":"#10b981","FAILURE":"#ef4444","DENIED":"#f59e0b","WARNING":"#f59e0b"}
                fig_s = go.Figure(go.Bar(
                    x=list(statuses.keys()), y=list(statuses.values()),
                    marker_color=[sc_colors.get(s,"#7a90b0") for s in statuses],
                    width=0.4,
                    text=list(statuses.values()),
                    textposition="outside",
                    textfont=dict(family="DM Mono", size=9, color="#7a90b0"),
                ))
                fig_s.update_layout(**PT, height=150,
                                    xaxis=dict(showgrid=False),
                                    yaxis=dict(showticklabels=False, showgrid=False))
                st.plotly_chart(fig_s, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("Log vide. Lance : `python -m mcp.host`")
        else:
            st.info("Fichier log introuvable. Lance d'abord : `python -m mcp.host`")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        # Serveurs
        st.markdown(
            "<div style='background:#131c2e;border:1px solid rgba(0,212,255,0.07);"
            "border-radius:12px;padding:20px 22px;margin-bottom:14px'>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<div style='font-family:Syne,sans-serif;font-size:13px;font-weight:700;"
            "color:#eef2f8;margin-bottom:14px'>Serveurs MCP</div>",
            unsafe_allow_html=True
        )
        servers = [
            ("⬡","scraping-server","Shopify API","#00d4ff",True),
            ("◈","ml-server","Pipeline ML","#8b5cf6",True),
            ("✦","llm-server","Groq LLaMA","#10b981",True),
        ]
        for ico, sid, desc, col, ok in servers:
            status_txt = "✓ APPROUVÉ" if ok else "✗ REFUSÉ"
            status_col = "#10b981"   if ok else "#ef4444"
            st.markdown(f"""
            <div style="background:#1a2540;border:1px solid rgba(0,212,255,0.07);
                        border-radius:8px;padding:12px 14px;margin-bottom:8px">
              <div style="display:flex;justify-content:space-between;align-items:center;
                          margin-bottom:4px">
                <span style="font-family:Syne,sans-serif;font-size:12px;font-weight:700;
                             color:{col}">{ico} {sid}</span>
                <span style="font-family:DM Mono,monospace;font-size:9px;
                             color:{status_col}">{status_txt}</span>
              </div>
              <span style="font-family:DM Mono,monospace;font-size:9px;color:#3d5070">{desc}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Principes MCP
        st.markdown(
            "<div style='background:#131c2e;border:1px solid rgba(0,212,255,0.07);"
            "border-radius:12px;padding:20px 22px'>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<div style='font-family:Syne,sans-serif;font-size:13px;font-weight:700;"
            "color:#eef2f8;margin-bottom:14px'>Principes MCP</div>",
            unsafe_allow_html=True
        )
        principes = [
            ("◆","Moindre privilège"),
            ("◆","Déclaration d'intention"),
            ("◆","Isolation des serveurs"),
            ("◆","Traçabilité totale"),
            ("◆","Approbation manuelle"),
        ]
        for sym, txt in principes:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:8px;padding:6px 0;
                        border-bottom:1px solid rgba(255,255,255,0.03)">
              <span style="color:#00d4ff;font-size:8px">{sym}</span>
              <span style="font-size:11px;color:#7a90b0">{txt}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  PAGE — DONNÉES BRUTES
# ══════════════════════════════════════════════════════════════════════
elif page == "Données":
    section_header(
        "Données Brutes",
        f"Exploration complète · {len(df_f):,} produits après filtres"
    )

    search = st.text_input("", placeholder="Rechercher un produit par nom…",
                           label_visibility="collapsed")
    if search:
        df_f = df_f[df_f["nom"].str.contains(search, case=False, na=False)]
        st.markdown(
            f"<div style='font-family:DM Mono,monospace;font-size:10px;color:#3d5070;"
            f"margin-bottom:8px'>{len(df_f)} résultats pour « {search} »</div>",
            unsafe_allow_html=True
        )

    all_cols = df_f.columns.tolist()
    default  = [c for c in ["nom","categorie","source","prix","remise_pct",
                             "score_composite","rating","disponible","rang"]
                if c in all_cols]
    selected = st.multiselect("Colonnes à afficher", all_cols, default=default)

    st.markdown(
        "<div style='background:#131c2e;border:1px solid rgba(0,212,255,0.07);"
        "border-radius:12px;padding:20px 22px'>",
        unsafe_allow_html=True
    )
    if selected:
        st.dataframe(df_f[selected].reset_index(drop=True),
                     use_container_width=True, height=480)
    else:
        st.dataframe(df_f.reset_index(drop=True),
                     use_container_width=True, height=480)

    col_e1, col_e2 = st.columns([1, 3])
    with col_e1:
        csv = df_f.to_csv(index=False).encode("utf-8-sig")
        st.download_button("↓ Exporter CSV", csv,
                           "ecommerce_export.csv","text/csv")
    with col_e2:
        st.markdown(
            f"<div style='font-family:DM Mono,monospace;font-size:10px;color:#3d5070;"
            f"padding-top:10px'>"
            f"Lignes <span style='color:#00d4ff'>{len(df_f):,}</span> · "
            f"Colonnes <span style='color:#00d4ff'>{len(df_f.columns)}</span> · "
            f"Taille <span style='color:#00d4ff'>"
            f"{df_f.memory_usage(deep=True).sum()/1024:.0f} KB</span></div>",
            unsafe_allow_html=True
        )
    st.markdown("</div>", unsafe_allow_html=True)
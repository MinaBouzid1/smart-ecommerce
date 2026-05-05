"""
Prompts optimisés pour les modèles LLaMA via Groq.
"""
from langchain_core.prompts import ChatPromptTemplate


# ══════════════════════════════════════════════════════════════════
#  RÉSUMÉ PRODUIT
# ══════════════════════════════════════════════════════════════════

PRODUCT_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Tu es un expert e-commerce. "
     "Résume les descriptions produits en 2-3 phrases claires en français. "
     "Sois concis et mets en avant les points forts du produit."),
    ("human",
     "Produit : {nom}\n"
     "Catégorie : {categorie}\n"
     "Prix : {prix}€\n"
     "Description : {description}\n\n"
     "Résumé commercial (2-3 phrases) :")
])


# ══════════════════════════════════════════════════════════════════
#  ANALYSE TENDANCES
# ══════════════════════════════════════════════════════════════════

TREND_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Tu es un analyste e-commerce senior. "
     "Tu analyses des données de marché Shopify et produis des rapports structurés. "
     "Réponds toujours en français avec des sections numérotées claires."),
    ("human",
     """Voici les données de notre dataset Shopify :

DONNÉES GÉNÉRALES :
- Produits analysés : {nb_produits}
- Boutiques : {nb_shops}
- Catégories : {nb_categories}

PRIX :
- Moyen : {prix_moyen}€  |  Médian : {prix_median}€
- Min : {prix_min}€  |  Max : {prix_max}€

PROMOTIONS :
- En promo : {pct_promo}%  |  Remise moyenne : {remise_moyenne}%

TOP 5 CATÉGORIES :
{top_categories}

TOP 10 PRODUITS :
{top_produits}

SEGMENTS ML :
{segments}

Génère un rapport avec ces sections :
1. Résumé exécutif (3-4 phrases)
2. Tendances principales (3-5 points)
3. Opportunités de marché
4. Risques détectés
5. Recommandations stratégiques (3-5 actions)""")
])


# ══════════════════════════════════════════════════════════════════
#  RECOMMANDATIONS MARKETING
# ══════════════════════════════════════════════════════════════════

MARKETING_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Tu es un expert en marketing digital e-commerce Shopify. "
     "Donne des recommandations concrètes et actionnables en français."),
    ("human",
     """Produit à analyser :
- Nom : {nom}
- Catégorie : {categorie}
- Prix : {prix}€  |  Remise : {remise_pct}%
- Score ML : {score_composite}/100
- Segment : {segment}
- Stock : {stock} unités
- En promotion : {en_promo}

Fournis en français :
1. Positionnement (1 phrase)
2. Audience cible
3. 3 axes de communication
4. Stratégie de prix
5. Canal marketing recommandé
6. 1 idée promotion concrète""")
])


# ══════════════════════════════════════════════════════════════════
#  CHATBOT
# ══════════════════════════════════════════════════════════════════

CHATBOT_SYSTEM = (
    "Tu es un assistant expert en analyse e-commerce Shopify. "
    "Tu as accès à un dataset de {nb_produits} produits analysés par ML.\n\n"
    "CONTEXTE :\n"
    "- Sources : {sources}\n"
    "- Catégories : {categories}\n"
    "- Prix : {prix_min}€ → {prix_max}€\n"
    "- Produits en promo : {pct_promo}%\n"
    "- Score moyen : {score_moyen}/100\n\n"
    "TOP 5 PRODUITS :\n{top5}\n\n"
    "Réponds en français. Sois précis et utilise les données chiffrées. "
    "Maximum 5 phrases sauf si plus de détails sont demandés."
)

CHATBOT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", CHATBOT_SYSTEM),
    ("placeholder", "{chat_history}"),
    ("human", "{question}"),
])


# ══════════════════════════════════════════════════════════════════
#  PROFIL CLIENT
# ══════════════════════════════════════════════════════════════════

CLIENT_PROFILE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Tu es expert en segmentation client e-commerce. "
     "Produis des profils clients actionnables en français."),
    ("human",
     """Segment : {segment}
Prix moyen : {prix_moyen}€  |  Remise moyenne : {remise_moyenne}%
Score moyen : {score_moyen}/100

Produits phares :
{produits}

Génère le profil client :
1. Profil démographique probable
2. Motivations d'achat
3. Sensibilité au prix
4. Canaux digitaux préférés
5. Message marketing adapté""")
])
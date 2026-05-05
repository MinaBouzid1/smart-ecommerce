"""
Chatbot intelligent pour interroger le dataset eCommerce en langage naturel.
Intégré dans le dashboard Streamlit.
"""
import pandas as pd
import logging
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

from llm.config import get_llm
from llm.prompts import CHATBOT_PROMPT

logger = logging.getLogger("LLM.Chatbot")


class EcommerceChatbot:
    """
    Chatbot avec mémoire de conversation pour interroger le dataset.

    Usage :
        bot = EcommerceChatbot(df)
        reponse = bot.chat("Quels sont les produits les moins chers ?")
        reponse = bot.chat("Et parmi eux, lesquels sont en promo ?")
    """

    def __init__(self, df: pd.DataFrame, max_history: int = 6):
        self.df          = df
        self.max_history = max_history
        self.history: list = []
        self.llm         = get_llm(temperature=0.4, max_tokens=800)
        self.chain       = CHATBOT_PROMPT | self.llm | StrOutputParser()
        self._context    = self._build_context()

    def _build_context(self) -> dict:
        """Prépare le contexte statique du dataset pour le LLM."""
        score_col = "score_composite" if "score_composite" in self.df.columns else "prix"

        top5 = self.df.nlargest(5, score_col)
        top5_str = "\n".join(
            f"  {i+1}. {row['nom'][:40]} — {row['prix']:.0f}€"
            f" — score: {row.get(score_col, 0):.2f}"
            for i, (_, row) in enumerate(top5.iterrows())
        )

        cats = self.df["categorie"].dropna().unique().tolist()
        cats_str = ", ".join(cats[:15]) + ("..." if len(cats) > 15 else "")

        sources = self.df["source"].dropna().unique().tolist()

        return {
            "nb_produits":  len(self.df),
            "sources":      ", ".join(sources),
            "categories":   cats_str,
            "prix_min":     f"{self.df['prix'].min():.2f}",
            "prix_max":     f"{self.df['prix'].max():.2f}",
            "pct_promo":    f"{(self.df['remise_pct'] > 0).mean() * 100:.1f}",
            "score_moyen":  f"{self.df[score_col].mean():.2f}",
            "top5":         top5_str,
        }

    def chat(self, question: str) -> str:
        """
        Envoie une question au chatbot et retourne la réponse.
        Maintient l'historique de conversation (mémoire glissante).
        """
        if not question.strip():
            return "Pose-moi une question sur le dataset !"

        # Enrichissement automatique de la question avec données réelles
        enriched = self._enrich_question(question)

        try:
            response = self.chain.invoke({
                **self._context,
                "chat_history": self.history,
                "question":     enriched,
            })
        except Exception as e:
            logger.error(f"Erreur chatbot : {e}")
            return f"Désolé, une erreur est survenue : {e}"

        # Mise à jour de l'historique (mémoire glissante)
        self.history.append(HumanMessage(content=question))
        self.history.append(AIMessage(content=response))

        # Fenêtre glissante : garde max_history derniers échanges
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-(self.max_history * 2):]

        return response

    def _enrich_question(self, question: str) -> str:
        """
        Enrichit la question avec des données réelles si pertinent.
        Permet au LLM de répondre avec des chiffres précis.
        """
        q_lower = question.lower()
        extra   = ""

        # Question sur une catégorie spécifique
        for cat in self.df["categorie"].dropna().unique():
            if str(cat).lower() in q_lower:
                cat_df = self.df[self.df["categorie"] == cat]
                extra += (
                    f"\n[Données réelles pour '{cat}': "
                    f"{len(cat_df)} produits, "
                    f"prix moyen {cat_df['prix'].mean():.0f}€, "
                    f"remise moyenne {cat_df['remise_pct'].mean():.1f}%]"
                )
                break

        # Question sur les promotions
        if any(w in q_lower for w in ["promo", "remise", "réduction", "solde"]):
            promo_df = self.df[self.df["remise_pct"] > 0]
            if not promo_df.empty:
                best = promo_df.nlargest(3, "remise_pct")
                extra += (
                    f"\n[Top 3 remises: "
                    + ", ".join(
                        f"{row['nom'][:25]} ({row['remise_pct']:.0f}%)"
                        for _, row in best.iterrows()
                    ) + "]"
                )

        # Question sur les prix
        if any(w in q_lower for w in ["moins cher", "pas cher", "économique", "budget"]):
            cheap = self.df.nsmallest(3, "prix")
            extra += (
                f"\n[Produits les moins chers: "
                + ", ".join(
                    f"{row['nom'][:25]} ({row['prix']:.0f}€)"
                    for _, row in cheap.iterrows()
                ) + "]"
            )

        return question + extra

    def reset(self):
        """Réinitialise l'historique de conversation."""
        self.history = []
        logger.info("Historique chatbot réinitialisé")

    def get_suggested_questions(self) -> list[str]:
        """Retourne des questions suggérées basées sur le dataset."""
        cats = self.df["categorie"].dropna().unique()[:3]
        questions = [
            "Quels sont les 5 meilleurs produits du dataset ?",
            "Quelles sont les tendances principales détectées ?",
            f"Analyse la catégorie '{cats[0]}'" if len(cats) > 0 else "",
            "Quels produits sont en promotion en ce moment ?",
            "Quelle stratégie marketing recommandes-tu pour les top produits ?",
            "Quels produits ont le meilleur rapport qualité/prix ?",
        ]
        return [q for q in questions if q]
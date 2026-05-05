import os
from dotenv import load_dotenv
load_dotenv(override=True)

def get_llm(temperature=None, max_tokens=None):
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("Clé GROQ_API_KEY manquante dans .env")

    _temperature = temperature if temperature is not None else float(os.getenv("LLM_TEMPERATURE", "0.3"))
    _max_tokens  = max_tokens  if max_tokens  is not None else int(os.getenv("LLM_MAX_TOKENS",  "1000"))
    _model       = os.getenv("LLM_MODEL", "llama-3.1-8b-instant").strip()

    try:
        # Méthode directe sans passer par langchain-groq
        from groq import Groq
        from langchain_core.language_models.chat_models import BaseChatModel
        from langchain_groq import ChatGroq

        # Force la création sans proxy
        import httpx
        http_client = httpx.Client()
        client      = Groq(api_key=api_key, http_client=http_client)

        llm = ChatGroq(
            model=_model,
            temperature=_temperature,
            max_tokens=_max_tokens,
            groq_api_key=api_key,
        )
        return llm

    except TypeError:
        # Fallback : langchain-groq plus ancien
        from langchain_groq import ChatGroq
        return ChatGroq(
            model_name=_model,
            temperature=_temperature,
            max_tokens=_max_tokens,
            groq_api_key=api_key,
        )

def get_model_info():
    load_dotenv(override=True)
    return {
        "provider":    "Groq",
        "model":       os.getenv("LLM_MODEL", "llama-3.1-8b-instant"),
        "temperature": float(os.getenv("LLM_TEMPERATURE", "0.3")),
        "max_tokens":  int(os.getenv("LLM_MAX_TOKENS", "1000")),
    }
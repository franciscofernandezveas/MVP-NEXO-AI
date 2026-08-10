# core/llm.py
import os
import logging
from typing import Type
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

OPENAI_API_KEY = (
    os.environ.get("DEMO_OPENAI_API_KEY")
    or os.environ.get("OPENAI_API_KEY")
)

if not OPENAI_API_KEY:
    logger.warning("⚠️ DEMO_OPENAI_API_KEY u OPENAI_API_KEY no están configuradas.")

OPENAI_API_KEY = OPENAI_API_KEY.strip().strip('"').strip("'") if OPENAI_API_KEY else None

# ------------------------------------------------------------------
# Modelo configurable por variable de entorno
# ------------------------------------------------------------------
MODEL_NAME = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip().lower()

# Modelos de razonamiento (o-series) y los GPT más recientes (4.1, 5, etc.)
# no aceptan 'max_tokens' ni 'temperature'.
IS_REASONING = any(name in MODEL_NAME for name in ["o1", "o3", "o4", "o5"])
IS_NEW_GPT = any(name in MODEL_NAME for name in ["gpt-4.1", "gpt-5"])

if IS_REASONING or IS_NEW_GPT:
    llm_kwargs = {
        "model": MODEL_NAME,
        "api_key": OPENAI_API_KEY,
        "max_completion_tokens": 2048,   # ← obligatorio para modelos nuevos
    }
else:
    llm_kwargs = {
        "model": MODEL_NAME,
        "temperature": 0.0,
        "max_tokens": 2048,              # ← válido para gpt-4o-mini, gpt-4o, etc.
        "api_key": OPENAI_API_KEY,
    }

LLM = ChatOpenAI(**llm_kwargs)


def with_structured_output(model: Type[BaseModel], method: str = "function_calling") -> ChatOpenAI:
    return LLM.with_structured_output(model, method=method)

# core/llm.py
import os
import logging
from typing import Type, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# API Key: soporta DEMO_OPENAI_API_KEY (MVP) u OPENAI_API_KEY (fallback)
# ------------------------------------------------------------------
OPENAI_API_KEY = (
    os.environ.get("DEMO_OPENAI_API_KEY")
    or os.environ.get("OPENAI_API_KEY")
)

if not OPENAI_API_KEY:
    logger.warning("⚠️  DEMO_OPENAI_API_KEY u OPENAI_API_KEY no están configuradas. "
                   "Las llamadas a OpenAI fallarán.")

# Limpiar comillas accidentales que puedan venir de Railway
OPENAI_API_KEY = OPENAI_API_KEY.strip().strip('"').strip("'") if OPENAI_API_KEY else None

# Configurar LLM con valores por defecto razonables
LLM = ChatOpenAI(
    model="gpt-5.4-mini",  # Modelo más económico para pruebas
    temperature=0.0,      # Para respuestas más determinísticas
    max_tokens=2048,      # Límite de tokens
    api_key=OPENAI_API_KEY,
)


def with_structured_output(model: Type[BaseModel], method: str = "function_calling") -> ChatOpenAI:
    """
    Crea un LLM con salida estructurada.
    Usa function_calling para evitar problemas con structured output de OpenAI.
    """
    return LLM.with_structured_output(model, method=method)

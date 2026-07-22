# core/llm.py
import os
from typing import Type, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

# Configurar LLM con valores por defecto razonables
LLM = ChatOpenAI(
    model="gpt-4o-mini",  # Modelo más económico para pruebas
    temperature=0.0,      # Para respuestas más determinísticas
    max_tokens=2048,      # Límite de tokens
)


def with_structured_output(model: Type[BaseModel], method: str = "function_calling") -> ChatOpenAI:
    """
    Crea un LLM con salida estructurada.
    Usa function_calling para evitar problemas con structured output de OpenAI.
    """
    return LLM.with_structured_output(model, method=method)

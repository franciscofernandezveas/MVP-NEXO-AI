import json
import re
import asyncio
from typing import Optional


def sse_event(event_type: str, **payload) -> str:
    return f"data: {json.dumps({'type': event_type, **payload}, ensure_ascii=False)}\n\n"


async def sse_stream_text(text: Optional[str], sleep_time: float = 0.003):
    if not text:
        return
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    for i, sentence in enumerate(sentences):
        sep = " " if i < len(sentences) - 1 else ""
        yield sse_event("chunk", content=sentence + sep)
        await asyncio.sleep(sleep_time)


def detect_yes_no_response(text: str) -> Optional[bool]:
    text_lower = text.lower().strip()
    yes_words = ['sí', 'si', 'yes', 'y', 'ok', 'dale', 'claro', 'por supuesto',
                 'obvio', 'desde luego', 'adelante', 'confirmo', 'afirmativo']
    no_words = ['no', 'nop', 'nope', 'negativo', 'mejor no', 'no gracias',
                'paso', 'cancel', 'cancelar']
    if any(text_lower == w or text_lower.startswith(w + " ") for w in yes_words):
        return True
    if any(text_lower == w or text_lower.startswith(w + " ") for w in no_words):
        return False
    return None

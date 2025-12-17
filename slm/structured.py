"""local/structured.py — constrain, validate, retry once, then fail honestly."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from .client import Ollama

log = logging.getLogger(__name__)

EXTRACT_PROMPT = """Classify the message and extract entities.

Return JSON with exactly these keys:
  "intent": one of "question", "command", "chitchat"
  "entities": list of strings — people, organisations, products, dates, ids, places
  "confidence": number between 0 and 1

Message:
{text}"""

REPAIR_PROMPT = """Your previous output did not match the required JSON schema.

Previous output:
{raw}

Validation error:
{error}

Return only corrected JSON with keys "intent", "entities", "confidence"."""


class Extraction(BaseModel):
    intent: Literal["question", "command", "chitchat"]
    entities: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


@dataclass
class Attempted:
    result: Extraction | None
    attempts: int          # 1 = valid first time, 2 = needed the repair
    raw: list[str]


def extract(client: Ollama, model: str, text: str, retries: int = 1,
            temperature: float = 0.0, schema: bool = False) -> Attempted:
    # schema=False uses format="json" (valid JSON, any shape) so the
    # validation step has something to catch; schema=True passes the JSON
    # schema to ollama's constrained decoding.
    fmt = Extraction.model_json_schema() if schema else "json"
    prompt = EXTRACT_PROMPT.format(text=text)
    raws: list[str] = []
    for attempt in range(retries + 1):
        raw, _ = client.generate(model, prompt, fmt=fmt, temperature=temperature, num_predict=256)
        raws.append(raw)
        try:
            return Attempted(Extraction.model_validate_json(raw), attempt + 1, raws)
        except ValidationError as e:
            if attempt == retries:
                log.warning("extract.failed model=%s text=%r error=%s", model, text[:80], e)
                return Attempted(None, attempt + 1, raws)   # fail visibly, never silently
            prompt = REPAIR_PROMPT.format(raw=raw, error=e)
    raise AssertionError("unreachable")

"""FastAPI wrapper: a streaming generate endpoint and the validated extractor.

    uvicorn slm.api:app --port 8100
"""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .client import Ollama
from .structured import Extraction, extract

MODEL = os.environ.get("SLM_MODEL", "qwen3:4b")

app = FastAPI(title="local-slm")
client = Ollama()


class GenerateRequest(BaseModel):
    prompt: str
    temperature: float = 0.0
    max_tokens: int = 512


class ExtractRequest(BaseModel):
    text: str


@app.post("/generate")
def generate(req: GenerateRequest) -> StreamingResponse:
    def tokens():
        for ev in client.stream(MODEL, req.prompt, temperature=req.temperature,
                                num_predict=req.max_tokens):
            if not ev.done:
                yield ev.text
    return StreamingResponse(tokens(), media_type="text/plain")


@app.post("/extract", response_model=Extraction)
def extract_endpoint(req: ExtractRequest) -> Extraction:
    a = extract(client, MODEL, req.text)
    if a.result is None:
        # 422 with the raw outputs, not a 200 with a guess
        raise HTTPException(422, {"error": "model output failed validation twice", "raw": a.raw})
    return a.result


@app.get("/health")
def health() -> dict:
    return {"ok": True, "model": MODEL}

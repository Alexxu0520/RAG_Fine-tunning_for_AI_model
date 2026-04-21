from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

from infer_compare_all import (
    ADAPTER_PATH,
    answer_base_only,
    answer_base_with_rag,
    answer_lora_only,
    answer_lora_with_rag,
    run_all,
)

app = FastAPI(title="RDR2 Compare UI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


class AskRequest(BaseModel):
    question: str
    mode: str = "all"
    adapter_path: Optional[str] = ADAPTER_PATH


@app.get("/")
def root():
    return FileResponse("static/index.html")


@app.post("/api/ask")
def ask(req: AskRequest):
    question = req.question.strip()
    adapter_path = req.adapter_path or ADAPTER_PATH

    if not question:
        return {"error": "Question cannot be empty."}

    try:
        if req.mode == "base":
            return {
                "mode": "base",
                "results": [
                    {"name": "base", "answer": answer_base_only(question)}
                ],
            }

        elif req.mode == "base_rag":
            return {
                "mode": "base_rag",
                "results": [
                    {"name": "base_rag", "answer": answer_base_with_rag(question, use_reranker=True)}
                ],
            }

        elif req.mode == "base_rag_no_rerank":
            return {
                "mode": "base_rag_no_rerank",
                "results": [
                    {"name": "base_rag_no_rerank", "answer": answer_base_with_rag(question, use_reranker=False)}
                ],
            }

        elif req.mode == "lora":
            return {
                "mode": "lora",
                "results": [
                    {"name": "lora", "answer": answer_lora_only(question, adapter_path=adapter_path)}
                ],
            }

        elif req.mode == "lora_rag":
            return {
                "mode": "lora_rag",
                "results": [
                    {"name": "lora_rag", "answer": answer_lora_with_rag(question, adapter_path=adapter_path, use_reranker=True)}
                ],
            }

        elif req.mode == "lora_rag_no_rerank":
            return {
                "mode": "lora_rag_no_rerank",
                "results": [
                    {"name": "lora_rag_no_rerank", "answer": answer_lora_with_rag(question, adapter_path=adapter_path, use_reranker=False)}
                ],
            }

        elif req.mode == "all":
            results = run_all(question, adapter_path)
            return {
                "mode": "all",
                "results": [{"name": name, "answer": answer} for name, answer in results],
            }

        else:
            return {"error": f"Unsupported mode: {req.mode}"}

    except Exception as e:
        return {"error": str(e)}
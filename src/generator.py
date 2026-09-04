
"""
Generation: takes the user's question + the top-k retrieved chunks and
asks an LLM to answer USING ONLY that context. This is the step that
turns "search results" into a natural-language, grounded answer, and
is what separates RAG from plain semantic search.
The prompt explicitly instructs the model to say it doesn't know if the
answer isn't in the context, instead of guessing -- this is what makes
the answer "grounded" rather than a hallucination risk.

"""
import os
from typing import List, Tuple


PROMPT_TEMPLATE = """You are a helpful assistant. Answer the user's question \
using ONLY the context below. If the answer is not contained in the context, \
say "I don't have enough information in the provided document to answer that."

Context:
{context}

Question: {question}

Answer:"""

def build_prompt(question: str, retrieved: List[Tuple[str, dict, float]]) -> str:
    blocks = []
    for i, (text, meta, score) in enumerate(retrieved):
        source = meta.get("source", "unknown")
        blocks.append(f"[Chunk {i + 1} | source: {source} | score: {score:.3f}]\n{text}")
    return PROMPT_TEMPLATE.format(context="\n\n".join(blocks), question=question)


def generate_answer(question: str, retrieved: List[Tuple[str, dict, float]],
                     model: str = "qwen/qwen3.6-27b") -> str:
    from groq import Groq

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not set. Get a free key at https://console.groq.com "
            "and put it in a .env file (copy .env.example)."
        )

    client = Groq(api_key=api_key)
    prompt = build_prompt(question, retrieved)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content

from typing import List, Tuple
from db import AudioChunk
from ai21 import tokenizers


def count_tokens(text: str) -> int:
    """Count tokens using AI21 tokenizer"""
    tokenizer = tokenizers.get_tokenizer()
    tokens = tokenizer.tokenize(text)
    return len(tokens)


def smart_chunk_selection(chunks: List[AudioChunk], max_tokens: int = 8000) -> str:
    """Smart chunk selection using embeddings and relevance scoring"""

    # Calculate average embedding as centroid
    all_embeddings = [chunk.embedding for chunk in chunks]
    centroid = [sum(x) / len(x) for x in zip(*all_embeddings)]

    # Score chunks by similarity to centroid
    chunk_scores: List[Tuple[AudioChunk, float]] = []
    for chunk in chunks:
        similarity = sum(a * b for a, b in zip(chunk.embedding, centroid))
        chunk_scores.append((chunk, similarity))

    # Sort by relevance
    chunk_scores.sort(key=lambda x: x[1], reverse=True)

    # Build combined text within token limit
    combined_text = ""
    current_tokens = 0

    for chunk, _ in chunk_scores:
        chunk_tokens = count_tokens(chunk.chunk_text)
        if current_tokens + chunk_tokens <= max_tokens:
            combined_text += " " + chunk.chunk_text
            current_tokens += chunk_tokens
        else:
            break

    return combined_text.strip()

from typing import List, Tuple
import tiktoken
from db import AudioChunk


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Accurate token counting using tiktoken"""
    encoder = tiktoken.encoding_for_model(model)
    return len(encoder.encode(text))


def truncate_to_token_limit(text: str, max_tokens: int = 7000) -> str:
    """Truncate text to stay within token limit while keeping whole sentences"""
    if count_tokens(text) <= max_tokens:
        return text

    encoder = tiktoken.encoding_for_model("gpt-4o")
    tokens = encoder.encode(text)
    truncated_tokens = tokens[:max_tokens]
    truncated_text = encoder.decode(truncated_tokens)

    # Try to end at a sentence boundary
    last_period = truncated_text.rfind(".")
    if last_period > 0:
        truncated_text = truncated_text[: last_period + 1]

    return truncated_text


def smart_chunk_selection(chunks: List[AudioChunk], max_tokens: int = 7000) -> str:
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

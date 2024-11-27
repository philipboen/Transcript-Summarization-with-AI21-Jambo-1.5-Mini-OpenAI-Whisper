from typing import List
from nltk.tokenize import sent_tokenize

def chunk_text(text: str, max_tokens: int = 7000) -> List[str]:
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        # Rough estimate: 1 token ~= 4 characters
        sentence_tokens = len(sentence) // 4
        if current_length + sentence_tokens > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_tokens
        else:
            current_chunk.append(sentence)
            current_length += sentence_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks
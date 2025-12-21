import re
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# модель эмбеддингов
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# чанки (режем по разделам "1." "2." "5." даже если перед ними пробелы)
def make_chunks(text: str, chunk_size: int = 2200):
    text = text.replace("\r", "\n")

    # режем по началу разделов вида "   5. ..." (с пробелами)
    parts = re.split(r"\n(?=\s*\d+\.\s)", text)

    parts = [p.strip() for p in parts if p.strip()]

    # fallback: если split не сработал
    if len(parts) <= 1:
        parts = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    for p in parts:
        if len(p) <= chunk_size:
            chunks.append(p)
        else:
            for i in range(0, len(p), chunk_size):
                chunks.append(p[i:i + chunk_size])

    return chunks

# делаем эмбеддинги
def embed(text_list):
    vectors = embed_model.encode(text_list)
    return np.array(vectors).astype("float32")

# faiss
def build_faiss_index(chunks):
    vectors = embed(chunks)
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index, vectors

# rag (с fallback по ключевым словам + без дублей)
def rag_search(chunks, index, vectors, query: str, top_k: int = 3) -> str:
    q = query.lower()
    keywords = [
        "телефон", "мобильн", "смартфон", "электрон", "устройств",
        "фото", "видео", "аудио", "съем", "съём", "запись", "камера",
        "согласие", "перемена", "урок"
    ]

    # 1) если вопрос про телефоны/съёмку — сначала ищем чанки по словам
    if any(k in q for k in keywords):
        hits = []
        seen = set()
        for ch in chunks:
            low = ch.lower()
            if any(k in low for k in keywords):
                t = ch.strip()
                if t and t not in seen:
                    seen.add(t)
                    hits.append(t)
            if len(hits) >= top_k:
                break
        if hits:
            return "\n\n---\n\n".join(hits)

    # 2) иначе обычный faiss
    q_vector = embed([query])
    k = min(max(top_k * 6, top_k), len(chunks))
    distances, ids = index.search(q_vector, k)

    seen = set()
    results = []
    for i in ids[0]:
        text = chunks[int(i)].strip()
        if text and text not in seen:
            seen.add(text)
            results.append(text)
        if len(results) >= top_k:
            break

    return "\n\n---\n\n".join(results)

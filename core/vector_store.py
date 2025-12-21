FAISS_STORES = {} # хранилище всех индексов в памяти
INDEX_COUNTER = 0 # счётчик для уникальных айди индексов

# делаем айди индекса
def save_index(chunks, index, vectors) -> str:
    global INDEX_COUNTER, FAISS_STORES  # используем глобальные переменные
    INDEX_COUNTER += 1  # увеличиваем счётчик
    index_id = f"index_{INDEX_COUNTER}"  # формируем айди индекса
    FAISS_STORES[index_id] = (chunks, index, vectors)  # сохраняем данные
    return index_id  # возвращаем айди индекса

# индекс по айди
def get_index(index_id: str):
    found = index_id in FAISS_STORES
    return FAISS_STORES.get(index_id)

import hashlib


def _chunk_id(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class SimpleChromaWrapper:
    def __init__(self, collection, embeddings):
        self.collection = collection
        self._embeddings = embeddings

    def similarity_search(self, query, k=2):
        q_emb = self._embeddings.embed_query(query)

        res = self.collection.query(
            query_embeddings=[q_emb],
            n_results=k
        )

        docs = res.get("documents", [])

        if docs and isinstance(docs[0], list):
            docs = docs[0]

        from types import SimpleNamespace
        return [SimpleNamespace(page_content=d) for d in docs]


def create_vector_store(chunks, embeddings, persist_directory=None):
    import chromadb

    if persist_directory:
        client = chromadb.PersistentClient(path=persist_directory)
    else:
        client = chromadb.Client()

    try:
        collection = client.get_collection("ai_study_buddy")
    except:
        collection = client.create_collection("ai_study_buddy")

    embeddings_list = embeddings.embed_documents(chunks)

    # Force pure float conversion
    normalized_embeddings = []
    for vec in embeddings_list:
        normalized_embeddings.append([float(x) for x in vec])

    ids = [_chunk_id(chunk) for chunk in chunks]

    print("Chunks:", len(chunks))
    print("Embedding shape:", len(normalized_embeddings[0]))

    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=normalized_embeddings
    )

    return SimpleChromaWrapper(collection, embeddings)
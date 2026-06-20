try:
    from langchain.vectorstores import Chroma
except Exception:
    Chroma = None

import hashlib


def _chunk_id(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class SimpleChromaWrapper:
    def __init__(self, collection, embeddings):
        self.collection = collection
        self._embeddings = embeddings

    def similarity_search(self, query, k=2):
        try:
            q_emb = self._embeddings.embed_query(query)
        except Exception:
            # try alternate method name
            q_emb = self._embeddings.embed_documents([query])[0]

        res = self.collection.query(query_embeddings=[q_emb], n_results=k)
        docs = res.get("documents") or res.get("results") or []
        if isinstance(docs, list) and docs and isinstance(docs[0], list):
            docs = docs[0]

        results = []
        from types import SimpleNamespace

        for d in docs:
            results.append(SimpleNamespace(page_content=d))
        return results


def create_vector_store(chunks, embeddings, persist_directory=None):
    """Create a vector store. Prefer LangChain's Chroma, otherwise use chromadb directly.

    Returns an object with `similarity_search(query, k)`.
    """
    if Chroma is not None:
        return Chroma.from_texts(chunks, embedding=embeddings, persist_directory=persist_directory)

    # Fallback: use chromadb directly
    import chromadb

    if persist_directory:
        client = chromadb.PersistentClient(path=persist_directory)
    else:
        client = chromadb.Client()

    # Create or get a collection
    try:
        collection = client.get_collection(name="ai_study_buddy")
    except Exception:
        collection = client.create_collection(name="ai_study_buddy")

    # Compute embeddings for docs
    try:
        embeddings_list = embeddings.embed_documents(chunks)
    except Exception:
        # If the embedding object uses a different method name
        embeddings_list = [embeddings.embed_query(c) for c in chunks]

    ids = [_chunk_id(chunk) for chunk in chunks]
    collection.upsert(ids=ids, documents=chunks, embeddings=embeddings_list)

    return SimpleChromaWrapper(collection, embeddings)

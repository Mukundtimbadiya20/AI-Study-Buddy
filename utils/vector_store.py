import hashlib
from types import SimpleNamespace


def _chunk_id(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class SimpleChromaWrapper:
    def __init__(self, collection, embeddings):
        self.collection = collection
        self._embeddings = embeddings

    def similarity_search(self, query, k=2):
        try:
            query_embedding = self._embeddings.embed_query(query)
            query_embedding = [float(x) for x in query_embedding]

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k
            )

            docs = results.get("documents", [])

            if docs and isinstance(docs[0], list):
                docs = docs[0]

            return [
                SimpleNamespace(page_content=doc)
                for doc in docs
            ]

        except Exception as e:
            print("Similarity search failed:", e)
            return []


def create_vector_store(chunks, embeddings, persist_directory=None):
    import chromadb
    from chromadb.config import Settings

    if not chunks:
        raise ValueError("No chunks available for vector store")

    # Disable telemetry
    settings = Settings(anonymized_telemetry=False)

    try:
        if persist_directory:
            client = chromadb.PersistentClient(
                path=persist_directory,
                settings=settings
            )
        else:
            client = chromadb.Client(settings=settings)
    except Exception as e:
        print("Chroma client creation failed:", e)
        raise

    # Collection
    try:
        collection = client.get_collection("ai_study_buddy")
    except Exception:
        collection = client.create_collection("ai_study_buddy")

    # Generate embeddings
    try:
        embeddings_list = embeddings.embed_documents(chunks)
    except Exception as e:
        print("Embedding generation failed:", e)
        raise

    if not embeddings_list:
        raise ValueError("Embeddings list is empty")

    normalized_embeddings = []

    for vec in embeddings_list:
        normalized_embeddings.append(
            [float(x) for x in vec]
        )

    ids = [_chunk_id(chunk) for chunk in chunks]

    print("===== VECTOR STORE DEBUG =====")
    print("Chunks count:", len(chunks))
    print("Embeddings count:", len(normalized_embeddings))
    print("Embedding dimension:", len(normalized_embeddings[0]))
    print("==============================")

    try:
        collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=normalized_embeddings
        )
    except Exception as e:
        print("Chroma upsert failed:", e)
        raise

    return SimpleChromaWrapper(collection, embeddings)

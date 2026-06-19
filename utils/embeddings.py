try:
    from langchain_huggingface import HuggingFaceEmbeddings

    def get_embedding_model():
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

except Exception:
    try:
        from sentence_transformers import SentenceTransformer

        class HFWrapper:
            def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
                self.model = SentenceTransformer(model_name)

            def embed_documents(self, texts):
                return [list(x) for x in self.model.encode(texts, show_progress_bar=False)]

            def embed_query(self, text):
                v = self.model.encode([text], show_progress_bar=False)[0]
                return list(v)

        def get_embedding_model():
            return HFWrapper()
    except Exception:
        import hashlib

        class SimpleEmbedding:
            """Deterministic low-quality fallback embedding used when no
            embedding libraries are available. Produces a fixed-length 32-D
            vector derived from SHA-256 digest to keep the app running.
            """
            def _embed_text(self, text):
                h = hashlib.sha256(text.encode('utf-8')).digest()
                return [b / 255.0 for b in h[:32]]

            def embed_documents(self, texts):
                return [self._embed_text(t) for t in texts]

            def embed_query(self, text):
                return self._embed_text(text)

        def get_embedding_model():
            return SimpleEmbedding()

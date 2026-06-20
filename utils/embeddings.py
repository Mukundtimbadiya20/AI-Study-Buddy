from sentence_transformers import SentenceTransformer
import hashlib


class HFWrapper:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        vectors = self.model.encode(texts, show_progress_bar=False)
        return [[float(x) for x in vec] for vec in vectors]

    def embed_query(self, text):
        vector = self.model.encode([text], show_progress_bar=False)[0]
        return [float(x) for x in vector]


class SimpleEmbedding:
    """Fallback embedding if sentence-transformers fails"""

    def _embed_text(self, text):
        h = hashlib.sha256(text.encode("utf-8")).digest()
        return [float(b) / 255.0 for b in h[:32]]

    def embed_documents(self, texts):
        return [self._embed_text(t) for t in texts]

    def embed_query(self, text):
        return self._embed_text(text)


def get_embedding_model():
    try:
        return HFWrapper()
    except Exception as e:
        print("Embedding model failed:", e)
        return SimpleEmbedding()
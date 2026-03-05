from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def generateEmbedding(self, text: str):
        if not text or not text.strip():
            raise ValueError("cannot generate embedding for empty text")

        sentences = [text]
        embeddings = self.model.encode(sentences)  # type: ignore
        if len(embeddings) < 1:
            raise Exception("encode() returned an empty list")

        return embeddings[0]


def verifyModel():
    # import torch
    # if torch.cuda.is_available():
    #     print(f"\nDEVICE: {torch.cuda.get_device_name(0)}")
    #     print(torch.__version__)
    #     print(torch.version.cuda)  # type: ignore
    #     print(torch.cuda.get_arch_list())
    # else:
    #     print("\nCUDA is not available")

    ss = SemanticSearch()
    print()
    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")


def embedText(text: str):
    ss = SemanticSearch()
    embedding = ss.generateEmbedding(text)

    print()
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

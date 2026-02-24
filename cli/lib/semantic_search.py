from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")


def verifyModel():
    # import torch
    # x = torch.rand(5, 3)
    # print(x)
    # print(torch.cuda.is_available())
    # print()

    ss = SemanticSearch()
    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")

    # self.model.encode(text)

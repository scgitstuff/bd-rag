from .search_utils import normalize


def normalizeCommand(scores: list[float]):
    print(scores)

    scores = normalize(scores)

    for score in scores:
        print(f"* {score:.4f}")

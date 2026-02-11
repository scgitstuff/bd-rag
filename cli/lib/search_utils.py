import json

MOVIES_FILE = "data/movies.json"
STOP_FILE = "data/stopwords.txt"


# since it is read only, I think it is ok to just use str, even though id is int
# using multiple types caused errors in calling code
def loadMovies() -> list[dict[str, str]]:
    with open(MOVIES_FILE, "r") as f:
        data = json.load(f)

    return data["movies"]


def loadStopWords() -> list[str]:
    with open(STOP_FILE, "r") as f:
        return f.read().splitlines()

import json
from functools import cache

MOVIES_FILE = "data/movies.json"


# since it is read only, I think it is ok to just use str, eventhough id is int
# using multiple types caused errors in calling code
@cache
def loadMovies() -> list[dict[str, str]]:
    with open(MOVIES_FILE, "r") as f:
        movies = json.load(f)

    return movies["movies"]

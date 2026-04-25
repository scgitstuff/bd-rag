import os
import time
from dotenv import load_dotenv
from google import genai
from .constants import EnhanceOptions, RerankMethods


load_dotenv()
_api_key = os.environ.get("GEMINI_API_KEY")
if not _api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

_client = genai.Client(api_key=_api_key)
_model = "gemma-3-27b-it"


def rerankResults(movies: list[dict[str, str]], rerankMethod: str, query: str):
    match rerankMethod:
        case RerankMethods.INDIVIDUAL:
            _rerankAll(movies, query)
            return
        case _:
            return


def _rerankAll(movies: list[dict[str, str]], query: str):
    for movie in movies:
        _rerankIndividual(movie, query)
        time.sleep(3)


def _rerankIndividual(movie: dict[str, str], query: str):
    contents = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {movie.get("title", "")} - {movie.get("description", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Output ONLY the number in your response, no other text or explanation.

Score:"""

    print(f"Rerank {movie.get("title", "")}")
    movie["LLM"] = _getContent(contents)


# the solution runs this code in rrfSearch()
# I think the solution is doing it wrong
# this is a pre-process on the query, not part of the search functionality
def enhanceQuery(enhance: str, query: str) -> str:
    match enhance:
        case EnhanceOptions.SPELL:
            return _spellCheck(query)
        case EnhanceOptions.REWRITE:
            return _rewrite(query)
        case EnhanceOptions.EXPAND:
            return _expand(query)
        case _:
            return query


def _expand(query: str) -> str:
    contents = f"""Expand the user-provided movie search query below with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
Output only the additional terms; they will be appended to the original query.

Examples:
- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

User query: "{query}"
"""

    corrected = _getContent(contents)
    if corrected:
        print(
            f"\nEnhanced query ({EnhanceOptions.EXPAND}): '{query}' -> '{corrected}'\n"
        )
        return f"{query} {corrected}".strip()

    return query


def _rewrite(query: str) -> str:
    contents = f"""Rewrite the user-provided movie search query below to be more specific and searchable.

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep the rewritten query concise (under 10 words)
- It should be a Google-style search query, specific enough to yield relevant results
- Don't use boolean logic

Examples:
- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

If you cannot improve the query, output the original unchanged.
Output only the rewritten query text, nothing else.

User query: "{query}"
"""

    corrected = _getContent(contents)
    if corrected:
        print(
            f"\nEnhanced query ({EnhanceOptions.REWRITE}): '{query}' -> '{corrected}'\n"
        )
        return corrected

    return query


def _spellCheck(query: str) -> str:
    contents = f"""Fix any spelling errors in the user-provided movie search query below.
Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
Preserve punctuation and capitalization unless a change is required for a typo fix.
If there are no spelling errors, or if you're unsure, output the original query unchanged.
Output only the final query text, nothing else.
User query: "{query}"
"""

    corrected = _getContent(contents)
    if corrected:
        print(
            f"\nEnhanced query ({EnhanceOptions.SPELL}): '{query}' -> '{corrected}'\n"
        )
        return corrected

    return query


def _getContent(contents: str) -> str:
    response = _client.models.generate_content(  # type: ignore
        model=_model, contents=contents
    )

    out: str = ""
    if response.text:
        out = response.text.strip().strip('"')

    return out

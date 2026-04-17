import os
from dotenv import load_dotenv
from google import genai


def test():
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")

    client = genai.Client(api_key=api_key)
    contents = "Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum."

    response = client.models.generate_content(  # type: ignore
        model="gemma-3-27b-it",
        contents=contents,
    )

    print()
    print(response.text)

    print()
    if response.usage_metadata is None:
        print("response.usage_metadata is None")
        return

    x = response.usage_metadata.prompt_token_count
    y = response.usage_metadata.candidates_token_count

    print(f"Prompt tokens: {x}")
    print(f"Response tokens: {y}")


test()

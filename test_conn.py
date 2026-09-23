import requests
import os
from dotenv import load_dotenv
load_dotenv()

def search_web(query: str, num_results: int = 5) -> list[dict]:
    api_key = os.getenv("SERP_API_KEY")
    response = requests.get(
        "https://serpapi.com/search",
        params={
            "q": query,
            "api_key": api_key,
            "num": num_results,
            "engine": "google",
        },
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()

    results = []
    for item in data.get("organic_results", [])[:num_results]:
        results.append({
            "title": item.get("title", ""),
            "url": item.get("link", ""),
        })
    return results

results = search_web("climate change refers to long-term shifts in temperatures")
print(results)
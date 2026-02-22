"""
News Tool — NewsAPI Integration
Fetches latest news articles for a given topic using the free tier API.
"""

import os
import httpx

# NewsAPI endpoint
NEWSAPI_BASE_URL = "https://newsapi.org/v2/everything"


async def get_news(topic: str) -> dict:
    """
    Fetch latest news articles for a topic from NewsAPI.

    Args:
        topic: The search topic (e.g., "India", "technology", "cricket")

    Returns:
        dict with keys: success, data/error
    """
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        return {
            "success": False,
            "error": "NEWS_API_KEY is not set in environment variables.",
        }

    params = {
        "q": topic,
        "apiKey": api_key,
        "pageSize": 5,       # Top 5 headlines
        "sortBy": "publishedAt",
        "language": "en",
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(NEWSAPI_BASE_URL, params=params)

        if response.status_code == 401:
            return {
                "success": False,
                "error": "Invalid News API key. Please check your NEWS_API_KEY.",
            }
        if response.status_code == 429:
            return {
                "success": False,
                "error": "NewsAPI rate limit exceeded. Free tier allows 100 requests/day.",
            }
        if response.status_code != 200:
            return {
                "success": False,
                "error": f"NewsAPI returned status {response.status_code}: {response.text}",
            }

        data = response.json()
        articles = data.get("articles", [])

        if not articles:
            return {
                "success": False,
                "error": f"No news articles found for topic '{topic}'.",
            }

        # Format articles into clean data
        news_items = []
        for article in articles[:5]:
            news_items.append({
                "title": article.get("title", "No title"),
                "source": article.get("source", {}).get("name", "Unknown"),
                "description": article.get("description", "No description available."),
                "url": article.get("url", ""),
                "published_at": article.get("publishedAt", ""),
            })

        return {"success": True, "data": {"topic": topic, "articles": news_items}}

    except httpx.TimeoutException:
        return {
            "success": False,
            "error": f"Request timed out while fetching news for '{topic}'. Try again later.",
        }
    except httpx.ConnectError:
        return {
            "success": False,
            "error": "Could not connect to NewsAPI. Check your internet connection.",
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error fetching news: {str(e)}",
        }


def format_news_response(result: dict) -> str:
    """Format the news result into a human-readable string."""
    if not result["success"]:
        return f"❌ News Error: {result['error']}"

    data = result["data"]
    lines = [f"📰 Latest News on '{data['topic']}':\n"]

    for i, article in enumerate(data["articles"], 1):
        lines.append(
            f"  {i}. {article['title']}\n"
            f"     📌 Source: {article['source']}\n"
            f"     📝 {article['description']}\n"
            f"     🔗 {article['url']}\n"
        )

    return "\n".join(lines)

"""
LangGraph Workflow — Agentic Intent Router
Routes user queries to Weather, News, or Gemini LLM based on detected intent.

Flow:  Input → Intent Detection (keyword-based) → Routing → Tool/LLM → Response

Design:
- Intent detection is ALWAYS keyword-based (instant, no API calls needed).
- Gemini is ONLY used for answering general questions ("other" intent).
- This maximizes Gemini free-tier quota for actual AI responses.
- Weather and News queries never touch Gemini → always instant.
"""

import os
import re
import json
import time
import asyncio
from typing import TypedDict, Literal

from langgraph.graph import StateGraph, END
from google import genai

from tools.weather import get_weather, get_weather_multi, format_weather_response, format_multi_weather_response
from tools.news import get_news, format_news_response


# ──────────────────────────────────────────────
# State Schema
# ──────────────────────────────────────────────

class AgentState(TypedDict):
    """Shared state that flows through the graph."""
    query: str          # User's original input
    intent: str         # Detected intent: "weather" | "news" | "other"
    city: str           # Extracted city name (for weather queries)
    cities: list        # Multiple cities (for multi-city weather)
    topic: str          # Extracted topic (for news queries)
    response: str       # Final formatted response


# ──────────────────────────────────────────────
# Gemini Client + Cooldown
# ──────────────────────────────────────────────

# Cooldown: skip Gemini for 60s after a rate-limit hit
_gemini_cooldown_until = 0.0
COOLDOWN_SECONDS = 60


def _is_gemini_on_cooldown() -> bool:
    return time.time() < _gemini_cooldown_until


def _set_gemini_cooldown():
    global _gemini_cooldown_until
    _gemini_cooldown_until = time.time() + COOLDOWN_SECONDS
    print(f"🧊 Gemini on cooldown for {COOLDOWN_SECONDS}s.")


def _clear_gemini_cooldown():
    global _gemini_cooldown_until
    _gemini_cooldown_until = 0.0


def _get_gemini_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set.")
    return genai.Client(api_key=api_key)


def _is_rate_limit_error(error_str: str) -> bool:
    lower = error_str.lower()
    return "429" in error_str or "quota" in lower or "resource_exhausted" in lower

# Models to try in order — different models have separate rate limits!
GEMINI_MODELS = [
    "gemini-2.5-flash-lite",   # Newest, likely highest free limits
    "gemini-2.5-flash",        # 2.5 flash
    "gemini-2.0-flash",        # 2.0 flash
    "gemini-1.5-flash",        # 1.5 flash (older, separate quota)
]


async def _call_gemini(prompt: str) -> str:
    """
    Call Gemini with model fallback chain.
    Tries multiple models — each has its own separate rate limit,
    so if one is exhausted, another may still work.
    """
    if _is_gemini_on_cooldown():
        remaining = int(_gemini_cooldown_until - time.time())
        raise Exception(f"429 Gemini on cooldown ({remaining}s remaining)")

    client = _get_gemini_client()
    last_error = None

    for model in GEMINI_MODELS:
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
            )
            _clear_gemini_cooldown()
            print(f"  [Gemini] Success with {model}")
            return response.text
        except Exception as e:
            error_str = str(e)
            last_error = e
            if _is_rate_limit_error(error_str) or "NOT_FOUND" in error_str or "404" in error_str:
                print(f"  [Gemini] {model} failed ({error_str[:60]}...), trying next...")
                continue
            else:
                raise e  # Non-recoverable error

    # All models failed — set cooldown
    _set_gemini_cooldown()
    raise last_error



# ──────────────────────────────────────────────
# Keyword-Based Intent Detection (Primary)
# ──────────────────────────────────────────────

# Weather keywords including common typos
WEATHER_KEYWORDS = [
    "weather", "whether", "wether", "wheather",
    "temperature", "forecast", "climate",
    "rain", "raining", "sunny", "humid", "humidity",
    "wind", "hot today", "cold today", "storm",
]

# Pattern 1: "weather in Chennai", "temperature at Goa"
WEATHER_PREP_PATTERN = re.compile(
    r"(?:weather|whether|wether|wheather|temperature|forecast|climate)"
    r"\s+(?:in|at|of|for)\s+([a-zA-Z][a-zA-Z\s]+)",
    re.IGNORECASE,
)

# Pattern 2: "weather chennai", "weather goa" (no preposition)
WEATHER_DIRECT_PATTERN = re.compile(
    r"(?:weather|whether|wether|wheather|temperature|forecast)"
    r"\s+([a-zA-Z][a-zA-Z\s]{1,30})$",
    re.IGNORECASE,
)

# Pattern 3: "chennai weather", "goa forecast" (city first)
WEATHER_REVERSE_PATTERN = re.compile(
    r"^([a-zA-Z][a-zA-Z\s]{1,30})\s+(?:weather|whether|wether|wheather|temperature|forecast|climate)",
    re.IGNORECASE,
)

# Pattern 4: "how is weather in chennai", "today weather goa", "today's weather in delhi"
WEATHER_ALT_PATTERN = re.compile(
    r"(?:how\s+is|what\s*'?s|today|today's|current|tell\s+me)\s+(?:the\s+)?(?:weather|whether|wether)"
    r"\s*(?:in|at|of|for)?\s+([a-zA-Z][a-zA-Z\s]+)",
    re.IGNORECASE,
)

# General question keywords — these should NOT go to weather tool
GENERAL_QUESTION_WORDS = [
    "what is", "what are", "define", "explain",
    "how does", "why is", "why does", "who is",
    "tell me about", "describe",
]

# News keywords
NEWS_KEYWORDS = [
    "news", "headline", "headlines",
    "latest", "breaking", "trending",
    "update", "updates", "article", "articles",
    "current affairs", "current events",
]

NEWS_TOPIC_PATTERN = re.compile(
    r"(?:news|headlines?|updates?)\s+(?:in|about|on|from|of|for|regarding)\s+(.+)",
    re.IGNORECASE,
)

NEWS_LATEST_PATTERN = re.compile(
    r"(?:latest|breaking|recent|top|trending)\s+(?:news|headlines?|updates?)"
    r"\s*(?:in|about|on|from|of|for|regarding)?\s*(.*)",
    re.IGNORECASE,
)


def _split_cities(raw: str) -> list:
    """Split a multi-city string like 'chennai and bangalore' into a list."""
    parts = re.split(r'\s+and\s+|\s*&\s*|\s*,\s*', raw)
    cities = [p.strip().rstrip('?.!,').title() for p in parts if p.strip()]
    return cities


def _is_general_question(query_lower: str) -> bool:
    """Check if the query is asking a general knowledge question about weather."""
    return any(query_lower.startswith(gq) for gq in GENERAL_QUESTION_WORDS)


def _extract_weather_cities(query: str) -> list | None:
    """
    Try all weather patterns to extract city names.
    Returns list of cities or None if no match.
    """
    # Skip if it's a general question like "what is weather"
    query_lower = query.lower().strip()
    if _is_general_question(query_lower):
        # Still allow "what is weather in chennai" — check for a city after a preposition
        match = WEATHER_PREP_PATTERN.search(query)
        if match:
            raw = match.group(1).strip().rstrip("?.!,")
            return _split_cities(raw)
        return None

    # Pattern 1: "weather in Chennai"
    match = WEATHER_PREP_PATTERN.search(query)
    if match:
        raw = match.group(1).strip().rstrip("?.!,")
        return _split_cities(raw)

    # Pattern 4: "today weather goa" (before direct, since it's more specific)
    match = WEATHER_ALT_PATTERN.search(query)
    if match:
        raw = match.group(1).strip().rstrip("?.!,")
        return _split_cities(raw)

    # Pattern 2: "weather goa" (no preposition)
    match = WEATHER_DIRECT_PATTERN.search(query)
    if match:
        raw = match.group(1).strip().rstrip("?.!,")
        return _split_cities(raw)

    # Pattern 3: "goa weather"
    match = WEATHER_REVERSE_PATTERN.search(query)
    if match:
        raw = match.group(1).strip().rstrip("?.!,")
        return _split_cities(raw)

    return None


def _detect_intent_keywords(query: str) -> dict:
    """
    Fast keyword-based intent detection.
    No API calls — runs instantly every time.
    Supports multi-city weather queries and flexible patterns.
    """
    query_lower = query.lower().strip()

    # ── Weather ──
    cities = _extract_weather_cities(query)
    if cities:
        return {
            "intent": "weather",
            "city": cities[0],
            "cities": cities,
            "topic": "",
        }

    # ── News ──
    if any(kw in query_lower for kw in NEWS_KEYWORDS):
        topic = ""

        match = NEWS_LATEST_PATTERN.search(query)
        if match and match.group(1).strip():
            topic = match.group(1).strip()
        else:
            match = NEWS_TOPIC_PATTERN.search(query)
            if match:
                topic = match.group(1).strip()

        if topic:
            topic = topic.rstrip("?.!,")

        return {"intent": "news", "city": "", "cities": [], "topic": topic if topic else query}

    # ── Default: general question → Gemini ──
    return {"intent": "other", "city": "", "cities": [], "topic": ""}


# ──────────────────────────────────────────────
# Graph Nodes
# ──────────────────────────────────────────────

async def detect_intent(state: AgentState) -> dict:
    """
    Node 1: Detect intent using KEYWORDS ONLY (no Gemini call).
    This saves ALL Gemini quota for answering actual questions.
    """
    query = state["query"]
    result = _detect_intent_keywords(query)

    intent = result["intent"]
    cities = result.get("cities", [])
    if intent == "weather":
        if len(cities) > 1:
            print(f"🌤️ Intent: weather → {len(cities)} cities: {cities}")
        else:
            print(f"🌤️ Intent: weather → city: {result['city']}")
    elif intent == "news":
        print(f"📰 Intent: news → topic: {result.get('topic', query)}")
    else:
        print(f"🧠 Intent: other → will use Gemini AI")

    return result


async def weather_tool(state: AgentState) -> dict:
    """Node 2a: Fetch weather data for one or more cities."""
    cities = state.get("cities", [])
    city = state.get("city", "")

    if not cities and not city:
        return {
            "response": (
                "❌ Could not determine which city to check weather for.\n"
                "Please mention a city name, e.g.:\n"
                '  • "Weather in Chennai"\n'
                '  • "Weather in Chennai and Bangalore"\n'
                '  • "Temperature in Cuddalore"'
            )
        }

    # Multi-city support
    if len(cities) > 1:
        results = await get_weather_multi(cities)
        return {"response": format_multi_weather_response(results)}
    else:
        target_city = cities[0] if cities else city
        result = await get_weather(target_city)
        return {"response": format_weather_response(result)}


async def news_tool(state: AgentState) -> dict:
    """Node 2b: Fetch news articles for the detected topic."""
    topic = state.get("topic", "")
    if not topic:
        topic = state["query"]

    result = await get_news(topic)
    return {"response": format_news_response(result)}


# ── Prompt Templates ──

AGENT_IDENTITY = """You are "Agentic AI" — a smart, friendly Weather & News AI Agent.
Your primary expertise is weather, climate, and real-time news.

IMPORTANT IDENTITY RULES:
- You are NOT a generic AI or large language model. You ARE "Agentic AI", a dedicated weather agent.
- If anyone asks "who are you", "what are you", "what's your name", or similar identity questions,
  always respond as: "I'm Agentic AI ⚡ — your personal weather and news assistant! 
  I can check real-time weather for any city or village worldwide, fetch the latest news headlines,
  and answer weather & climate questions like an expert. Just ask me!"
- Never say you are "a large language model" or "trained by Google" — stay in character.
- Always be enthusiastic, helpful, and use emojis to keep things engaging.
"""

WEATHER_PROMPT_TEMPLATE = AGENT_IDENTITY + """
You are answering a WEATHER-related question. Provide expert-level weather insights.

When answering:
- Provide current conditions if a city is mentioned
- Include practical tips (e.g., "carry an umbrella ☂️", "wear sunscreen 🧴")
- Give context about seasonal patterns
- If asked about a specific place, describe its typical weather
- Keep responses concise but informative
- Use weather emojis to make responses engaging (🌤️ 🌧️ 🌡️ ❄️ 💨 etc.)

Question: {query}"""

GENERAL_PROMPT_TEMPLATE = AGENT_IDENTITY + """
You are answering a general question. While your specialty is weather and news,
you can still help with other questions — just always stay in character as Agentic AI.

Answer clearly and concisely. If the question relates to weather or climate, 
provide expert-level insights. For other topics, be helpful but brief.

Question: {query}"""


def _get_prompt_for_query(query: str) -> str:
    """Choose the right prompt template based on query content."""
    query_lower = query.lower()
    weather_terms = ["weather", "climate", "temperature", "rain", "forecast",
                     "humid", "wind", "sunny", "cold", "hot", "storm", "snow",
                     "season", "monsoon", "cyclone", "tornado"]
    if any(term in query_lower for term in weather_terms):
        return WEATHER_PROMPT_TEMPLATE.format(query=query)
    return GENERAL_PROMPT_TEMPLATE.format(query=query)


async def gemini_response(state: AgentState) -> dict:
    """
    Node 2c: Answer general questions using Gemini.
    Uses weather-focused prompt for weather queries,
    general prompt for everything else.
    """
    if state.get("response"):
        return {}

    query = state["query"]

    try:
        prompt = _get_prompt_for_query(query)
        answer = await _call_gemini(prompt)
        return {"response": f"🤖 AI Response:\n\n{answer}"}

    except Exception as e:
        error_str = str(e)
        if _is_rate_limit_error(error_str):
            remaining = int(_gemini_cooldown_until - time.time()) if _is_gemini_on_cooldown() else COOLDOWN_SECONDS
            return {
                "response": (
                    f"⚠️ Gemini AI is rate-limited (free tier: ~15 requests/min).\n"
                    f"Please try again in ~{remaining} seconds.\n\n"
                    "💡 Meanwhile, these work instantly without any limits:\n"
                    '  • "Weather in Chennai"\n'
                    '  • "Weather in Cuddalore"\n'
                    '  • "Latest news in India"\n'
                    '  • "News about technology"'
                )
            }
        return {"response": f"❌ AI Error: {str(e)}"}


# ──────────────────────────────────────────────
# Conditional Router
# ──────────────────────────────────────────────

def route_by_intent(state: AgentState) -> Literal["weather_tool", "news_tool", "gemini_response"]:
    """Route to the appropriate tool based on detected intent."""
    intent = state.get("intent", "other")
    if intent == "weather":
        return "weather_tool"
    elif intent == "news":
        return "news_tool"
    else:
        return "gemini_response"


# ──────────────────────────────────────────────
# Build the LangGraph
# ──────────────────────────────────────────────

def build_agent_graph() -> StateGraph:
    """
    Construct and compile the LangGraph.

    Graph:
        start → detect_intent → [weather_tool | news_tool | gemini_response] → end
    """
    graph = StateGraph(AgentState)

    graph.add_node("detect_intent", detect_intent)
    graph.add_node("weather_tool", weather_tool)
    graph.add_node("news_tool", news_tool)
    graph.add_node("gemini_response", gemini_response)

    graph.set_entry_point("detect_intent")

    graph.add_conditional_edges(
        "detect_intent",
        route_by_intent,
        {
            "weather_tool": "weather_tool",
            "news_tool": "news_tool",
            "gemini_response": "gemini_response",
        },
    )

    graph.add_edge("weather_tool", END)
    graph.add_edge("news_tool", END)
    graph.add_edge("gemini_response", END)

    return graph.compile()


agent_graph = build_agent_graph()

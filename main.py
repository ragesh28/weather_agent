"""
Agentic AI — FastAPI Application
POST /ask endpoint that routes queries through a LangGraph workflow.

Run with:
    uvicorn main:app --reload
"""

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from graph import agent_graph


# ──────────────────────────────────────────────
# Load Environment Variables
# ──────────────────────────────────────────────

load_dotenv(override=True)


# ──────────────────────────────────────────────
# Pydantic Models
# ──────────────────────────────────────────────

class AskRequest(BaseModel):
    """Request body for the /ask endpoint."""
    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="The user's question or query.",
        json_schema_extra={"examples": ["Weather in Chennai", "Latest news in India", "What is quantum computing?"]},
    )


class AskResponse(BaseModel):
    """Response body from the /ask endpoint."""
    query: str = Field(description="The original user query.")
    intent: str = Field(description="Detected intent: weather, news, or other.")
    response: str = Field(description="The formatted response from the appropriate tool or LLM.")


class HealthResponse(BaseModel):
    """Response body for the /health endpoint."""
    status: str = "ok"
    service: str = "Agentic AI"
    version: str = "1.0.0"


# ──────────────────────────────────────────────
# App Lifespan — Startup Checks
# ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup checks and log configuration status."""
    print("=" * 50)
    print("🚀 Agentic AI — Starting up...")
    print("=" * 50)

    # Check for required API keys
    keys = {
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
        "WEATHER_API_KEY": os.getenv("WEATHER_API_KEY"),
        "NEWS_API_KEY": os.getenv("NEWS_API_KEY"),
    }

    for name, value in keys.items():
        status = "✅ Set" if value else "❌ Missing"
        print(f"  {name}: {status}")

    missing = [k for k, v in keys.items() if not v]
    if missing:
        print(f"\n⚠️  Warning: Missing keys: {', '.join(missing)}")
        print("   Some features will not work. Copy .env.example to .env and add your keys.")

    print("\n✅ Server ready!")
    print("   🌐 Web UI:  http://localhost:8000")
    print("   📡 API:     POST http://localhost:8000/ask")
    print("   📖 Docs:    http://localhost:8000/docs")
    print("=" * 50)

    yield  # App runs here

    print("\n👋 Agentic AI — Shutting down.")


# ──────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────

app = FastAPI(
    title="Agentic AI",
    description=(
        "An AI agent that routes user queries to the right tool:\n"
        "- **Weather** queries → OpenWeatherMap API\n"
        "- **News** queries → NewsAPI\n"
        "- **Other** questions → Google Gemini LLM\n\n"
        "Built with FastAPI + LangGraph + Gemini 2.0 Flash."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.post("/ask", response_model=AskResponse, summary="Ask the AI agent")
async def ask(request: AskRequest):
    """
    Send a query to the AI agent. The agent will:
    1. Detect the intent (weather / news / other)
    2. Route to the appropriate tool or LLM
    3. Return a formatted response

    **Examples:**
    - "Weather in Chennai" → Weather data
    - "Latest news in India" → Top 5 news headlines
    - "What is quantum computing?" → Gemini LLM response
    """
    try:
        initial_state = {
            "query": request.query,
            "intent": "",
            "city": "",
            "cities": [],
            "topic": "",
            "response": "",
        }

        # Invoke the LangGraph workflow
        result = await agent_graph.ainvoke(initial_state)

        return AskResponse(
            query=result["query"],
            intent=result.get("intent", "unknown"),
            response=result.get("response", "No response generated."),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Agent processing failed: {str(e)}",
        )


@app.get("/health", response_model=HealthResponse, summary="Health check")
async def health():
    """Check if the service is running."""
    return HealthResponse()


@app.get("/", include_in_schema=False)
async def root():
    """Serve the web UI."""
    return FileResponse("static/index.html")


# Mount static files (CSS, JS, images etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ──────────────────────────────────────────────
# Location Search API (Geocoding)
# ──────────────────────────────────────────────

@app.get("/search-location", summary="Search for a location")
async def search_location(q: str):
    """
    Search for a city, town, or village using Nominatim (OpenStreetMap) geocoding.
    Returns up to 5 matching locations with name, state, country, and coordinates.
    Useful for finding small villages and rural areas.
    """
    import httpx

    results = []

    async with httpx.AsyncClient(timeout=10.0) as client:
        # Search using Nominatim
        headers = {"User-Agent": "TaskWeatherAgent/1.0"}
        params = {"q": q, "format": "json", "limit": 5, "addressdetails": 1}
        response = await client.get("https://nominatim.openstreetmap.org/search", params=params, headers=headers)

        if response.status_code == 200:
            data = response.json()
            for loc in data:
                name = loc.get("name", "")
                address = loc.get("address", {})
                
                # Extract state and country cleanly from addressdetails
                state = address.get("state", address.get("county", ""))
                country = address.get("country_code", "").upper()
                
                # Use display_name for the label if state is missing
                label = f"{name}, {state}" if state else name
                if not state and loc.get("display_name"):
                    parts = loc["display_name"].split(", ")
                    if len(parts) > 2:
                        label = f"{parts[0]}, {parts[-2]}"
                
                # Avoid exact duplicates
                if not any(r["name"] == name and r["state"] == state and r["country"] == country for r in results):
                    results.append({
                        "name": name,
                        "state": state,
                        "country": country,
                        "lat": float(loc.get("lat", 0)),
                        "lon": float(loc.get("lon", 0)),
                        "label": label,
                    })

    return {"query": q, "results": results[:5]}


# ──────────────────────────────────────────────
# Run directly (alternative to uvicorn CLI)
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

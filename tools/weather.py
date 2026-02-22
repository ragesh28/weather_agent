"""
Weather Tool — OpenWeatherMap Integration
Fetches current weather data for a given city using the free tier API.
Features:
- Geocoding fallback for small/rural towns
- Spelling suggestions when a city isn't found
- Multi-city support
"""

import os
import httpx

# Open-Meteo API endpoints
OPEN_METEO_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
NOMINATIM_GEO_URL = "https://nominatim.openstreetmap.org/search"


async def _fetch_weather_by_coords(lat: float, lon: float, client: httpx.AsyncClient) -> httpx.Response:
    """Fetch weather using Open-Meteo with latitude/longitude coordinates."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m",
        "wind_speed_unit": "ms",
        "timezone": "auto"
    }
    return await client.get(OPEN_METEO_WEATHER_URL, params=params)


async def _geocode_city(city: str, client: httpx.AsyncClient) -> list:
    """
    Use Nominatim (OpenStreetMap) Geocoding API to find coordinates for a city or village.
    Nominatim has vastly more exact village data globally than Open-Meteo.
    """
    results_all = []

    # Nominatim requires a User-Agent header
    headers = {"User-Agent": "TaskWeatherAgent/1.0"}
    params = {"q": city, "format": "json", "limit": 5, "addressdetails": 0}
    response = await client.get(NOMINATIM_GEO_URL, params=params, headers=headers)
    
    if response.status_code == 200:
        results_all = response.json()

    return results_all


async def _get_weather_for_city(city: str, client: httpx.AsyncClient) -> dict:
    """
    Get weather for a single city/village using Nominatim geocoding and Open-Meteo forecast.
    """
    print(f"  📍 Looking up coordinates for '{city}' via Nominatim OpenStreetMap...")

    # Step 1: ALWAYS Geocode first (since Open-Meteo requires coords)
    geo_results = await _geocode_city(city, client)

    if geo_results:
        # Use the best match
        best = geo_results[0]
        geo_name = best.get("name", city)
        location_label = best.get("display_name", geo_name)
        lat = float(best["lat"])
        lon = float(best["lon"])
        geo_country = "N/A" # Nominatim format='json' doesn't auto-split country cleanly without addressdetails=1

        # We can extract state roughly from display_name if needed
        parts = location_label.split(', ')
        geo_state = parts[-2] if len(parts) > 2 else ""
        if len(parts) > 0:
            geo_country = parts[-1]

        print(f"  ✅ Geocoded to: {location_label} ({lat}, {lon})")

        # Check if the geocoded name differs from input (spelling correction)
        if len(geo_name) > 3 and len(city) > 3:
            # Simple check if there's a strong string difference to avoid noisy corrections
            is_correction = geo_name.lower() != city.lower() and city.lower() not in geo_name.lower()
        else:
            is_correction = geo_name.lower() != city.lower()

        # Step 2: Fetch weather by coordinates from Open-Meteo
        response = await _fetch_weather_by_coords(lat, lon, client)

        if response.status_code == 200:
            data = response.json()
            weather_info = _extract_weather_info(data, geo_name, geo_country)

            # If it was a spelling correction, note it
            if is_correction:
                weather_info["corrected_from"] = city
                weather_info["corrected_to"] = geo_name

            # Build suggestions from other geo results
            suggestions = []
            for geo in geo_results[1:4]:
                name = geo.get("name", "")
                label = geo.get("display_name", name)
                if label and label != location_label and label not in suggestions:
                    suggestions.append(label)

            return {"success": True, "data": weather_info, "suggestions": suggestions}
        else:
            return {"success": False, "error": f"Open-Meteo API error {response.status_code}: {response.text}"}
    else:
        # No geocoding results at all — suggest common cities
        return {
            "success": False,
            "error": f"Location '{city}' not found.",
            "suggestions": _get_spelling_suggestions(city),
        }


def _get_wmo_description(code: int) -> str:
    """Convert WMO weather codes to human-readable descriptions."""
    code_map = {
        0: "Clear Sky",
        1: "Mainly Clear", 2: "Partly Cloudy", 3: "Overcast",
        45: "Fog", 48: "Depositing Rime Fog",
        51: "Light Drizzle", 53: "Moderate Drizzle", 55: "Dense Drizzle",
        56: "Light Freezing Drizzle", 57: "Dense Freezing Drizzle",
        61: "Slight Rain", 63: "Moderate Rain", 65: "Heavy Rain",
        66: "Light Freezing Rain", 67: "Heavy Freezing Rain",
        71: "Slight Snowfall", 73: "Moderate Snowfall", 75: "Heavy Snowfall",
        77: "Snow Grains",
        80: "Slight Rain Showers", 81: "Moderate Rain Showers", 82: "Violent Rain Showers",
        85: "Slight Snow Showers", 86: "Heavy Snow Showers",
        95: "Thunderstorm",
        96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail"
    }
    return code_map.get(code, "Unknown Condition")


def _extract_weather_info(data: dict, city_name: str, country_code: str) -> dict:
    """Extract structured weather info from Open-Meteo response."""
    current = data.get("current", {})
    return {
        "city": city_name,
        "country": country_code,
        "temperature_celsius": current.get("temperature_2m", "N/A"),
        "feels_like_celsius": current.get("apparent_temperature", "N/A"),
        "description": _get_wmo_description(current.get("weather_code", -1)),
        "humidity_percent": current.get("relative_humidity_2m", "N/A"),
        "wind_speed_mps": current.get("wind_speed_10m", "N/A"),
    }


# Common Indian city names for spelling suggestions
COMMON_CITIES = [
    "Chennai", "Mumbai", "Delhi", "Kolkata", "Bangalore", "Hyderabad",
    "Ahmedabad", "Pune", "Jaipur", "Lucknow", "Kanpur", "Nagpur",
    "Indore", "Thane", "Bhopal", "Visakhapatnam", "Vadodara", "Coimbatore",
    "Madurai", "Trichy", "Salem", "Tirunelveli", "Cuddalore", "Pondicherry",
    "Kochi", "Thiruvananthapuram", "Mangalore", "Mysore", "Hubli",
    "Chandigarh", "Guwahati", "Patna", "Ranchi", "Bhubaneswar",
    "London", "New York", "Tokyo", "Paris", "Dubai", "Singapore",
]


def _get_spelling_suggestions(misspelled: str) -> list:
    """Return likely city name suggestions for a misspelled city."""
    misspelled_lower = misspelled.lower()
    suggestions = []

    for city in COMMON_CITIES:
        city_lower = city.lower()
        # Check if first 3+ characters match
        if len(misspelled_lower) >= 3 and city_lower.startswith(misspelled_lower[:3]):
            suggestions.append(city)
        # Check if the city contains the misspelled word or vice versa
        elif misspelled_lower in city_lower or city_lower in misspelled_lower:
            suggestions.append(city)
        # Simple edit distance: check if most characters match
        elif len(misspelled_lower) >= 4:
            matching = sum(1 for a, b in zip(misspelled_lower, city_lower) if a == b)
            if matching >= len(misspelled_lower) * 0.6:
                suggestions.append(city)

    return suggestions[:5]


async def get_weather(city: str) -> dict:
    """
    Fetch current weather for one or more cities.
    Supports multi-city queries separated by "and", ",", "&".
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            result = await _get_weather_for_city(city, client)
            return result

    except httpx.TimeoutException:
        return {"success": False, "error": f"Request timed out for '{city}'. Try again."}
    except httpx.ConnectError:
        return {"success": False, "error": "Could not connect to Open-Meteo."}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {str(e)}"}


async def get_weather_multi(cities: list[str]) -> list[dict]:
    """Fetch weather for multiple cities simultaneously using Open-Meteo."""
    results = []
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            for city in cities:
                result = await _get_weather_for_city(city.strip(), client)
                results.append(result)
    except Exception as e:
        results.append({"success": False, "error": f"Error: {str(e)}"})

    return results


def format_weather_response(result: dict) -> str:
    """Format a single weather result into a human-readable string."""
    if not result.get("success"):
        error_msg = f"❌ Weather Error: {result.get('error', 'Unknown error')}"
        suggestions = result.get("suggestions", [])
        if suggestions:
            error_msg += f"\n\n💡 Did you mean: {', '.join(suggestions)}?"
        return error_msg

    d = result["data"]
    response = ""

    # Show correction notice if spelling was fixed
    if d.get("corrected_from"):
        response += f"💡 Showing weather for '{d['corrected_to']}' (you searched '{d['corrected_from']}')\n\n"

    response += (
        f"🌤️ Weather in {d['city']}, {d['country']}:\n"
        f"  🌡️ Temperature: {d['temperature_celsius']}°C (feels like {d['feels_like_celsius']}°C)\n"
        f"  ☁️ Condition: {d['description']}\n"
        f"  💧 Humidity: {d['humidity_percent']}%\n"
        f"  💨 Wind Speed: {d['wind_speed_mps']} m/s"
    )

    suggestions = result.get("suggestions", [])
    if suggestions:
        response += f"\n\n📍 Other nearby: {', '.join(suggestions)}"

    return response


def format_multi_weather_response(results: list[dict]) -> str:
    """Format multiple weather results into a combined response."""
    parts = []
    for i, result in enumerate(results):
        parts.append(format_weather_response(result))

    return "\n\n" + "─" * 30 + "\n\n".join(parts)

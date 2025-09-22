import os
import json
import httpx
from typing import Any
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

# Initialize MCP server
mcp = FastMCP("WeatherServer")

load_dotenv()

# OpenWeather API configuration
OPENWEATHER_API_BASE = "https://api.openweathermap.org/data/2.5/weather"
API_KEY = os.getenv("OPENWEATHER_API_KEY")
USER_AGENT = "weather-app/1.0"


async def fetch_weather(city: str) -> dict[str, Any] | None:
    """
    Fetch weather information from OpenWeather API.

    :param city: City name (must use English, e.g., Beijing)
    :return: Weather data dictionary, or dictionary containing error info if failed
    """
    params = {
        "q": city,
        "appid": API_KEY,
        "units": "metric",
        "lang": "zh_cn"
    }
    headers = {"User-Agent": USER_AGENT}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                OPENWEATHER_API_BASE,
                params=params,
                headers=headers,
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()  # Return as dictionary
        except httpx.HTTPStatusError as e:
            return {"error": f"HTTP error: {e.response.status_code}"}
        except Exception as e:
            return {"error": f"Request failed: {str(e)}"}


def format_weather(data: dict[str, Any] | str) -> str:
    """
    Format weather data into a human-readable string.

    :param data: Weather data (can be dict or JSON string)
    :return: Formatted weather information string
    """

    # If input is a string, try to parse it as JSON
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception as e:
            return f"Unable to parse weather data: {e}"

    # If error is included in the data, return it directly
    if "error" in data:
        return f"⚠️ {data['error']}"

    # Extract weather fields with defaults
    city = data.get("name", "Unknown")
    country = data.get("sys", {}).get("country", "Unknown")
    temp = data.get("main", {}).get("temp", "N/A")
    humidity = data.get("main", {}).get("humidity", "N/A")
    wind_speed = data.get("wind", {}).get("speed", "N/A")

    weather_list = data.get("weather", [{}])
    description = weather_list[0].get("description", "Unknown")

    return (
        f"🌍 {city}, {country}\n"
        f"🌡️ Temperature: {temp}°C\n"
        f"💧 Humidity: {humidity}%\n"
        f"🌬️ Wind Speed: {wind_speed} m/s\n"
        f"☁️ Weather: {description}\n"
    )


@mcp.tool()
async def query_weather(city: str) -> str:
    """
    Enter the English name of a city to query today's weather.

    :param city: City name (must be in English)
    :return: Formatted weather information
    """
    data = await fetch_weather(city)
    return format_weather(data)


if __name__ == "__main__":
    # Run MCP server via standard I/O
    mcp.run(transport='stdio')

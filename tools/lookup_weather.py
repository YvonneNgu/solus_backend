import logging
import aiohttp
from typing import Dict, Any
from langfuse.client import StatefulClient

logger = logging.getLogger("openai-video-agent")

async def lookup_weather(
    context,
    location: str,
    get_current_trace,
) -> Dict[str, Any]:
    """Look up weather information for a given location.
    
    Args:
        location: The location to look up weather information for.
    """
    logger.info(f"Getting weather for {location}")
    
    # Create a span in Langfuse for tracking
    span = get_current_trace().span(name="weather_lookup", metadata={"location": location})
    
    try:
        # Use wttr.in API to get weather data
        url = f"https://wttr.in/{location}?format=%C+%t"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    weather_data = await response.text()
                    result = {
                        "condition": weather_data.split()[0],
                        "temperature": weather_data.split()[1],
                        "location": location
                    }
                    logger.info(f"Weather result: {result}")
                    return result
                else:
                    error_msg = f"Failed to get weather data, status code: {response.status}"
                    logger.error(error_msg)
                    span.update(level="ERROR")
                    return {"error": error_msg}
    except Exception as e:
        error_msg = f"Weather lookup error: {str(e)}"
        logger.error(error_msg)
        span.update(level="ERROR")
        return {"error": error_msg}
    finally:
        span.end()

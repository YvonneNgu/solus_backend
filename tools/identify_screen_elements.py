import logging
import asyncio
import base64
import os
import json
from datetime import UTC, datetime
from typing import Dict, Any, List
from google import genai
from google.genai import types
from livekit.agents.utils.images.image import encode, EncodeOptions

logger = logging.getLogger("IDENTIFY ELEMENT TOOL")

async def identify_screen_elements(
    most_recent_frame,
    session,
) -> Dict[str, Any]:
    """Identify interactive elements and their exact positions on the user's screen.
    
    This tool analyzes the most recent screen capture to identify interactive elements along with their positions on screen.
    """
    logger.info("Identifying interactive elements on screen")
    
    try:
        
        if not most_recent_frame:
            logger.warning("No recent frame available")
            return {
                "success": False,
                "error": "No recent screen capture available. Ask the user to share their screen.",
                "elements": []
            }            

        # tell the user that the agent is analyzing the screen, have to wait
        session.say(
            text="I am analyzing the screen... Please give me a minute..."
        )

        # Encode the VideoFrame to JPEG bytes
        options = EncodeOptions(format="JPEG", quality=95)
        image_bytes = encode(most_recent_frame, options)

        # Encode as base64
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')

        # Initialize Gemini client
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        model = "gemini-2.5-flash"

        # Update prompt to match the style used in the TypeScript code
        prompt = """Give the segmentation masks for the interactive components. Output a JSON list of segmentation masks where each entry contains the 2D bounding box in the key "box_2d", and the text label in the key "label". Use descriptive labels. Do not return masks."""

        # Create content for Gemini
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(
                        mime_type="image/jpeg",
                        data=image_b64,
                    ),
                    types.Part.from_text(text=prompt),
                ],
            ),
        ]
        
        # Configure generation
        generate_content_config = types.GenerateContentConfig(
            max_output_tokens=8192,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            response_mime_type="text/plain",
        )
        
        # Generate content
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=model,
            contents=contents,
            config=generate_content_config,
        )
        
        # Process and structure the response
        raw_response = response.text
        
        logger.info(f"Received Gemini analysis: {raw_response[:100]}...")
       
        # Extract JSON content from the response (it might be wrapped in markdown code blocks)
        json_content = raw_response
        if "```json" in raw_response:
            json_content = raw_response.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_response:
            json_content = raw_response.split("```")[1].strip()
            
        # Parse the JSON string into a Python structure
        try:
            # Parse the JSON content into a Python list of dictionaries
            parsed_elements = json.loads(json_content)
            
            # Log the number of elements found
            logger.info(f"Successfully parsed {len(parsed_elements)} UI elements from response")
            
            return {
                "success": True,
                "interactive_ui_components": parsed_elements,  # Return parsed Python objects instead of JSON string
                "timestamp": datetime.now(UTC).isoformat()
            }
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to parse screen elements: {str(e)}",
                "raw_json": json_content,
                "interactive_ui_components": []
            }
        
    except Exception as e:
        error_msg = f"Screen element identification error: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "interactive_ui_components": []
        }

def filter_mask_content(raw_response: str) -> List[Dict[str, Any]]:
    """Filter out the mask content from the Gemini API response.
    
    Args:
        raw_response: The raw response text from Gemini API
        
    Returns:
        List of dictionaries with filtered element data (no mask content)
    """
    try:
        # Extract JSON content from the response (it might be wrapped in markdown code blocks)
        json_content = raw_response
        if "```json" in raw_response:
            json_content = raw_response.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_response:
            json_content = raw_response.split("```")[1].strip()
        
        # Parse the JSON content
        elements = json.loads(json_content)
        
        # Filter out mask content
        filtered_elements = []
        for element in elements:
            filtered_element = {
                "box_2d": element.get("box_2d", []),
                "label": element.get("label", "Unknown element")
            }
            filtered_elements.append(filtered_element)
        
        logger.info(f"Filtered {len(filtered_elements)} UI elements from response")
        return filtered_elements
    except Exception as e:
        logger.error(f"Error filtering mask content: {str(e)}")
        return []

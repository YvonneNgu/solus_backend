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
) -> str:
    """Identify interactive elements and their exact positions on the user's screen.
    
    This tool analyzes the most recent screen capture to identify interactive elements along with their positions on screen.
    Returns only the interactive UI components as text.
    """
    logger.info("Identifying interactive elements on screen")
    
    try:
       
        # tell the user that the agent is analyzing the screen, have to wait
        await session.say(
            text="Let me analyze the screen... ",
            add_to_chat_ctx=False
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
        
        # Retry logic for JSON parsing with maximum 2 retries
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                # Generate content
                response = await asyncio.to_thread(
                    client.models.generate_content,
                    model=model,
                    contents=contents,
                    config=generate_content_config,
                )
                
                # Process and structure the response
                raw_response = response.text
                
                logger.info(f"Received Gemini analysis (attempt {attempt + 1}): {raw_response[:100]}...")
               
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
                    
                    # Filter out unwanted 'mask' field from each element if present
                    # Keep only 'box_2d' and 'label' fields as these are the required fields
                    filtered_elements = []
                    for element in parsed_elements:
                        if isinstance(element, dict):
                            # Create new dict with only required fields
                            filtered_element = {}
                            if 'box_2d' in element:
                                filtered_element['box_2d'] = element['box_2d']
                            if 'label' in element:
                                filtered_element['label'] = element['label']
                            
                            # Only add element if it has the required fields
                            if 'box_2d' in filtered_element and 'label' in filtered_element:
                                filtered_elements.append(filtered_element)
                            else:
                                logger.warning(f"Skipping element missing required fields: {element}")
                    
                    # Log the number of elements found after filtering
                    logger.info(f"Successfully parsed and filtered {len(filtered_elements)} UI elements from response")
                    
                    # Convert filtered elements to string and return
                    return str(filtered_elements)
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response on attempt {attempt + 1}: {str(e)}")
                    
                    # If this is not the last attempt, continue to retry
                    if attempt < max_retries:
                        # tell the user that the agent is analyzing the screen, have to wait
                        await session.say(
                            text="I had some problems. Let me try again... ",
                            add_to_chat_ctx=False
                        )
                        logger.info(f"Retrying content generation (attempt {attempt + 2}/{max_retries + 1})")
                        continue
                    else:
                        # Last attempt failed, return raw response as text
                        logger.warning("All JSON parsing attempts failed, returning raw response")
                        return raw_response
                        
            except Exception as e:
                logger.error(f"Error generating content on attempt {attempt + 1}: {str(e)}")
                
                # If this is not the last attempt, continue to retry
                if attempt < max_retries:
                    logger.info(f"Retrying due to generation error (attempt {attempt + 2}/{max_retries + 1})")
                    continue
                else:
                    # Last attempt failed with error
                    raise e
        
    except Exception as e:
        error_msg = f"Screen element identification error: {str(e)}"
        logger.error(error_msg)
        return f"{error_msg}"
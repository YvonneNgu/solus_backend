import json
import logging
from datetime import UTC, datetime
from typing import Dict, List, Any

from livekit import rtc
from langfuse.client import StatefulClient

logger = logging.getLogger("openai-video-agent")

async def display_instructions(
    context,
    instruction_text: str,
    instruction_speech: str,
    bounding_box: List[int],
    visual_cue_type: str,
    session,
    room,
    get_current_trace
) -> Dict[str, Any]:
    """Display navigation guidance with visual cues on the user's screen.
    
    Args:
        context: The run context
        instruction_text: Simple text instruction to display to the user, e.g. "Tap here"
        instruction_speech: Spoken instruction to guide the user, e.g. "Tap the menu icon in the top right corner"
        bounding_box: Bounding box coordinates originally from the identify_screen_elements tool
        visual_cue_type: Type of visual cue to display (default: "arrow")
        session: The agent session
        room: The LiveKit room
        get_current_trace: Function to get the current trace
    """
    span = get_current_trace().span(
        name="display_instructions",
        metadata={
            "instruction_text": instruction_text,
            "instruction_speech": instruction_speech,
            "bounding_box": bounding_box,
            "visual_cue_type": visual_cue_type
        }
    )
    
    try:
        logger.info(f"Instruction: {instruction_text}, {instruction_speech}")
        logger.info(f"Bounding box: {bounding_box}")
        logger.info(f"Visual cue type: {visual_cue_type}")

        # Prepare the payload for the frontend
        payload = {
            "instruction_text": instruction_text,
            "bounding_box": bounding_box,
            "visual_cue_type": visual_cue_type,
            "timestamp": datetime.now(UTC).isoformat()
        }

        # Get the first remote participant (assuming single user scenario)
        remote_participants = list(room.remote_participants.values())
        if not remote_participants:
            logger.error("No remote participants found for navigation guidance")
            span.update(level="ERROR")
            return {
                "success": False,
                "error": "No remote participants available",
                "payload": payload
            }

        target_participant = remote_participants[0]
        logger.info(f"Sending navigation guidance to participant: {target_participant.identity}")

        try:
            # Perform RPC call to frontend
            response = await room.local_participant.perform_rpc(
                destination_identity=target_participant.identity,
                method="display-navigation-guidance",
                payload=json.dumps(payload),
                response_timeout=5.0  # 5 second timeout for UI operations
            )
            
            # received by frontend, tts the instruction at the same time
            logger.info(f"Navigation guidance RPC successful. Response: {response}")
            #session.say(instruction_speech)
            span.update(level="DEFAULT")
            
            return {
                "success": True,
                "response": response,
                "payload": payload
            }

        except rtc.RpcError as rpc_error:
            error_msg = f"RPC call failed: {rpc_error.code} - {rpc_error.message}"
            logger.error(error_msg)
            span.update(level="ERROR")
            
            return {
                "success": False,
                "error": error_msg,
                "error_code": rpc_error.code,
                "payload": payload
            }

    except Exception as e:
        error_msg = f"Failed to display navigation guidance: {str(e)}"
        logger.error(error_msg)
        span.update(level="ERROR")
        return {
            "success": False,
            "error": error_msg,
            "payload": payload if 'payload' in locals() else None
        }
    finally:
        span.end()

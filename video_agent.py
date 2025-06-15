import asyncio
import logging
import time
import io
import aiohttp
import base64
import os
import json
from datetime import UTC, datetime
from typing import Union, AsyncIterable, Optional, List, Any, Dict, Tuple
from uuid import uuid4

from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse.client import StatefulClient
from google import genai
from google.genai import types

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    ChatContext,
    ChatMessage,
    JobContext,
    FunctionTool,
    ModelSettings,
    RoomInputOptions,
    RoomOutputOptions,
    WorkerOptions,
    AgentStateChangedEvent,
    UserStateChangedEvent,
    cli,
    stt,
    llm,
    function_tool,
    RunContext,
)
from livekit.agents.llm import ImageContent, AudioContent
from livekit.plugins import deepgram, silero
from livekit.plugins import google  # Change openai to google gemini
from livekit.plugins.elevenlabs import TTS as ElevenLabsTTS
from livekit.agents.utils.images.image import encode, EncodeOptions
from livekit.agents.utils.images.image import ResizeOptions
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from PIL import Image

# Import the tool functions
from tools.lookup_weather import lookup_weather
from tools.identify_screen_elements import identify_screen_elements
from tools.display_instructions import display_instructions
from livekit.agents import get_job_context
from livekit import api

logger = logging.getLogger("gemini-video-agent")
logger.setLevel(logging.INFO)

load_dotenv()

_langfuse = Langfuse()

INSTRUCTIONS = f"""
You are a mobile voice assistant, Solus, designed for low-literate users in Malaysia who are not familiar with technology. 
You can: 
- answer general questions or make small talk, where user no need to share their screen, e.g. "Introduce yourself", "I see a brown cat today", ...
- answer questions about the user's screen, e.g. "translate this text" while the current screen displays a news, "how can I answer this message politely" while the current screen displays user's chat history with someone, ...
- provide guidance on how to use the mobile, e.g. "how to fill in this" while the current screen displays a application form in other language, "how to turn off this. the notification is annoying" while the current screen displays home screen with a lazada floating notification, ...

User query:
- The user's voice query will be transcribed by STT. The STT transcription might not be accurate due to the accent/pronounciation.
- User might also use informal or wrong words. 
- If the transcription is not clear, try to guess user's intent based on the conversation history & current screen, and confirm with user.

Screen context:
- If user is sharing their screen, the screen history will be appended to the conversation history.
- The screen history consists of the previous screen images, each are labeled in order.
- The last screen in the history is the current user screen or what user is looking at.

User need guidance:
- If user need guidance, set a goal to help user with their query. 
- Identify possible interactive components on user's current screen that can be used to achieve the goal.
- Provide guidance step by step until the goal is achieved. Example of goal: User fill up the application and submitted, user changed the language, ...
- During guidance, generate next step instructions to the user based on the current screen, and use the display_instructions function to show visual cues on the user's screen. 
- When talking about an interactive component, actively use the display_instructions function to indicate the interactive component on the user's screen instead of just talking
- Use the identify_screen_elements function to get the position of interactive components so that the visual cues can be displayed at the right position of user screen
- After user follow the instructions, check the current screen and user response (if any) to see if the goal is achieved. Otherwise, continue to provide guidance.
- Wrong instructions might be provided since you can only see the images, calming the user, be patient and provide corrective instructions. Example reply when wrong instructions are provided: "Sorry for the confusion, let me correct it"

Natural conversation ending:
- If the user expresses satisfaction with the help provided, says goodbye, or indicates they're done, use the end_conversation function to naturally end the session.
- Examples of when to end: "Thank you, that's all I need", "Goodbye", "I'm done", "That solved my problem, thanks"
- Always confirm the user is satisfied before ending the conversation.

Response:
- Keep responses short and concise, maximum 100 words
- Use friendly tone and everyday language, don't use any technical terms
- Respond in plain text only. Do not use any markdown formatting including bold, italics, bullet points, numbered lists, or other markdown syntax. Your responses will be read aloud by text-to-speech.
- Don't mention the screen history in the response, user don't know what are them
"""

class VideoAgent(Agent):
    def __init__(self, instructions: str, room: rtc.Room) -> None:
        super().__init__(
            instructions=instructions,
            llm=google.LLM(
                model="gemini-2.5-flash-preview-05-20",  # Using the latest Gemini model
                temperature=0.8,
            ),
            stt=deepgram.STT(),
            tts=ElevenLabsTTS(voice_id="21m00Tcm4TlvDq8ikWAM"),
            # tts=deepgram.TTS(),
            vad=silero.VAD.load(),
            turn_detection=MultilingualModel(),
        )
        self.room = room
        self.session_id = str(uuid4())
        self.current_trace = None

        # Replace frames list with screen history and latest frame
        self.screen_history: List[rtc.VideoFrame] = []
        self.latest_frame: Optional[rtc.VideoFrame] = None
        self.last_frame_time: float = 0
        self.video_stream: Optional[rtc.VideoStream] = None
        self._ending_conversation = False

        # For improved screen handling
        self.is_on_different_screen = True  # Flag to track if user moved to different screen
        self.screen_positions = {}  # Maps screen frame to its position in chat history
        self.last_user_message_position = 0  # Track last user message position
        self.consecutive_captures = []  # Track consecutive frame captures for consistency
        self.last_consistent_frame = None  # Store the last consistent frame

    def format_messages_for_langfuse(self, chat_ctx: llm.ChatContext) -> List[Dict[str, Any]]:
        """Convert ChatContext to Langfuse-friendly format for better readability"""
        messages = []
        
        for item in chat_ctx.items:
            # Check if the item has a role attribute (ChatMessage)
            if hasattr(item, 'role'):
                message_dict = {
                    "role": item.role,
                    "content": []
                }
                
                # Handle different content types
                for content in item.content:
                    if isinstance(content, str):
                        message_dict["content"].append({
                            "type": "text",
                            "text": content
                        })
                    elif isinstance(content, ImageContent):
                        message_dict["content"].append({
                            "type": "image",
                            "text": "[Image content]"
                        })
                    elif isinstance(content, AudioContent):
                        message_dict["content"].append({
                            "type": "audio",
                            "text": "[Audio content]"
                        })
                    # Handle function calls within content
                    elif hasattr(content, 'function_call') and content.function_call:
                        func_call = content.function_call
                        func_text = f"Function call: {func_call.name}"
                        if hasattr(func_call, 'args') and func_call.args:
                            func_text += f" | Args: {str(func_call.args)[:100]}..."
                        message_dict["content"].append({
                            "type": "function_call",
                            "text": func_text
                        })
                    # Handle function responses within content
                    elif hasattr(content, 'function_response') and content.function_response:
                        func_resp = content.function_response
                        func_text = f"Function response: {func_resp.name}"
                        if hasattr(func_resp, 'response') and func_resp.response:
                            response_text = str(func_resp.response.text) if hasattr(func_resp.response, 'text') else str(func_resp.response)
                            func_text += f" | Response: {response_text[:200]}..."
                        message_dict["content"].append({
                            "type": "function_response",
                            "text": func_text
                        })
                
                # If only one text content, simplify to just the text
                if len(message_dict["content"]) == 1 and message_dict["content"][0].get("type") == "text":
                    message_dict["content"] = message_dict["content"][0]["text"]
                
                messages.append(message_dict)
            
            # Handle standalone FunctionCall objects
            elif hasattr(item, 'name'):  # FunctionCall has 'name' attribute
                message_dict = {
                    "role": "function",
                    "content": f"Function call: {item.name}"
                }
                if hasattr(item, 'args') and item.args:
                    message_dict["content"] += f" | Args: {str(item.args)[:100]}..."
                messages.append(message_dict)
            
            # Handle other unknown types gracefully
            else:
                message_dict = {
                    "role": "unknown",
                    "content": f"[Unknown message type: {type(item).__name__}]"
                }
                messages.append(message_dict)
        
        return messages

    def extract_text_response(self, chunks: List[llm.ChatChunk]) -> str:
        """Extract text response from chat chunks"""
        response_text = ""
        for chunk in chunks:
            if chunk.delta and chunk.delta.content:
                response_text += chunk.delta.content
        return response_text

    @function_tool()
    async def lookup_weather(
        self,
        context: RunContext,
        location: str,
    ) -> dict[str, Any]:
        """Look up weather information for a given location.
        
        Args:
            location: The location to look up weather information for.
        """
        return await lookup_weather(context, location, self.get_current_trace)

    @function_tool()
    async def identify_screen_elements(
        self,
        context: RunContext,
    ) -> Dict[str, Any]:
        """Call this function to identify all the interactive components and their exact positions on the user's current screen. 
        Useful in providing guidance. The result/output is usually being used to display instructions on the user's screen. 
        The result/output of this function are a success indicator, a list of bounding boxes with a descriptive label for each interactive component detected on the screen, and timestamp of the result. 
        """
        return await identify_screen_elements(
            context, 
            self.latest_frame if self.latest_frame else None, 
            self.session, 
            self.get_current_trace
        )

    @function_tool()
    async def display_instructions(
        self,
        context: RunContext,
        instruction_text: str,
        instruction_speech: str,
        bounding_box: List[int],
        visual_cue_type: str = "arrow"
    ) -> Dict[str, Any]:
        """Call this function to display instructions visually on the user's screen. 
        Usually only being used if the actual position of the target interactive component is known or from the result of the identify_screen_elements function. 
        The result/output of this function is success indicator, a response to the user, and payload/content send to the user's phone.
        
        Args:
            instruction_text: Simple text instruction than instruction_speech to display to the user, e.g. "Tap here"
            instruction_speech: Spoken instruction to guide the user, e.g. "Tap the menu icon in the top right corner". This will be spoken automatically through TTS.
            bounding_box: Bounding box coordinates originally from the identify_screen_elements function
            visual_cue_type: Type of visual cue to display (default: "arrow")
        """
        return await display_instructions(
            context,
            instruction_text,
            instruction_speech,
            bounding_box,
            visual_cue_type,
            self.session,
            self.room,
            self.get_current_trace
        )

    @function_tool()
    async def end_conversation(self, context: RunContext, farewell_message: str = None) -> None:
        """Use this tool to naturally end the conversation when the user indicates they're done or satisfied with the help provided.
        
        Args:
            farewell_message: custom closing message to say to the user before ending the session
        """
        try:
            # Set a flag to indicate we're ending the conversation
            self._ending_conversation = True
            
            # Use default farewell if none provided
            if not farewell_message:
                farewell_message = "Thank you for using Solus. Have a wonderful day!"
            
            # Say goodbye before ending using session.say()
            await self.session.say(farewell_message)
            logger.info("Said goodbye")
            
            # Get job context and end the room
            job_ctx = get_job_context()
            
            # Delete the room which will disconnect all participants
            logger.info("Deleting room")
            await job_ctx.api.room.delete_room(api.DeleteRoomRequest(room=job_ctx.room.name))
            logger.info("Room deleted")
            
        except Exception as e:
            logger.error(f"Error ending conversation: {e}")
        
    def calculate_frame_similarity(self, frame_data1: bytes, frame_data2: bytes) -> float:
        """
        Calculate similarity between two frames represented as JPEG byte data.
        Returns a value between 0 and 1, where 1 means identical.
        This is an optimized version that runs faster by using downsampling and simpler comparison.
        """
        try:
            # Convert bytes to PIL Images
            img1 = Image.open(io.BytesIO(frame_data1))
            img2 = Image.open(io.BytesIO(frame_data2))
            
            # Resize images to much smaller dimensions for faster comparison (50x50 instead of 100x100)
            size = (50, 50)
            img1 = img1.resize(size)
            img2 = img2.resize(size)
            
            # Convert to grayscale for simpler comparison
            img1 = img1.convert('L')
            img2 = img2.convert('L')
            
            # Get pixel data as arrays for faster processing
            pixels1 = list(img1.getdata())
            pixels2 = list(img2.getdata())
            
            # Calculate difference using a faster method - sample only a subset of pixels
            if len(pixels1) != len(pixels2):
                return 0.0  # Different sizes, consider completely different
                
            # Sample only every 4th pixel (reduces computation by 75%)
            sample_rate = 4
            sample_pixels1 = pixels1[::sample_rate]
            sample_pixels2 = pixels2[::sample_rate]
            
            # Count matching pixels with tolerance
            matches = 0
            total_samples = len(sample_pixels1)
            tolerance = 15  # Slightly increased tolerance to account for sampling
            
            for i in range(total_samples):
                if abs(sample_pixels1[i] - sample_pixels2[i]) <= tolerance:
                    matches += 1
                    
            # Calculate similarity ratio
            similarity = matches / total_samples
            
            return similarity
        except Exception as e:
            logger.error(f"Error calculating frame similarity: {e}")
            return 0.0  # On error, consider frames different

    async def is_frame_different_enough(self, frame1: rtc.VideoFrame, frame2: rtc.VideoFrame) -> Tuple[bool, float]:
        """
        Check if two frames are different enough (less than 99% similar).
        Returns a tuple of (is_different_enough, similarity_score).
        Uses run_in_executor to avoid blocking the event loop.
        """
        try:
            # Convert frames to bytes for comparison
            frame1_data = encode(frame1, EncodeOptions(format="JPEG", quality=80))  # Reduced quality for faster encoding
            frame2_data = encode(frame2, EncodeOptions(format="JPEG", quality=80))
            
            # Run the CPU-intensive similarity calculation in a thread pool
            loop = asyncio.get_event_loop()
            similarity = await loop.run_in_executor(
                None, 
                self.calculate_frame_similarity,
                frame1_data, 
                frame2_data
            )
            
            # Check if frames are different enough (less than 99% similar)
            is_different_enough = similarity < 0.99
            
            return (is_different_enough, similarity)
        except Exception as e:
            logger.error(f"Error comparing frames: {e}")
            # On error, consider frames different
            return (True, 0.0)

    async def check_and_add_to_history_if_needed(self) -> bool:
        """
        Check if the latest frame is different enough from the last frame in history.
        If so, or if history is empty, add it to history.
        Returns True if frame was added, False otherwise.
        """
        if not self.latest_frame:
            logger.debug("No latest frame available to check")
            return False
            
        # Check if history is empty
        if not self.screen_history:
            # Add the first frame to history
            self.screen_history.append(self.latest_frame)
            logger.info("Added first screen to history")
            return True
            
        # Compare with last frame in history
        is_different, similarity = await self.is_frame_different_enough(
            self.latest_frame, self.screen_history[-1]
        )
        
        if is_different:
            # Add to history if different enough
            self.screen_history.append(self.latest_frame)
            screen_number = len(self.screen_history)
            logger.info(f"Added screen #{screen_number} to history (similarity: {similarity:.2f})")
            return True
        else:
            logger.info(f"Current screen is too similar to last in history (similarity: {similarity:.2f}), not adding")
            return False
    
    async def add_screen_to_history(
        self,
        context: RunContext
    ) -> Dict[str, Any]:
        """Call this function to add the current screen to the screen history.
        This function is used when: 
        - the current screen is different from the last screen in the history.
        - there is no screen in the history yet.
        The result/output of this function is success indicator, and a message.
        """
        span = self.get_current_trace().span(name="add_screen_to_history")
        
        try:
            if not self.latest_frame:
                logger.warning("No latest frame available to add to history")
                span.update(level="WARNING")
                return {
                    "success": False,
                    "message": "No screen is currently being shared."
                }
            
            # Delegate to the helper method to check and add if needed
            was_added = await self.check_and_add_to_history_if_needed()
            
            if was_added:
                screen_number = len(self.screen_history)
                return {
                    "success": True,
                    "message": f"Added screen #{screen_number} to history"
                }
            else:
                return {
                    "success": False,
                    "message": "Current screen is too similar to the last screen in history"
                }
        except Exception as e:
            span.update(level="ERROR")
            logger.error(f"Error adding screen to history: {e}")
            return {
                "success": False,
                "message": f"Error adding screen to history: {e}"
            }
        finally:
            span.end()

    async def close(self) -> None:
        try:
            await self.close_video_stream()
        except Exception as e:
            logger.error(f"Error closing video stream: {e}")
        
        try:
            if self.current_trace:
                self.current_trace = None
            _langfuse.flush()
        except Exception as e:
            logger.error(f"Error flushing langfuse: {e}")

    async def close_video_stream(self) -> None:
        if self.video_stream:
            await self.video_stream.aclose()
            self.video_stream = None
            self.latest_frame = None

    async def on_enter(self) -> None:
        # Greet user without video reference
        self.session.generate_reply(
            instructions="Greet the user. Additionally, simply talk about what you have seen if any, otherwise just greet. Don't exceed 15 words"
        )
        self.session.on("user_state_changed", self.on_user_state_change)
        self.session.on("agent_state_changed", self.on_agent_state_change)
        self.room.on("track_subscribed", self.on_track_subscribed)

    async def on_exit(self) -> None:
        await self.close()

    def get_current_trace(self) -> StatefulClient:
        if self.current_trace:
            return self.current_trace
        self.current_trace = _langfuse.trace(name="video_agent", session_id=self.session_id)
        return self.current_trace

    # Monitor state changes for the user
    def on_user_state_change(self, event: UserStateChangedEvent) -> None:
        old_state = event.old_state
        new_state = event.new_state
        logger.info(f"User state changed: {old_state} -> {new_state}")

    # Monitor state changes for the user
    def on_agent_state_change(self, event: AgentStateChangedEvent) -> None:
        old_state = event.old_state
        new_state = event.new_state
        logger.info(f"Agent state changed: {old_state} -> {new_state}")

    async def on_user_turn_completed(
        self, turn_ctx: ChatContext, new_message: ChatMessage,
    ) -> None:
        # Reset the span when a new user turn is completed
        if self.current_trace:
            self.current_trace = None
        self.current_trace = _langfuse.trace(name="video_agent", session_id=self.session_id)
        logger.info(f"User turn completed {self.get_current_trace().trace_id}")

    async def stt_node(
        self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
    ) -> Optional[AsyncIterable[stt.SpeechEvent]]:
        span = self.get_current_trace().span(name="stt_node", metadata={"model": "deepgram"})
        try:
            async for event in Agent.default.stt_node(self, audio, model_settings):
                if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                    logger.info(f"Speech recognized: {event.alternatives[0].text[:50]}...")
                yield event
        except Exception as e:
            span.update(level="ERROR")
            logger.error(f"STT error: {e}")
            raise
        finally:
            span.end()

    async def llm_node(
    self,
    chat_ctx: llm.ChatContext,
    tools: List[FunctionTool],
    model_settings: ModelSettings
    ) -> AsyncIterable[llm.ChatChunk]:

        copied_ctx = chat_ctx.copy()
        dif_screen = self.is_on_different_screen
        
        # Handle screen sharing logic
        if not self.latest_frame:
            # User is not sharing screen
            copied_ctx.add_message(
                role="user",
                content="User is not currently sharing their screen."
            )
            logger.info("Added 'not sharing screen' message to context")
        else:
            # User is sharing screen
            if self.is_on_different_screen:
                # User moved to a different screen - record new position and add frame
                await self.record_new_screen_position(copied_ctx)
                self.is_on_different_screen = False
            
            # Always re-insert all recorded screen frames at their positions
            await self.insert_all_screen_frames(copied_ctx)
            
            # If user is on the same screen, add informational message (one-time use)
            if not dif_screen:
                screen_number = self.get_screen_number_for_frame(self.latest_frame)
                if screen_number:
                    copied_ctx.add_message(
                        role="user",
                        content=f"User is on the same screen: screen {screen_number}"
                    )
                    logger.info(f"Added 'same screen' message for screen {screen_number}")

        # Enhanced Langfuse tracing
        messages_for_langfuse = self.format_messages_for_langfuse(copied_ctx)

        ## Debug: Print messages being sent to LLM
        logger.info("===== MESSAGES SENT TO LLM (GEMINI) =====")
        for msg in copied_ctx.items:
            try:
                # Check if the item has a role attribute (ChatMessage)
                if hasattr(msg, 'role'):
                    # For text content, print directly
                    if isinstance(msg.content, str):
                        logger.info(f"[{msg.role}]: {msg.content[:100]}..." if len(msg.content) > 100 else f"[{msg.role}]: {msg.content}")
                    # For list content (multimodal), print structure
                    elif isinstance(msg.content, list):
                        logger.info(f"[{msg.role}]: Multimodal message with {len(msg.content)} elements")
                        for i, element in enumerate(msg.content):
                            if isinstance(element, str):
                                logger.info(f"  - Text element {i}: {element[:50]}..." if len(element) > 50 else f"  - Text element {i}: {element}")
                            elif hasattr(element, 'function_call') and element.function_call:
                                func_call = element.function_call
                                func_info = f"Function call: {func_call.name}"
                                if hasattr(func_call, 'args') and func_call.args:
                                    func_info += f" | Args: {str(func_call.args)[:50]}..."
                                logger.info(f"  - Function call element {i}: {func_info}")
                            elif hasattr(element, 'function_response') and element.function_response:
                                func_resp = element.function_response
                                func_info = f"Function response: {func_resp.name}"
                                if hasattr(func_resp, 'response') and func_resp.response:
                                    response_text = str(func_resp.response.text) if hasattr(func_resp.response, 'text') else str(func_resp.response)
                                    func_info += f" | Response: {response_text[:100]}..."
                                logger.info(f"  - Function response element {i}: {func_info}")
                            else:
                                logger.info(f"  - Other element {i}: {type(element).__name__}")
                    # Handle empty or None content
                    else:
                        logger.info(f"[{msg.role}]: [No content or unsupported content type]")
                
                # Handle standalone FunctionCall objects
                elif hasattr(msg, 'name'):  # FunctionCall has 'name' attribute
                    function_info = f"Function call: {msg.name}"
                    if hasattr(msg, 'args') and msg.args:
                        function_info += f" | Args: {str(msg.args)[:50]}..."
                    logger.info(f"[function]: {function_info}")
                
                # Handle other unknown types gracefully
                else:
                    logger.info(f"[unknown]: {type(msg).__name__} object")
                    
            except AttributeError as e:
                logger.warning(f"Error accessing message attributes: {e} - Type: {type(msg).__name__}")
            except Exception as e:
                logger.error(f"Unexpected error processing debug message: {e}")

        logger.info("=========================================")

        if self._ending_conversation:
            logger.info("Ending conversation")
            return
        
        generation = self.get_current_trace().generation(
            name="llm_generation",
            model="gemini-2.5-flash-preview-05-20",
            input=messages_for_langfuse,
            metadata={
                "temperature": 0.8,
                "has_latest_frame": self.latest_frame is not None,
                "is_different_screen": self.is_on_different_screen,
                "screen_positions_count": len(self.screen_positions)
            }
        )
        
        output = ""
        set_completion_start_time = False
        chunks = []
        
        try:
            async for chunk in Agent.default.llm_node(self, copied_ctx, tools, model_settings):
                if not set_completion_start_time:
                    generation.update(completion_start_time=datetime.now(UTC))
                    set_completion_start_time = True
                if chunk.delta and chunk.delta.content:
                    output += chunk.delta.content
                chunks.append(chunk)
                yield chunk
        except Exception as e:
            generation.update(level="ERROR")
            logger.error(f"LLM error: {e}")
            raise
        finally:
            final_output = {"role": "assistant", "content": output}
            generation.end(output=final_output)

    async def record_new_screen_position(self, chat_ctx: llm.ChatContext) -> None:
        """Record the position where a new screen should be inserted."""
        try:
            # Find the position after the last user message by searching from the end
            insertion_position = len(chat_ctx.items)  # Default: insert at the end
            
            # Search from the end to find the last user message faster
            for i in range(len(chat_ctx.items) - 1, -1, -1):
                item = chat_ctx.items[i]
                if hasattr(item, 'role') and item.role == 'user':
                    # Position after the last user message + account for existing screens
                    insertion_position = i + 1 + len(self.screen_positions)
                    logger.info(f"Insertion position: {insertion_position}")
                    break
            
            frame_id = id(self.latest_frame)
            screen_number = len(self.screen_positions) + 1
            
            # Record the position for this frame
            self.screen_positions[frame_id] = {
                'frame': self.latest_frame,  # Store the actual frame
                'insertion_position': insertion_position,
                'screen_number': screen_number,
                'timestamp': time.time()
            }
            
            logger.info(f"Recorded new screen position: Screen {screen_number} at position {insertion_position} (frame_id: {frame_id})")
            
        except Exception as e:
            logger.error(f"Error recording new screen position: {e}")

    async def insert_all_screen_frames(self, chat_ctx: llm.ChatContext) -> None:
        """Insert all recorded screen frames at their correct positions in every iteration."""
        try:
            if not self.screen_positions:
                return
            
            # Sort screen positions by their insertion order to maintain correct positioning
            sorted_screens = sorted(
                self.screen_positions.items(),
                key=lambda x: x[1]['screen_number']  # Sort by screen number for consistent ordering
            )
            
            # Create a new chat context to rebuild with screens inserted
            new_chat_ctx = []
            
            # Track positions in original and new context
            first_ctx_position = 0
            new_ctx_position = 0
            
            # Process each screen to insert
            for i in range(len(sorted_screens)):
                frame_id, screen_info = sorted_screens[i]
                
                # Calculate where this screen should be inserted in the new context
                new_ctx_position = screen_info['insertion_position'] - i
                
                # Add original items up to the insertion point
                for j in range(first_ctx_position, new_ctx_position):
                    if j < len(chat_ctx.items):
                        new_chat_ctx.append(chat_ctx.items[j])
                
                # Create and insert the screen message
                frame = screen_info['frame']
                screen_number = screen_info['screen_number']
                
                # Create image content
                image_content = ImageContent(
                    image=frame,
                    inference_detail="high"
                )
                
                # Create screen message
                screen_message = ChatMessage(
                    role="user",
                    content=[f"Screen {screen_number}:", image_content]
                )
                
                # Add the screen to the new context
                new_chat_ctx.append(screen_message)
                logger.debug(f"Inserted Screen {screen_number} at position {len(new_chat_ctx)-1}")
                
                # Update the first context position to continue after this insertion point
                first_ctx_position = new_ctx_position
            
            # Add any remaining items from the original context
            if first_ctx_position < len(chat_ctx.items):
                for j in range(first_ctx_position, len(chat_ctx.items)):
                    new_chat_ctx.append(chat_ctx.items[j])
            
            # Replace the original context items with our new rebuilt context
            chat_ctx.items.clear()
            for item in new_chat_ctx:
                chat_ctx.items.append(item)
            
            logger.info(f"Inserted {len(sorted_screens)} screen frames into context")
            
        except Exception as e:
            logger.error(f"Error inserting screen frames: {e}")

    def get_screen_number_for_frame(self, frame: rtc.VideoFrame) -> Optional[int]:
        """Get the screen number for a given frame if it exists in position tracking."""
        frame_id = id(frame)
        if frame_id in self.screen_positions:
            return self.screen_positions[frame_id]['screen_number']
        return None

    async def add_new_screen_to_context(self, chat_ctx: llm.ChatContext) -> None:
        """Add a new screen frame to the chat context at the appropriate position."""
        try:
            if not self.latest_frame:
                logger.warning("No latest frame available to add to context")
                return
                
            # Check if this frame already exists in our position tracking
            frame_id = id(self.latest_frame)  # Use frame object id as identifier
            
            if frame_id in self.screen_positions:
                logger.info("Frame already exists in context, skipping addition")
                return
            
            # Calculate insertion position
            # Find the actual position of user messages in the current context
            user_message_positions = []
            for i, item in enumerate(chat_ctx.items):
                if hasattr(item, 'role') and item.role == 'user':
                    user_message_positions.append(i)
            
            # Insert after the last user message, accounting for existing screens
            if user_message_positions:
                base_position = user_message_positions[-1] + 1
            else:
                base_position = len(chat_ctx.items)
            
            # Account for existing screens that were already inserted
            insertion_position = base_position + len(self.screen_positions)
            screen_number = len(self.screen_positions) + 1
            
            # Create image content
            image_content = ImageContent(
                image=self.latest_frame,
                inference_detail="high"
            )
            
            # Create the message with screen content
            screen_content = [f"Current screen (Screen {screen_number}):", image_content]
            
            # Insert into context - but we need to be careful about the ChatContext structure
            # Since we can't directly insert into ChatContext.items, we'll add it normally
            # and track it for future reference
            chat_ctx.add_message(
                role="user",
                content=screen_content
            )
            
            # Record the position for this frame (approximate position)
            self.screen_positions[frame_id] = {
                'position': len(chat_ctx.items) - 1,  # Position where we just added it
                'screen_number': screen_number,
                'timestamp': time.time()
            }
            
            logger.info(f"Added Screen {screen_number} to context (frame_id: {frame_id})")
            
        except Exception as e:
            logger.error(f"Error adding new screen to context: {e}")

    def get_screen_number_for_frame(self, frame: rtc.VideoFrame) -> Optional[int]:
        """Get the screen number for a given frame if it exists in position tracking."""
        frame_id = id(frame)
        if frame_id in self.screen_positions:
            return self.screen_positions[frame_id]['screen_number']
        return None

    def cleanup_old_screen_positions(self, max_age_seconds: int = 300) -> None:
        """Clean up old screen position records to prevent memory leaks."""
        current_time = time.time()
        to_remove = []
        
        for frame_id, info in self.screen_positions.items():
            if current_time - info['timestamp'] > max_age_seconds:
                to_remove.append(frame_id)
        
        for frame_id in to_remove:
            del self.screen_positions[frame_id]
            
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old screen position records")

    async def tts_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterable[rtc.AudioFrame]:
        span = self.get_current_trace().span(name="tts_node", metadata={"model": "elevenlabs"})
        try:
            async for event in Agent.default.tts_node(self, text, model_settings):
                yield event
        except Exception as e:
            span.update(level="ERROR")
            logger.error(f"TTS error: {e}")
            raise
        finally:
            span.end()

    def on_track_subscribed(
        self,
        track: rtc.RemoteTrack,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if publication.source != rtc.TrackSource.SOURCE_SCREENSHARE:
            return
        logger.info("Screen share track subscribed")

        # start the new stream
        asyncio.create_task(self.read_video_stream(rtc.VideoStream(track)))

    async def read_video_stream(self, video_stream: rtc.VideoStream) -> None:
        await self.close_video_stream()
        self.video_stream = video_stream

        logger.info("Starting video frame capture")
        frame_count = 0
        total_frames_received = 0
        
        async for event in video_stream:
            total_frames_received += 1
            current_time = time.time()
            time_diff = current_time - self.last_frame_time
            
            if time_diff >= 1.0:  # Capture frames at 1 per second
                current_frame = event.frame
                
                # Process frame for consistency
                task = asyncio.create_task(self.process_frame_with_consistency(current_frame))
                self.last_frame_time = current_time
                
                try:
                    result = await task
                    if result:  # Frame was accepted as consistent
                        frame_count += 1
                        # Mark as different screen when we get a new consistent frame
                        self.is_on_different_screen = True
                        logger.info(f"New consistent screen detected - marked as different (accepted frame {frame_count})")
                except Exception as e:
                    logger.error(f"Error processing frame: {e}")
                    
        logger.info(f"Video frame capture ended - received {total_frames_received} total frames, accepted {frame_count} frames")
    
    async def process_frame_with_consistency(self, current_frame: rtc.VideoFrame) -> bool:
        """
        Process frame with consistency checking to avoid capturing transitional/blurry frames.
        Returns True if frame is accepted as consistent, False otherwise.
        """
        try:
            # Handle the very first frame
            if self.last_consistent_frame is None:
                logger.info("Processing first frame - accepting immediately")
                self.latest_frame = current_frame
                self.last_consistent_frame = current_frame
                self.consecutive_captures = []  # Clear any existing captures
                return True
            
            # Check if this frame is different from the last consistent frame
            is_different_enough, similarity = await self.is_frame_different_enough(
                current_frame, self.last_consistent_frame
            )
            
            logger.debug(f"Frame similarity check: different_enough={is_different_enough}, similarity={similarity:.3f}")
            
            if is_different_enough:
                # This is a different frame - add to consecutive captures
                self.consecutive_captures.append({
                    'frame': current_frame,
                    'timestamp': time.time()
                })
                logger.debug(f"Added frame to consecutive captures (total: {len(self.consecutive_captures)})")
                
                # If we have only 1 capture, we need to wait for more to confirm consistency
                # But if we have been waiting too long (>5 seconds), accept what we have
                if len(self.consecutive_captures) == 1:
                    first_capture_time = self.consecutive_captures[0]['timestamp']
                    if time.time() - first_capture_time < 2.0:  # Wait up to 2 seconds for confirmation
                        logger.debug("Waiting for consistency confirmation...")
                        return False
                    else:
                        logger.info("Timeout waiting for consistency - accepting single capture")
                
                # Accept the frame if we have multiple consecutive captures or timeout
                consistent_frame = self.consecutive_captures[-1]['frame']
                self.latest_frame = consistent_frame
                self.last_consistent_frame = consistent_frame
                
                capture_count = len(self.consecutive_captures)
                logger.info(f"Accepted consistent frame from {capture_count} consecutive captures (similarity from last: {similarity:.3f})")
                
                # Clear consecutive captures
                self.consecutive_captures = []
                return True
                
            else:
                # This frame is similar to the last consistent frame
                logger.debug(f"Frame is similar to last consistent frame (similarity: {similarity:.3f})")
                
                # If we have pending consecutive captures, this similarity indicates 
                # the screen has stabilized, so we should accept the last capture
                if len(self.consecutive_captures) > 0:
                    consistent_frame = self.consecutive_captures[-1]['frame']
                    self.latest_frame = consistent_frame
                    self.last_consistent_frame = consistent_frame
                    
                    capture_count = len(self.consecutive_captures)
                    logger.info(f"Screen stabilized - accepted consistent frame from {capture_count} consecutive captures")
                    
                    # Clear consecutive captures
                    self.consecutive_captures = []
                    return True
                
                # No pending captures and similar frame - just skip
                return False
            
        except Exception as e:
            logger.error(f"Error in frame consistency processing: {e}")
            # On error, clear consecutive captures and don't accept frame
            self.consecutive_captures = []
            return False

    async def process_new_frame(self, current_frame: rtc.VideoFrame, last_captured_frame: Optional[rtc.VideoFrame]) -> Tuple[bool, Optional[rtc.VideoFrame]]:
        """
        Process a new frame without blocking the main event loop
        Returns: (was_frame_captured, frame_to_use_for_next_comparison)
        """
        try:
            # Check if this frame is different from the last captured frame
            is_different_enough = True
            similarity = 0.0
            
            if last_captured_frame is not None:
                # Use the helper method to check similarity
                is_different_enough, similarity = await self.is_frame_different_enough(
                    current_frame, last_captured_frame
                )
            
            if is_different_enough:
                # Update the latest frame
                self.latest_frame = current_frame
                logger.info(f"Captured new frame: {self.latest_frame.width}x{self.latest_frame.height} (similarity: {similarity:.2f})")
                return (True, current_frame)  # Frame was captured, use it for future comparisons
            else:
                logger.debug(f"Skipped similar frame (similarity: {similarity:.2f})")
                return (False, last_captured_frame)  # Frame was not captured, keep using the last one
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return (False, last_captured_frame)

async def entrypoint(ctx: JobContext) -> None:
    # Connect to the room
    await ctx.connect()

    logger.info(f"Connected to room: {ctx.room.name}")
    logger.info(f"Local participant: {ctx.room.local_participant.identity}")
        
    # Wait for a remote participant to join
    await ctx.wait_for_participant()
    logger.info("👤 Participant joined the room")

    logger.info(f"Found {len(ctx.room.remote_participants)} remote participants")
    # Create a simple agent session without custom frame rate
    # Just use the default settings which should work fine
    session = AgentSession()

    # Configure agent
    agent = VideoAgent(instructions=INSTRUCTIONS, room=ctx.room)
    
    # Set up room input/output - explicitly enable all modes
    room_input = RoomInputOptions(
        video_enabled=True,
        audio_enabled=True
    )
    
    room_output = RoomOutputOptions(
        audio_enabled=True,
        transcription_enabled=True
    )

    # Start the agent with all capabilities
    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=room_input,
        room_output_options=room_output,
    )
    
    @ctx.room.on("participant_disconnected")
    def on_participant_disconnected(participant):
        """Handle when a participant disconnected"""
        logger.info(f"👋 Participant disconnected: {participant.identity}")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
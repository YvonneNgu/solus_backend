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
        
        # Check if latest frame should be added to history
        if self.latest_frame:
            # Automatically check and add to history if needed
            await self.check_and_add_to_history_if_needed()        
        else:
            # No frames available - user is not sharing their screen
            copied_ctx.add_message(
                role="user",
                content="The user is not currently sharing their screen."
            )
            logger.warning("No captured frames available for this conversation")

        # Add screen history to context
        if self.screen_history:
            for i, frame in enumerate(self.screen_history, 1):
                image_content = ImageContent(
                    image=frame,
                    inference_detail="high"
                )
                copied_ctx.add_message(
                    role="user",
                    content=[f"Screen history: Screen {i}:", image_content]
                )
                logger.info(f"Added Screen {i} to chat context")

        # Enhanced Langfuse tracing
        messages_for_langfuse = self.format_messages_for_langfuse(copied_ctx)

        # Debug: Print messages being sent to LLM
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
            input=messages_for_langfuse,  # Use formatted messages
            metadata={
                "temperature": 0.8,
                "has_screen_history": len(self.screen_history) > 0,
                "screen_count": len(self.screen_history),
                "has_latest_frame": self.latest_frame is not None
            }
        )
        
        output = ""
        set_completion_start_time = False
        chunks = []
        
        try:
            async for chunk in Agent.default.llm_node(self, copied_ctx, tools, model_settings):
                if not set_completion_start_time:
                    generation.update(
                        completion_start_time=datetime.now(UTC),
                    )
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
            # Create a simple output format for better readability
            final_output = {
                "role": "assistant",
                "content": output
            }
            generation.end(output=final_output)

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
        # close open streams
        await self.close_video_stream()
        self.video_stream = video_stream

        logger.info("Starting video frame capture")
        frame_count = 0
        last_captured_frame = None  # Store the last captured frame for comparison
        
        async for event in video_stream:
            # Capture frames at 1 per second
            current_time = time.time()
            if current_time - self.last_frame_time >= 1.0:
                # Get current frame
                current_frame = event.frame
                
                # Process frame in background to avoid blocking
                task = asyncio.create_task(self.process_new_frame(current_frame, last_captured_frame))
                
                # Update capture time immediately to maintain capture rate
                self.last_frame_time = current_time
                
                # Wait for the task to complete to get the result
                try:
                    result = await task
                    if result[0]:  # Frame was captured
                        frame_count += 1
                        last_captured_frame = result[1]
                except Exception as e:
                    logger.error(f"Error waiting for frame processing: {e}")
                    
        logger.info(f"Video frame capture ended - captured {frame_count} unique frames")
    
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
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
from dataclasses import dataclass

from dotenv import load_dotenv
from langfuse import get_client
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
from livekit.agents import get_job_context
from livekit import api
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from PIL import Image

# Import the tool functions
from tools.lookup_weather import lookup_weather as lookup_weather_tool
from tools.identify_screen_elements import identify_screen_elements as identify_screen_elements_tool
from tools.display_instructions import display_instructions as display_instructions_tool

from frame_observer import FrameObserver

logger = logging.getLogger("multi-agent-system")
logger.setLevel(logging.INFO)

load_dotenv()
langfuse = get_client()

# Session data to track navigation goals
@dataclass
class NavigationSessionData:
    current_goal: str | None = None
    goal_achieved: bool = False

# Define shared tools outside the classes
@function_tool()
async def lookup_weather(
    context: RunContext,
    location: str,
) -> dict[str, Any]:
    """Look up weather information for a given location.
    
    Args:
        location: The location to look up weather information for.
    """
    
    with langfuse.start_as_current_span(
        name="lookup_weather_tool",
        input={"location": location}
    ) as span:
        try:
            result = await lookup_weather_tool(context, location)
            span.update(output=result)
            return result
        except Exception as e:
            span.update(level="ERROR", status_message=str(e))
            raise

# Base agent class with shared functionality
class BaseVideoAgent(Agent):
    def __init__(self, instructions: str, room: rtc.Room, chat_ctx: ChatContext = None, tools: List[FunctionTool] = None, **kwargs) -> None:
        
        super().__init__(
            instructions=instructions,
            llm=google.LLM(
                model="gemini-2.5-flash",
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                temperature=0.8,
            ),
            stt=google.STT(),
            tts=google.TTS(),
            vad=silero.VAD.load(),
            turn_detection=MultilingualModel(),
            tools=tools,
            chat_ctx=chat_ctx,  # Pass chat context to preserve conversation history
            **kwargs
        )
        
        self.room = room
        self.session_id = str(uuid4())
        self.current_trace = None

        self.latest_frame: Optional[rtc.VideoFrame] = None
        self.last_frame_time: float = 0
        self.video_stream: Optional[rtc.VideoStream] = None
        self._ending_conversation = False
        self._tasks = []  # Track async tasks
        self.is_on_different_screen = True  # Flag to track if user moved to different screen
        self.screen_number = 0  # Track current screen number
        
        self.frame_observer = FrameObserver(self) # observe frame change
        
        self.summary_message = None # summarize function messages in chat ctx
        self.summary_message_time:float = None

    async def close(self) -> None:
        try:
            # ensure stop observing frame change
            await self.frame_observer.stop_observation()
            
            # Cancel all tasks
            for task in self._tasks:
                if not task.done():
                    task.cancel()
            if self._tasks:
                await asyncio.gather(*self._tasks, return_exceptions=True)
            
            await self.close_video_stream()
        except Exception as e:
            logger.error(f"Error closing agent: {e}")
        
        try:
            if self.current_trace:
                self.current_trace.end()
                self.current_trace = None
            langfuse.flush()
        except Exception as e:
            logger.error(f"Error flushing langfuse: {e}")

    async def close_video_stream(self) -> None:
        if self.video_stream:
            await self.video_stream.aclose()
            self.video_stream = None
            self.latest_frame = None

    async def on_exit(self) -> None:
        # Only close video stream if we still own it (not transferred to another agent)
        if self.video_stream:
            await self.close_video_stream()
        
        try:
            if self.current_trace:
                self.current_trace.end()
                self.current_trace = None
            langfuse.flush()
        except Exception as e:
            logger.error(f"Error flushing langfuse: {e}")

    def on_user_state_change(self, event: UserStateChangedEvent) -> None:
        old_state = event.old_state
        new_state = event.new_state

    def on_agent_state_change(self, event: AgentStateChangedEvent) -> None:
        old_state = event.old_state
        new_state = event.new_state
        logger.info(f"Agent state changed: {old_state} -> {new_state}")

    async def on_user_turn_completed(
        self, turn_ctx: ChatContext, new_message: ChatMessage,
    ) -> None:
        """User turn completed callback."""
        logger.info(f"User turn completed {langfuse.get_current_trace_id()}")
        
        # Set cancellation flag if we're a NavigationAgent and display is active
        if isinstance(self, NavigationAgent) and self.display_active:
            logger.info("User spoke while display active - setting cancellation flag")
            self.cancel_current_display = True


            
    def print_chat_context(self, chat_ctx: llm.ChatContext) -> None:
        """Debug method to print chat context being sent to LLM"""
        
        logger.info("===== CHAT CONTEXT SENT TO LLM (GEMINI) =====")
        for i, item in enumerate(chat_ctx.items):
            try:
                # Check item type first, then role
                item_type = getattr(item, 'type', None)
                role = getattr(item, 'role', 'unknown')
                
                def extract_content(content):
                    """Helper function to extract and format content properly"""
                    if isinstance(content, list):
                        if len(content) == 1 and isinstance(content[0], str):
                            return content[0]
                        elif len(content) > 1:
                            # Handle multimodal content
                            text_parts = []
                            has_image = False
                            for element in content:
                                if isinstance(element, str):
                                    text_parts.append(element)
                                else:
                                    has_image = True
                            
                            combined_text = " ".join(text_parts) if text_parts else ""
                            if has_image:
                                combined_text += " [ImageContent]" if combined_text else "[ImageContent]"
                            return combined_text
                        else:
                            return str(content)
                    elif isinstance(content, str):
                        return content
                    else:
                        return str(content)
                
                # Handle based on type first
                if item_type == 'function_call':
                    func_name = getattr(item, 'name', 'unknown_function')
                    arguments = getattr(item, 'arguments', '{}')
                    args_str = arguments[:50] + "..." if len(arguments) > 50 else arguments
                    logger.info(f"[{i}] [function] : {func_name}, args: {args_str}")
                    
                elif item_type == 'function_call_output':
                    func_name = getattr(item, 'name', 'unknown_function')
                    output = getattr(item, 'output', 'No output')
                    is_error = getattr(item, 'is_error', False)
                    
                    truncated_output = output[:100] + "..." if len(output) > 100 else output
                    error_prefix = "ERROR: " if is_error else ""
                    logger.info(f"[{i}] [function] : Response from {func_name}: {error_prefix}{truncated_output}")
                    
                elif role == 'system':
                    content = getattr(item, 'content', '')
                    formatted_content = extract_content(content)
                    truncated_content = formatted_content[:50] + "..." if len(formatted_content) > 50 else formatted_content
                    logger.info(f"[{i}] [system]   : {truncated_content}")
                    
                elif role == 'assistant':
                    # Check if this is a tool call
                    if hasattr(item, 'tool_calls') and item.tool_calls:
                        tool_call = item.tool_calls[0]  # Assuming single tool call
                        func_name = tool_call.function.name
                        args = tool_call.function.arguments
                        args_str = str(args)[:50] + "..." if len(str(args)) > 50 else str(args)
                        logger.info(f"[{i}] [function] : {func_name}, args: {args_str}")
                    else:
                        content = getattr(item, 'content', '')
                        formatted_content = extract_content(content)
                        logger.info(f"[{i}] [assistant]: {formatted_content}")
                        
                elif role == 'user':
                    content = getattr(item, 'content', '')
                    formatted_content = extract_content(content)
                    logger.info(f"[{i}] [user]     : {formatted_content}")
                        
                elif role == 'tool':
                    func_name = getattr(item, 'name', 'unknown_function')
                    content = getattr(item, 'content', '')
                    
                    # Try to parse and truncate the response
                    try:
                        formatted_content = extract_content(content)
                        if len(formatted_content) > 100:
                            truncated_content = formatted_content[:100] + "..."
                        else:
                            truncated_content = formatted_content
                        logger.info(f"[{i}] [function] : Response from {func_name}: {truncated_content}")
                    except:
                        logger.info(f"[{i}] [function] : Response from {func_name}: {str(content)[:100]}...")
                        
                else:
                    # Handle unknown roles - try to detect if it's a function call/response
                    content = getattr(item, 'content', None)
                    name = getattr(item, 'name', None)
                    tool_calls = getattr(item, 'tool_calls', None)
                    
                    if tool_calls:
                        # This is likely a function call
                        tool_call = tool_calls[0]
                        func_name = tool_call.function.name
                        args = tool_call.function.arguments
                        args_str = str(args)[:50] + "..." if len(str(args)) > 50 else str(args)
                        logger.info(f"[{i}] [function] : {func_name}, args: {args_str}")
                    elif name and hasattr(item, 'output'):
                        # This is likely a function response with output attribute
                        output = getattr(item, 'output', 'No output')
                        is_error = getattr(item, 'is_error', False)
                        truncated_output = output[:100] + "..." if len(output) > 100 else output
                        error_prefix = "ERROR: " if is_error else ""
                        logger.info(f"[{i}] [function] : Response from {name}: {error_prefix}{truncated_output}")
                    elif name:
                        # This is likely a function response
                        formatted_content = extract_content(content) if content else "No response"
                        truncated_content = formatted_content[:100] + "..." if len(formatted_content) > 100 else formatted_content
                        logger.info(f"[{i}] [function] : Response from {name}: {truncated_content}")
                    elif content:
                        formatted_content = extract_content(content)
                        logger.info(f"[{i}] [{role}]: {formatted_content}")
                    else:
                        # Try to find any meaningful attributes
                        attrs = []
                        for attr in ['type', 'id', 'call_id', 'arguments', 'output']:
                            if hasattr(item, attr):
                                value = getattr(item, attr)
                                if value:
                                    attrs.append(f"{attr}={str(value)[:30]}")
                        
                        if attrs:
                            logger.info(f"[{i}] [{role}]: {', '.join(attrs)}")
                        else:
                            logger.info(f"[{i}] [{role}]: {type(item).__name__} object")
                    
            except AttributeError as e:
                logger.warning(f"[{i}] Error accessing message attributes: {e} - Type: {type(item).__name__}")
            except Exception as e:
                logger.error(f"[{i}] Unexpected error processing debug message: {e}")

        logger.info("=========================================")
        


    def on_track_subscribed(
        self,
        track: rtc.RemoteTrack,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if publication.source != rtc.TrackSource.SOURCE_SCREENSHARE:
            return
        logger.info("Screen share track subscribed")
        self._create_video_stream(track)

    def _create_video_stream(self, track: rtc.Track):
        """Create video stream to buffer the latest video frame."""
        # Close any existing stream
        if self.video_stream is not None:
            asyncio.create_task(self.close_video_stream())
        
        # Create a new stream to receive frames
        self.video_stream = rtc.VideoStream(track)
        
        async def read_stream():
            logger.info("Starting video frame capture")
            frame_count = 0
            
            try:
                async for event in self.video_stream:
                    current_time = time.time()
                    time_diff = current_time - self.last_frame_time
                    
                    if time_diff >= 1.0:  # Capture frames at 1 per second
                        current_frame = event.frame
                        self.last_frame_time = time.time()
                        
                        if self.latest_frame is None:
                            self.latest_frame = current_frame
                            self.is_on_different_screen = True  # Mark as different screen
                            logger.info("Set first frame as latest frame")
                            frame_count += 1
                        else:
                            try:
                                is_different, similarity = await self.is_frame_different_enough(current_frame, self.latest_frame)
                                if is_different:
                                    self.latest_frame = current_frame
                                    self.is_on_different_screen = True
                                    logger.info("New frame detected")
                                    frame_count += 1
                                else:
                                    logger.info(f"Skipped - similar frame {similarity:.3f}")
                                
                            except Exception as e:
                                logger.error(f"Error comparing frames: {e}")
                                return (True, "error")
                        
            except Exception as e:
                logger.error(f"Error in video stream: {e}")
            finally:
                logger.info(f"Video frame capture ended - captured {frame_count} frames")
        
        # Store the async task
        task = asyncio.create_task(read_stream())
        task.add_done_callback(lambda t: self._tasks.remove(t) if t in self._tasks else None)
        self._tasks.append(task)
        
    def calculate_frame_similarity(self, frame_data1: bytes, frame_data2: bytes) -> float:
        """Calculate similarity between two frames."""
        try:
            img1 = Image.open(io.BytesIO(frame_data1))
            img2 = Image.open(io.BytesIO(frame_data2))
            
            size = (50, 50)
            img1 = img1.resize(size)
            img2 = img2.resize(size)
            
            img1 = img1.convert('L')
            img2 = img2.convert('L')
            
            pixels1 = list(img1.getdata())
            pixels2 = list(img2.getdata())
            
            if len(pixels1) != len(pixels2):
                return 0.0
                
            sample_rate = 4
            sample_pixels1 = pixels1[::sample_rate]
            sample_pixels2 = pixels2[::sample_rate]
            
            matches = 0
            total_samples = len(sample_pixels1)
            tolerance = 15
            
            for i in range(total_samples):
                if abs(sample_pixels1[i] - sample_pixels2[i]) <= tolerance:
                    matches += 1
                    
            similarity = matches / total_samples
            return similarity
        except Exception as e:
            logger.error(f"Error calculating frame similarity: {e}")
            return 0.0

    async def is_frame_different_enough(self, frame1: rtc.VideoFrame, frame2: rtc.VideoFrame) -> Tuple[bool, float]:
        """Check if two frames are different enough (less than 98% similar)."""
        try:
            frame1_data = encode(frame1, EncodeOptions(format="JPEG", quality=80))
            frame2_data = encode(frame2, EncodeOptions(format="JPEG", quality=80))
            
            loop = asyncio.get_event_loop()
            similarity = await loop.run_in_executor(
                None, 
                self.calculate_frame_similarity,
                frame1_data, 
                frame2_data
            )
            
            is_different_enough = similarity < 0.98
            return (is_different_enough, similarity)
        except Exception as e:
            logger.error(f"Error comparing frames: {e}")
            return (True, 0.0)

# Conversation Agent - handles general questions and decides when to hand off
class ConversationAgent(BaseVideoAgent):
    def __init__(self, room: rtc.Room, chat_ctx: ChatContext = None) -> None:
        instructions = """
You are Solus, a friendly mobile voice assistant designed for low-literate users in Malaysia who are not familiar with technology.

Your primary role is to:
- Answer general questions and make small talk
- Analyze user requests to determine if they need navigation guidance
- Hand off to the navigation agent when users need step-by-step mobile guidance

When to hand off to navigation:
- User asks for help with filling forms, changing settings, finding features
- User says things like "how to", "where is", "help me change", "I want to", "how can I"
- User needs guidance navigating their mobile interface

When NOT to hand off:
- General questions about their screen content (e.g., "translate this text", "what does this say")
- Small talk or casual conversation
- Simple informational queries

Response guidelines:
- Keep responses short and concise, maximum 100 words
- Use friendly tone and everyday language
- No technical terms or markdown formatting
- Reply in English unless user requests other languages
- Don't mention screen history or technical details
"""

        # ConversationAgent tools: lookup_weather, hand_off_to_navigation, end_conversation
        super().__init__(
            instructions=instructions,
            room=room,
            chat_ctx=chat_ctx,
            tools=[lookup_weather]  # Will add handoff and end_conversation tools below
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions="Greet the user warmly. Don't exceed 15 words"
        )
        self.session.on("user_state_changed", self.on_user_state_change)
        self.session.on("agent_state_changed", self.on_agent_state_change)
        self.room.on("track_subscribed", self.on_track_subscribed)

    @function_tool()
    async def hand_off_to_navigation(
        self, 
        context: RunContext[NavigationSessionData], 
        goal: str
    ) -> Agent:
        """Hand off to the navigation agent when user needs step-by-step mobile guidance.
        
        Args:
            goal: Clear description of what the user wants to achieve (e.g., "fill out the application form", "change app language to English", "share location in WhatsApp")
        """
        with langfuse.start_as_current_span(
            name="hand_off_to_navigation",
            input={"goal": goal}
        ) as span:
            try:
                logger.info(f"Handing off to navigation agent with goal: {goal}")
                
                # Store the goal in session userdata
                context.userdata.current_goal = goal
                context.userdata.goal_achieved = False
                
                # Create navigation agent with preserved chat context
                navigation_agent = NavigationAgent(
                    room=self.room,
                    goal=goal,
                    chat_ctx=self.chat_ctx  # Pass current chat context
                )
                logger.info("hand off - self.chat_ctx")
                self.print_chat_context(self.chat_ctx)
                logger.info("hand off - self.session.chat_ctx")
                self.print_chat_context(self.session._chat_ctx)
                
                # Transfer basic frame state (but not the video stream itself)
                navigation_agent.latest_frame = self.latest_frame
                navigation_agent.last_frame_time = self.last_frame_time
                navigation_agent.is_on_different_screen = self.is_on_different_screen
                navigation_agent.screen_number = self.screen_number
                
                # Get the current video track from the existing stream
                current_track = None
                if self.video_stream:
                    current_track = self.video_stream._track
                    logger.info("Found existing video track for transfer")
                
                # Close the current video stream properly
                await self.close_video_stream()
                
                # Create new video stream for navigation agent if we have a track
                if current_track:
                    navigation_agent._create_video_stream(current_track)
                    logger.info("Created new video stream for navigation agent")
                
                logger.info("Successfully handed off to navigation agent")
                
                span.update(output={"success": True, "goal": goal})
                
                # Return new agent
                return navigation_agent
                
            except Exception as e:
                span.update(level="ERROR", status_message=str(e))
                logger.error(f"Error handing off to navigation: {e}")
                raise

    @function_tool()
    async def end_conversation(
        self, 
        context: RunContext, 
        farewell_message: str = None
    ) -> None:
        """Use this tool to naturally end the conversation when the user indicates they're done.
        
        Args:
            farewell_message: Custom closing message to say to the user before ending the session
        """
        with langfuse.start_as_current_span(
            name="end_conversation"
        ) as span:
            try:
                self._ending_conversation = True
                
                if not farewell_message:
                    farewell_message = "Thank you for using Solus. Have a wonderful day!"
                
                await context.session.say(farewell_message)
                logger.info("Said goodbye")
                
                #add code here
                # Code to close UI for user (frontend app) using RPC
                job_ctx = get_job_context()
                room = job_ctx.room
                remote_participants = list(room.remote_participants.values())

                if remote_participants:
                    target_participant = remote_participants[0]
                    logger.info(f"Sending close UI command to participant: {target_participant.identity}")
                    
                    try:
                        # Perform RPC call to close frontend UI
                        response = await room.local_participant.perform_rpc(
                            destination_identity=target_participant.identity,
                            method="close-voice-assistant",
                            payload="",
                            response_timeout=5.0
                        )
                        logger.info(f"Close UI RPC successful. Response: {response}")
                        
                    except rtc.RpcError as rpc_error:
                        logger.error(f"Close UI RPC call failed: {rpc_error.code} - {rpc_error.message}")
                    except Exception as e:
                        logger.error(f"Error sending close UI command: {str(e)}")
                else:
                    logger.warning("No remote participants found to send close UI command")
                
                logger.info("Deleting room")
                await job_ctx.api.room.delete_room(api.DeleteRoomRequest(room=job_ctx.room.name))
                logger.info("Room deleted")
                
            except Exception as e:
                logger.error(f"Error ending conversation: {e}")
                
    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: List[FunctionTool],
        model_settings: ModelSettings
    ) -> AsyncIterable[llm.ChatChunk]:
        
        dif_screen = self.is_on_different_screen
        logger.info("Is on different screen: %s", dif_screen)
        logger.info("Latest frame exists: %s", bool(self.latest_frame))
        
        # merge assistant message in session chat_ctx into chat_ctx
        # chat_ctx only has function message
        chat_ctx.merge(other_chat_ctx=self.session._chat_ctx, exclude_function_call=True,exclude_instructions=True)
                
        # Handle screen state and messages
        if self.latest_frame is None:
            # User is not sharing screen: create copy and add temporary message
            temp_chat_ctx = chat_ctx.copy()
            temp_chat_ctx.add_message(
                role="user",
                content="User is not currently sharing their screen."
            )
            logger.info("Added 'not sharing screen' temporary message to context")
        else:
            # User is sharing screen: check if user is on dif screen
            if dif_screen:
                # User moved to a different screen: 
                # - add latest screen frame to PERMANENT context
                # - reset flag
                self.screen_number += 1
                
                image_content = ImageContent(
                    image=self.latest_frame
                )
                
                # Add to PERMANENT context (this should persist)
                chat_ctx.add_message(
                    role="user",
                    content=[f"Screen {self.screen_number}:", image_content]
                )
                logger.info(f"Added screen {self.screen_number} frame to permanent context")
                
                # Create copy after adding to permanent context (so copy includes the new screen)
                temp_chat_ctx = chat_ctx.copy()
                
                self.is_on_different_screen = False
            else:
                # user is on the same screen: create copy and add TEMPORARY message
                temp_chat_ctx = chat_ctx.copy()
                temp_chat_ctx.add_message(
                    role="user",
                    content=f"User is on the screen {self.screen_number}"
                )
                logger.info(f"Added 'same screen' temporary message for screen {self.screen_number}")

        # Print chat context for debugging (use temp context)
        self.print_chat_context(temp_chat_ctx)

        if self._ending_conversation:
            logger.info("Ending conversation")
            return
        
        # Use context manager for generation
        with langfuse.start_as_current_generation(
            name="llm_generation",
            model="gemini-2.5-flash",
            input=temp_chat_ctx.to_provider_format('google'),
            metadata={
                "temperature": 0.8,
                "is_sharing_screen": self.latest_frame is not None,
                "screen_number": self.screen_number,
                "is_different_screen": dif_screen,
                "agent": self.room.local_participant.name,
                "timestamp": datetime.now(UTC).isoformat()
            }
        ) as generation:
        
            output = ""
            set_completion_start_time = False
            chunks = []
            start_time = time.time()
            
            try:
                # Use temporary context for LLM call, but pass original context for Agent methods
                async for chunk in Agent.default.llm_node(self, temp_chat_ctx, tools, model_settings):
                    if not set_completion_start_time:
                        generation.update(completion_start_time=datetime.now(UTC))
                        set_completion_start_time = True
                    if chunk.delta and chunk.delta.content:
                        output += chunk.delta.content
                    chunks.append(chunk)
                    yield chunk
                
            except Exception as e:
                generation.update(level="ERROR", status_message=str(e))
                logger.error(f"LLM error: {e}")
                raise
                
            finally:
                # Calculate response time for performance monitoring            
                logger.info("response_time = %d", time.time() - start_time)
                
                await self.update_chat_ctx(chat_ctx)
                logger.info("Update chat ctx")
                
                final_output = {"role": "assistant", "content": output}
                logger.info(f"Assistant: {output}")
                generation.update(output=final_output)

    async def stt_node(
        self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
    ) -> Optional[AsyncIterable[stt.SpeechEvent]]:
        """STT node for ConversationAgent with custom processing."""
        
        with langfuse.start_as_current_span(
            name="conversation_stt_node", 
            metadata={"model": "google", "agent": "conversation"}
        ) as span:
            try:
                async for event in Agent.default.stt_node(self, audio, model_settings):
                    if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                        logger.info(f"ConversationAgent - Speech recognized: {event.alternatives[0].text[:50]}...")
                    yield event
            except Exception as e:
                span.update(level="ERROR", status_message=str(e))
                logger.error(f"ConversationAgent STT error: {e}")
                raise

    async def tts_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterable[rtc.AudioFrame]:
        """TTS node for ConversationAgent with custom processing."""
        
        with langfuse.start_as_current_span(
            name="conversation_tts_node", 
            metadata={"model": "google cloud tts", "agent": "conversation"}  
        ) as span:
            try:
                async for event in Agent.default.tts_node(self, text, model_settings):
                    yield event
            except Exception as e:
                span.update(level="ERROR", status_message=str(e))
                logger.error(f"ConversationAgent TTS error: {e}")
                raise
    
    
# Navigation Agent - handles step-by-step UI guidance
class NavigationAgent(BaseVideoAgent):
    def __init__(self, room: rtc.Room, goal: str, chat_ctx: ChatContext = None) -> None:
        instructions = f"""
You are Solus, a friendly mobile voice assistant designed for low-literate users in Malaysia who are not familiar with technology. Your role is to help the user achieve this specific goal through step-by-step guidance. You will be given `interactive_ui_components` which is an array of objects each with `label` and `box_2d: [top, left, bottom, right]`.  

CURRENT GOAL: {goal}

**Loop Behavior**  
2. Determine the *single next* action the user should take to advance toward their goal (e.g. tap “Settings”).  
3. Speak a **very simple**, patient instruction (“Please tap the Settings button.”).  
4. Find the position of the target UI element in the interactive ui components list given, then display the visual cue on the user screen by calling `display_instructions()`.  
5. Repeat **only when** the screen updates, until the goal is achieved or the user wants to stop.  

Response guidelines:
- Keep responses short and clear
- Use everyday language, no technical terms
- Say "Sorry for the confusion, let me correct it" when providing corrections
- Response in plain text only, no markdown formatting. Your response will be read aloud to the user

Remember: 
- You can only succeed when the user achieves the goal: {goal}
- If user want to stop the guiding process, response nothing and hand back to conversation agent
"""

        # NavigationAgent tools: display_instructions, hand_back_to_conversation
        super().__init__(
            instructions=instructions,
            room=room,
            chat_ctx=chat_ctx   # Preserve conversation context
        )
        
        self.goal = goal
        
        self.identified = False
        self.display_active = False
        
        # Cache for identified frames to avoid redundant identifications
        self.identified_frames = []  # List of {frame: VideoFrame, components: dict}
        #self.last_message_time:float = None
        
        # Flag to cancel current display instructions when user speaks
        self.cancel_current_display = False

    async def _find_cached_frame_components(self, current_frame: rtc.VideoFrame) -> Optional[str]:
        """
        Helper function to check if current frame matches any cached frame.
        Returns the cached components if found, None otherwise.
        
        Args:
            current_frame: The current video frame to compare
            
        Returns:
            String containing the cached interactive_ui_components if found, None otherwise
        """
        if not current_frame or not self.identified_frames:
            return None
            
        try:
            # Search through cached frames starting from the most recent (reverse order)
            for cached_entry in reversed(self.identified_frames):
                cached_frame = cached_entry["frame"]

                is_different, similarity = await self.is_frame_different_enough(current_frame, cached_frame)
                if not is_different: # look for frame that has similarity > 98%
                    logger.info("Found matching cached frame components with similarity: %s", similarity)
                    return cached_entry["components"]
                    
        except Exception as e:
            logger.error(f"Error comparing frames: {e}")
            
        return None
        
    def _cache_frame_components(self, frame: rtc.VideoFrame, components: str) -> None:
        """
        Cache the frame and its identified components.
        
        Args:
            frame: The video frame to cache
            components: The identified interactive UI components as text
        """
        try:
            # Add to cache (keep only last 5 frames to manage memory)
            self.identified_frames.append({
                "frame": frame,
                "components": components
            })
            
            # Keep cache size reasonable
            if len(self.identified_frames) > 5:
                self.identified_frames.pop(0)
                
            logger.info(f"Cached frame components. Cache size: {len(self.identified_frames)}")
            
        except Exception as e:
            logger.error(f"Error caching frame components: {e}")

    async def on_enter(self) -> None:
        # Start background identification if we have a frame (non-blocking)
        if self.latest_frame:
            logger.info("Starting background screen element identification on enter")
            await self.identify_screen_elements(self.session)

        await self.session.generate_reply(
            instructions=f"Briefly acknowledge that you'll help with: {self.goal}. Ask the user to share their screen if they are not sharing."
        )
        self.session.on("user_state_changed", self.on_user_state_change)
        self.session.on("agent_state_changed", self.on_agent_state_change)
        self.room.on("track_subscribed", self.on_track_subscribed)
        
    async def identify_screen_elements(self, session) -> bool:
        """Identify screen elements and cache them if successful.
        
        Args:
            session: The session object for identification
            
        Returns:
            bool: True if identification was successful, False otherwise
        """
        try:
            result = await identify_screen_elements_tool(
                self.latest_frame if self.latest_frame else None,
                session
            )
            
            # The result is now a string (text) from the modified identify_screen_elements tool function
            if result and not result.startswith("Screen element identification error"):
                self.identified = True
                logger.info("Success, set value of Identified to: %s", self.identified)
                
                # Cache the frame and its components for future use
                # We store the result string directly
                self._cache_frame_components(self.latest_frame, result)
                return True
            
            return False
        
        except Exception as e:
            error_msg = f"Error identifying screen elements: {str(e)}"
            logger.error(error_msg)
            return False

    async def get_interactive_ui_components(
        self,
        session,
    ) -> str:
        """Get interactive UI components and their positions on the user's current screen.
        
        Returns:
            str: The interactive UI components as text.
        """
        # First check if we have cached components for this frame
        if self.latest_frame:
            cached_components = await self._find_cached_frame_components(self.latest_frame)
            if cached_components:
                # Found cached result, return it and mark as identified
                self.identified = True
                logger.info("Using cached frame components, set identified to: %s", self.identified)
                # Return the cached components directly as they're already a string
                return cached_components
        
        # If no cached components found, return the last available components if any exist
        if self.identified_frames:
            # Use the most recent cached components
            last_components = self.identified_frames[-1]["components"]
            logger.info("Using last cached components from previous screen")
            return last_components
        
        # If identification failed or no cached result found, return error message
        return "Screen element identification failed"

    @function_tool()
    async def display_instructions(
        self,
        context: RunContext,
        instruction_text: str,
        instruction_speech: str,
        bounding_box: List[int],
        visual_cue_type: str = "arrow"
    ) -> Dict[str, Any]:
        """Display instructions visually on the user's screen with voice guidance.

        Args:
            instruction_text: Simple text instruction to display to the user, e.g. "Tap here"
            instruction_speech: Spoken instruction to guide the user, e.g. "Tap the menu icon in the top right corner"
            bounding_box: Bounding box coordinates originally from the interactive_ui_components 
            visual_cue_type: Type of visual cue to display (default: "arrow")
        """
        with langfuse.start_as_current_span(
            name="display_instructions",
            input={
                "instruction_text": instruction_text,
                "instruction_speech": instruction_speech,
                "bounding_box": bounding_box,
                "visual_cue_type": visual_cue_type
            }
        ) as span:
            # Reset cancellation flag at start of new display instruction
            self.cancel_current_display = False
            
            # Stop any previous active display
            if self.display_active:
                logger.info("Stopping previous display instruction")
                await self.stop_display_instructions()
                self.display_active = False
                await self.frame_observer.stop_observation()
            
            if not self.identified:
                # Need to identify screen elements first - do this asynchronously
                logger.info("Screen elements not identified, triggering identification...")
                success = await self.identify_screen_elements(context.session)
                if not success:
                    return {
                        "success": False,
                        "error": "Failed to identify screen elements. Please ensure your screen is clearly visible and try again."
                    }
            
            display_timeout = 30  # 30 seconds timeout for re-display loop
            max_redisplay_attempts = 3  # Prevent infinite loops
            attempt_count = 0
            
            while attempt_count < max_redisplay_attempts:
                # Check for cancellation before each attempt
                if self.cancel_current_display:
                    logger.info("Display instruction cancelled by user speech before attempt")
                    return None  # Return None so no response triggers LLM
                
                attempt_count += 1
                logger.info(f"Display instruction attempt {attempt_count}/{max_redisplay_attempts}")
                
                # Display the instructions
                result = await display_instructions_tool(
                    instruction_text,
                    instruction_speech,
                    bounding_box,
                    visual_cue_type,
                    self.room,
                )
                
                # Fail to display virtual instructions on frontend app, the agent will try to guide with speech only
                if result.get('success') is not True:
                    span.update(level="ERROR")
                    span.update(output=result)
                    # Clean up chat context with failure summary
                    self.summary_message = f"[Assistant fail to display instructions: {instruction_text} at {bounding_box}, tried with speech only]"
                    self.summary_message_time = time.time() + 0.003
                    return result
                
                # Set display active and start observation
                self.display_active = True
                logger.info("Display is active")
                
                # Start observation with long timeout (let display_instructions handle timeout)
                await self.frame_observer.start_observation(
                    on_change_callback=self._on_screen_change
                )
                
                # Set a very long observer timeout - let display_instructions handle the timeout
                self.frame_observer.set_max_observation_time(display_timeout + 10)
                
                # Wait for either frame change or timeout
                start_time = asyncio.get_event_loop().time()
                screen_changed = False
                
                while (asyncio.get_event_loop().time() - start_time) < display_timeout:
                    # Check for cancellation first
                    if self.cancel_current_display:
                        logger.info("Display instruction cancelled by user speech")
                        await self.stop_display_instructions()
                        self.display_active = False
                        await self.frame_observer.stop_observation()
                        return None  # Return None so no response triggers LLM
                    
                    await asyncio.sleep(0.5)
                    
                    # Check if screen changed (frame observer detected change)
                    if not self.frame_observer.is_active:
                        # Frame observer stopped, meaning screen change was detected
                        screen_changed = True
                        logger.info("Screen change detected, display instruction successful")
                        break
                
                if screen_changed:
                    # user followed the instruction - before timeout
                    span.update(output=result)
                    # Clean up chat context with success summary
                    self.summary_message = f"[Assistant displayed instructions: '{instruction_text}' at {bounding_box}]"
                    self.summary_message_time = time.time() + 0.003
                    await self.identify_screen_elements(self.session)

                    return result
                else:
                    # Timeout reached, stop current display and re-display
                    logger.info(f"Display timeout reached after {display_timeout} seconds, attempting re-display")
                    
                    # Stop current display
                    await self.stop_display_instructions()
                    self.display_active = False
                    
                    # Stop observation
                    await self.frame_observer.stop_observation()
                    
                    # Short delay before re-display
                    await asyncio.sleep(1)
            
            # If we've exhausted all attempts
            logger.warning(f"Failed to get user interaction after {max_redisplay_attempts} attempts")
            span.update(output={
                "success": False,
                "error": f"User did not follow instruction after {max_redisplay_attempts} display attempts"
            })
            # Clean up chat context with no response summary
            self.summary_message = f"[Assistant displayed instructions: {instruction_text} at {bounding_box}, but user did not follow the instructions]"
            self.summary_message_time = time.time() + 0.003
            return {
                "success": False,
                "error": f"User did not follow instruction after {max_redisplay_attempts} display attempts. Ask user if they have any other questions. "
            }

    async def _on_screen_change(self):
        """Single callback that handles all screen change scenarios"""
        self.identified = False
        logger.info("Screen changed - reset identified flag")
        
        # Clear cancellation flag since user actually followed instruction
        self.cancel_current_display = False
        
        # If display is active, stop it (user followed instruction)
        if self.display_active:
            logger.info("User followed the instruction - stopping display")
            await self.stop_display_instructions()
            self.display_active = False
        
        # Stop current observation to prevent callback loops
        await self.frame_observer.stop_observation()
        logger.info("Stopped observation - will restart when needed")
            
            
    async def stop_display_instructions(self) -> None:
        """Send RPC to stop display instructions at frontend"""
        try:
            job_ctx = get_job_context()
            room = job_ctx.room
            remote_participants = list(room.remote_participants.values())

            if remote_participants: # check if there is any user
                target_participant = remote_participants[0]
                logger.info(f"Stop display UI. Participant: {target_participant.identity}")
                
                try:
                    response = await room.local_participant.perform_rpc(
                        destination_identity=target_participant.identity,
                        method="clear-navigation-guidance",
                        payload="",
                        response_timeout=3.0
                    )
                    logger.info(f"Stop display successful. Response: {response}")
                    
                except rtc.RpcError as rpc_error:
                    logger.error(f"Stop display RPC failed: {rpc_error.code} - {rpc_error.message}")
                except Exception as e:
                    logger.error(f"Error sending stop display command: {str(e)}")
            else:
                logger.warning("No remote participants found to send stop display command")
                
        except Exception as e:
            logger.error(f"Error stopping display instructions: {e}")

    @function_tool()
    async def hand_back_to_conversation(
        self, 
        context: RunContext[NavigationSessionData],
        completion_message: str = None
    ) -> Agent:
        """Hand back control to conversation agent when the navigation goal is achieved.
        
        Args:
            completion_message: Message to confirm goal completion (e.g., "Great! You've successfully changed the language to English.")
        """
        with langfuse.start_as_current_span(
            name="hand_back_to_conversation",
            input={"goal": self.goal, "completion_message": completion_message}
        ) as span:
            try:
                logger.info(f"Handing back to conversation agent. Goal achieved: {self.goal}")
                
                # Stop any active frame observation before transfer
                await self.frame_observer.stop_observation()
                
                # Mark goal as achieved
                context.userdata.goal_achieved = True
                context.userdata.current_goal = None
                
                if not completion_message:
                    completion_message = f"Perfect! I've helped you {self.goal}. Is there anything else I can help you with?"
                
                # Create conversation agent with preserved context
                conversation_agent = ConversationAgent(
                    room=self.room,
                    chat_ctx=self.session._chat_ctx  # Pass current chat context
                )
                
                # Transfer the complete video stream state back
                conversation_agent.latest_frame = self.latest_frame
                conversation_agent.last_frame_time = self.last_frame_time
                conversation_agent.video_stream = self.video_stream
                conversation_agent._tasks = self._tasks  # Transfer tasks
                conversation_agent.is_on_different_screen = self.is_on_different_screen
                conversation_agent.screen_number = self.screen_number
                conversation_agent.summary_message = None
                conversation_agent.summary_message_time = None
                
                # Don't close video stream on current agent since we're transferring it
                self.video_stream = None  # Remove reference to prevent double closure
                self._tasks = []  # Clear tasks reference
                
                logger.info("Transferred video stream state back to conversation agent")
                
                span.update(output={"success": True, "goal_achieved": self.goal})
                
                return completion_message, conversation_agent
                
            except Exception as e:
                span.update(level="ERROR", status_message=str(e))
                logger.error(f"Error handing back to conversation: {e}")
                raise
            
    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: List[FunctionTool],
        model_settings: ModelSettings
    ) -> AsyncIterable[llm.ChatChunk]:
        """
        A node in the processing pipeline that processes text generation with an LLM.

        By default, this node uses the agent's LLM to process the provided context. It may yield plain text (as str) for straightforward text generation, or ChatChunk objects that can include text and optional tool calls. ChatChunk is helpful for capturing more complex outputs such as function calls, usage statistics, or other metadata.

        Can override this node to customize how the LLM is used or how tool invocations and responses are handled.
        """
        
        dif_screen = self.is_on_different_screen
        logger.info("Is on different screen: %s", dif_screen)
        logger.info("Latest frame exists: %s", bool(self.latest_frame))
        
        # merge assistant message in session chat_ctx into chat_ctx
        # chat_ctx only has function message
        chat_ctx.merge(other_chat_ctx=self.session._chat_ctx, exclude_function_call=True,exclude_instructions=True)
        
        # Handle function messages in chat ctx - for nav agent
        if self.summary_message is not None and self.summary_message_time is not None:
            # just display the instruction
            # remove function message, add summary message, update to session chat ctx
            chat_ctx = chat_ctx.copy(exclude_function_call=True)
            #chat_ctx.add_message(
            #    role="assistant",
            #    content=self.summary_message,
            #    created_at=self.summary_message_time
            #)
            self.summary_message=None
            # Set session chat context to the updated context
            self.session._chat_ctx = chat_ctx
            logger.info("Updated session._chat_ctx with summary message")
            
        # elif self.summary_message is None and self.summary_message_time is not None:
        #     # just identify elements, llm need to know the function response
        #     # or: error messages of display_instructions tool call
        #     # Append new messages (after last display) to session._chat_ctx
        #     if self.last_message_time:
        #         recent_messages = []
        #         for item in chat_ctx.items:
        #             # Check if item has created_at attribute and is newer than summary_message_time
        #             if hasattr(item, 'created_at') and item.created_at > self.last_message_time:
        #                 recent_messages.append(item)
            
        #         if recent_messages:
        #             # Insert the recent messages into session chat context
        #             self.session._chat_ctx.insert(recent_messages)
        #         else:
        #             logger.info("No recent messages found to append to session context")
                    
        #         self.last_message_time = None
        #         chat_ctx = self.session._chat_ctx
        #     else:
        #         logger.error("The last_message_time is null")
        
        
        # Handle screen state and messages
        if self.latest_frame is None:
            # User is not sharing screen: create copy and add temporary message
            temp_chat_ctx = chat_ctx.copy()
            temp_chat_ctx.add_message(
                role="user",
                content="User is not currently sharing their screen."
            )
            logger.info("Added 'not sharing screen' temporary message to context")
        else:
            # User is sharing screen: check if user is on dif screen
            if dif_screen:
                # User moved to a different screen: 
                # - add latest screen frame to PERMANENT context
                # - reset flag
                self.screen_number += 1
                
                image_content = ImageContent(
                    image=self.latest_frame
                )
                
                # Add to PERMANENT context (this should persist)
                chat_ctx.add_message(
                    role="user",
                    content=[f"Screen {self.screen_number}:", image_content]
                )
                logger.info(f"Added screen {self.screen_number} frame to permanent context")
                
                # Create copy after adding to permanent context (so copy includes the new screen)
                temp_chat_ctx = chat_ctx.copy()
                
                self.is_on_different_screen = False
            else:
                # user is on the same screen: create copy and add TEMPORARY message
                temp_chat_ctx = chat_ctx.copy()
                temp_chat_ctx.add_message(
                    role="user",
                    content=f"User is on the screen {self.screen_number}"
                )
                logger.info(f"Added 'same screen' temporary message for screen {self.screen_number}")
            
            # add interactive ui components to the context
            interactive_ui_components = await self.get_interactive_ui_components(self.session)
            temp_chat_ctx.add_message(
                role="user",
                content= interactive_ui_components
            )
            logger.info("Added interactive ui components to the context")

        # Print chat context for debugging (use temp context)
        self.print_chat_context(temp_chat_ctx)

        if self._ending_conversation:
            logger.info("Ending conversation")
            return
        
        # Use context manager for generation
        with langfuse.start_as_current_generation(
            name="llm_generation",
            model="gemini-2.5-flash",
            input=temp_chat_ctx.to_provider_format('google'),
            metadata={
                "temperature": 0.8,
                "is_sharing_screen": self.latest_frame is not None,
                "screen_number": self.screen_number,
                "is_different_screen": dif_screen,
                "agent": self.room.local_participant.name,
                "timestamp": datetime.now(UTC).isoformat()
            }
        ) as generation:
        
            output = ""
            set_completion_start_time = False
            chunks = []
            start_time = time.time()
            
            try:
                # Use temporary context for LLM call, but pass original context for Agent methods
                async for chunk in Agent.default.llm_node(self, temp_chat_ctx, tools, model_settings):
                    if not set_completion_start_time:
                        generation.update(completion_start_time=datetime.now(UTC))
                        set_completion_start_time = True
                    if chunk.delta and chunk.delta.content:
                        output += chunk.delta.content
                    chunks.append(chunk)
                    yield chunk
                                    
                # if self.summary_message is None and self.summary_message_time is not None:
                #     self.last_message_time = (time.time() + start_time)/2
                
            except Exception as e:
                generation.update(level="ERROR", status_message=str(e))
                logger.error(f"LLM error: {e}")
                raise
                
            finally:
                # Calculate response time for performance monitoring            
                logger.info("response_time = %d", time.time() - start_time)
                
                await self.update_chat_ctx(chat_ctx)
                logger.info("Update chat ctx")
                
                final_output = {"role": "assistant", "content": output}
                logger.info(f"Assistant: {output}")
                generation.update(output=final_output)

    async def stt_node(
        self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
    ) -> Optional[AsyncIterable[stt.SpeechEvent]]:
        """STT node for NavigationAgent with custom processing."""
        
        with langfuse.start_as_current_span(
            name="navigation_stt_node", 
            metadata={"model": "google", "agent": "navigation"}
        ) as span:
            try:
                async for event in Agent.default.stt_node(self, audio, model_settings):
                    if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                        logger.info(f"NavigationAgent - Speech recognized: {event.alternatives[0].text[:50]}...")
                    yield event
            except Exception as e:
                span.update(level="ERROR", status_message=str(e))
                logger.error(f"NavigationAgent STT error: {e}")
                raise

    async def tts_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterable[rtc.AudioFrame]:
        """TTS node for NavigationAgent with custom processing."""
        
        with langfuse.start_as_current_span(
            name="navigation_tts_node", 
            metadata={"model": "google cloud tts", "agent": "navigation"}  
        ) as span:
            try:
                async for event in Agent.default.tts_node(self, text, model_settings):
                    yield event
            except Exception as e:
                span.update(level="ERROR", status_message=str(e))
                logger.error(f"NavigationAgent TTS error: {e}")
                raise

async def entrypoint(ctx: JobContext) -> None:
    # Connect to the room
    await ctx.connect()

    logger.info(f"Connected to room: {ctx.room.name}")
    logger.info(f"Local participant: {ctx.room.local_participant.identity}")
        
    # Wait for a remote participant to join
    await ctx.wait_for_participant()
    logger.info("👤 Participant joined the room")

    logger.info(f"Found {len(ctx.room.remote_participants)} remote participants")
    
    # Create session with userdata for navigation state
    session = AgentSession[NavigationSessionData](
        userdata=NavigationSessionData(),
        max_tool_steps=8
    )

    # Start with the ConversationAgent
    conversation_agent = ConversationAgent(room=ctx.room)
    
    # Set up room input/output - enable all modes
    room_input = RoomInputOptions(
        video_enabled=True,
        audio_enabled=True
    )
    
    room_output = RoomOutputOptions(
        audio_enabled=True,
        transcription_enabled=True
    )

    # Start the session with conversation agent
    await session.start(
        agent=conversation_agent,
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
import asyncio
import logging
import time
import io
import aiohttp
import base64
import os
import json
from datetime import UTC, datetime
from typing import Union, AsyncIterable, Optional, List, Any, Dict
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
from livekit.plugins import deepgram, openai, silero
from livekit.agents.utils.images.image import encode, EncodeOptions
from livekit.agents.utils.images.image import ResizeOptions
from livekit.plugins.turn_detector.english import EnglishModel
from PIL import Image
#from knowledge_manager import KnowledgeManager

logger = logging.getLogger("openai-video-agent")
logger.setLevel(logging.INFO)

load_dotenv()

_langfuse = Langfuse()

# Initialize knowledge manager
#knowledge_manager = KnowledgeManager()

INSTRUCTIONS = f"""
You are a mobile voice assistant, Solus, who can answer general questions, help with app navigation, and answer questions about the user's screen. 

User can share their screen to show you the issue. But sometimes they just want to talk to you.

You have some powerful tools:
- Look up weather information for a given location
- Identify interactive UI elements and their positions on the screen to help with app navigation. When users ask for help navigating an app, you can use this tool to analyze the screen and guide them to specific buttons, fields, or other interactive elements by providing the position in the result of this tool.

The tools might fail, tell the user if you fail to use the tool and provide another way to help them.
IMPORTANT: Respond in plain text only. Do not use any markdown formatting including bold, italics, bullet points, numbered lists, or other markdown syntax. Your responses will be read aloud by text-to-speech.

When screen sharing is available, state what you see briefly if you don't receive any query from user.

If user's query need their screen context but no screen sharing is detected, let the user know they need to share their screen for visual assistance.

Keep responses short, maximum 100 words while staying helpful and accurate.


"""
# {knowledge_manager.format_knowledge()}

class VideoAgent(Agent):
    def __init__(self, instructions: str, room: rtc.Room) -> None:
        super().__init__(
            instructions=instructions,
            llm=openai.LLM(model="gpt-4.1-mini-2025-04-14"),
            stt=deepgram.STT(),
            tts=deepgram.TTS(),
            vad=silero.VAD.load(),
            turn_detection=EnglishModel(),
        )
        self.room = room
        self.session_id = str(uuid4())
        self.current_trace = None

        self.frames: List[rtc.VideoFrame] = []
        self.last_frame_time: float = 0
        self.video_stream: Optional[rtc.VideoStream] = None

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
        logger.info(f"Getting weather for {location}")
        
        # Create a span in Langfuse for tracking
        span = self.get_current_trace().span(name="weather_lookup", metadata={"location": location})
        
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

    @function_tool()
    async def identify_screen_elements(
        self,
        context: RunContext,
    ) -> Dict[str, Any]:
        """Identify interactive elements and their exact positions on the user's screen.
        
        This tool analyzes the most recent screen capture to identify interactive elements along with their positions on screen.
        """
        logger.info("Identifying interactive elements on screen")
        
        # Create a span in Langfuse for tracking
        span = self.get_current_trace().span(name="screen_element_identification")
        
        try:
            # Check if we have any frames
            if not self.frames and not self.video_stream:
                logger.warning("No screen frames available")
                return {
                    "success": False,
                    "error": "No screen sharing detected. Ask the user to share their screen.",
                    "elements": []
                }
            
            # Get the most recent frame
            most_recent_frame = self.frames[-1] if self.frames else None
            
            if not most_recent_frame:
                logger.warning("No recent frame available")
                return {
                    "success": False,
                    "error": "No recent screen capture available. Ask the user to share their screen.",
                    "elements": []
                }            

            # tell the user that the agent is analyzing the screen, have to wait
            self.session.say(
                text="I am analyzing the screen... Please give me a minute..."
            )

            # Encode the VideoFrame to JPEG bytes
            options = EncodeOptions(format="JPEG", quality=85)
            image_bytes = encode(most_recent_frame, options)

            # Encode as base64
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')

            # Initialize Gemini client
            client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
            model = "gemini-2.5-flash-preview-05-20"

            # Prepare the prompt
            prompt = """Give the segmentation masks for the interactive components. Output a JSON list of segmentation masks where each entry contains the 2D bounding box in the key "box_2d", the segmentation mask in key "mask", and the text label in the key "label". Use descriptive labels."""

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

            span.update(input={"mime_type": "image/jpeg", "data": image_b64, "text": prompt})
            
            # Configure generation
            generate_content_config = types.GenerateContentConfig(
                max_output_tokens=4096,
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
            raw_analysis = response.text
            
            logger.info(f"Received Gemini analysis: {raw_analysis[:100]}...")
            
            # Filter out mask content from the response
            filtered_elements = self.filter_mask_content(raw_analysis)
            
            return {
                "success": True,
                "elements": filtered_elements,
                "timestamp": datetime.now(UTC).isoformat()
            }
            
        except Exception as e:
            error_msg = f"Screen element identification error: {str(e)}"
            logger.error(error_msg)
            span.update(level="ERROR")
            return {
                "success": False,
                "error": error_msg,
                "elements": []
            }
        finally:
            span.end()

    def filter_mask_content(self, raw_response: str) -> List[Dict[str, Any]]:
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

    async def close(self) -> None:
        await self.close_video_stream()
        if self.current_trace:
            self.current_trace = None
        _langfuse.flush()

    async def close_video_stream(self) -> None:
        if self.video_stream:
            await self.video_stream.aclose()
            self.video_stream = None

    async def on_enter(self) -> None:
        # Just generate a basic intro without video reference
        self.session.generate_reply(
            instructions="introduce yourself very briefly and simply talk about what you have seen if you receive any screen content, not exceeding 20 words"
        )
        self.session.on("user_state_changed", self.on_user_state_change)
        self.session.on("agent_state_changed", self.on_agent_state_change)
        self.room.on("track_subscribed", self.on_track_subscribed)

    async def on_exit(self) -> None:
        await self.session.generate_reply(
            instructions="tell the user a friendly goodbye before you exit",
        )
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
        frames_to_use = self.current_frames()

        if frames_to_use:
            for position, frame in frames_to_use:
                # Use the original frame for LLM context
                image_content = ImageContent(
                    image=frame,
                    inference_detail="high"
                )
                copied_ctx.add_message(
                    role="user",
                    content=[f"{position.title()} view of user during speech:", image_content]
                )
                logger.info(f"Added {position} frame to chat context")
        else:
            # No frames available - user is not sharing their screen
            copied_ctx.add_message(
                role="system",
                content="The user is not currently sharing their screen. Let them know they need to share their screen for you to provide visual assistance."
            )
            logger.warning("No captured frames available for this conversation")

        messages = openai.utils.to_chat_ctx(copied_ctx, cache_key=self.llm)
        
        generation = self.get_current_trace().generation(
            name="llm_generation",
            model="gpt-4.1-mini-2025-04-14",
            input=messages,
        )
        output = ""
        set_completion_start_time = False
        try:
            async for chunk in Agent.default.llm_node(self, copied_ctx, tools, model_settings):
                if not set_completion_start_time:
                    generation.update(
                        completion_start_time=datetime.now(UTC),
                    )
                    set_completion_start_time = True
                if chunk.delta and chunk.delta.content:
                    output += chunk.delta.content
                yield chunk
        except Exception as e:
            generation.update(level="ERROR")
            logger.error(f"LLM error: {e}")
            raise
        finally:
            generation.end(output=output)

    async def tts_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterable[rtc.AudioFrame]:
        span = self.get_current_trace().span(name="tts_node", metadata={"model": "deepgram"})
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
        async for event in video_stream:
            # Capture frames at 1 per second
            current_time = time.time()
            if current_time - self.last_frame_time >= 1.0:
                # Store the frame and update time
                frame = event.frame
                self.frames.append(frame)
                self.last_frame_time = current_time

                frame_count += 1
                logger.info(f"Captured frame #{frame_count}: {frame.width}x{frame.height}")
        logger.info(f"Video frame capture ended - captured {frame_count} frames")

    def current_frames(self) -> List[rtc.VideoFrame]:
        # Add strategic frames from the conversation to provide better context
        # We'll use the first and last frames if available, plus a middle frame for longer sequences
        current_frames = []
        if len(self.frames) > 0:
            # Always use the most recent frame
            current_frames.append(("most recent", self.frames[-1]))

            # For sequences with multiple frames, also include the first frame
            if len(self.frames) >= 3:
                current_frames.append(("first", self.frames[0]))

                # For longer sequences (5+ frames), also include a middle frame
                if len(self.frames) >= 5:
                    mid_idx = len(self.frames) // 2
                    current_frames.append(("middle", self.frames[mid_idx]))
        logger.info(f"Adding {len(current_frames)} frames to conversation (from {len(self.frames)} available)")
        # clear the frames after using them to avoid memory bloat
        self.frames = []
        # return frames in reverse order so earliest frames appear first in context
        return list(reversed(current_frames))


async def entrypoint(ctx: JobContext) -> None:
    # Connect to the room
    await ctx.connect()

    logger.info(f"Connected to room: {ctx.room.name}")
    logger.info(f"Local participant: {ctx.room.local_participant.identity}")

    if len(ctx.room.remote_participants) == 0:
        logger.info("No remote participants in room, exiting")
        return

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


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
from contextlib import asynccontextmanager
import os
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from livekit.api import LiveKitAPI, ListRoomsRequest, AccessToken, VideoGrants, CreateRoomRequest
from pydantic import BaseModel
import uvicorn

load_dotenv()

# Get LiveKit API key and secret from environment variables
api_key = os.environ.get('LIVEKIT_API_KEY')
api_secret = os.environ.get('LIVEKIT_API_SECRET')
livekit_url = os.environ.get('LIVEKIT_URL')

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development, in production specify domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/get-token")
async def get_token(participant: str):
    if not api_key or not api_secret:
        raise HTTPException(status_code=500, detail="LiveKit API credentials not configured")

    room_name = ""
    async with LiveKitAPI() as client:
        room = await client.room.create_room(
            CreateRoomRequest(
                name=f"test-room-{uuid4().hex}",
                departure_timeout=60,
            ),
        )
        room_name = room.name
    
    token = AccessToken(api_key, api_secret) \
        .with_identity("participant") \
        .with_name(participant) \
        .with_grants(VideoGrants(
            room=room_name,
            room_join=True,
            can_subscribe=True,
            can_publish=True,
            can_publish_data=True,
        ))
    
    return {
        "token": token.to_jwt(),
        "url": livekit_url
    }

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=5000, reload=True)

"""
Frame Observer

Handles frame change observation after display instructions are shown.
Automatically stops display when user interaction is detected through frame changes.
"""

import asyncio
import logging
from typing import Optional, Callable, Tuple
from dataclasses import dataclass

from livekit import rtc

logger = logging.getLogger("frame_observer")

@dataclass
class ObserverState:
    """Manages frame change observation state"""
    is_active: bool = False
    task: Optional[asyncio.Task] = None
    callback: Optional[Callable] = None
    max_observation_time: int = 300  # in seconds

class FrameObserver:
    """
    Observes frame changes after display instructions are shown.
    Automatically stops display when frame changes are detected.
    """
    
    def __init__(self, agent_instance):
        """
        Initialize frame observer.
        
        Args:
            agent_instance: The agent instance that has video frame capabilities
        """
        self.agent = agent_instance
        self.state = ObserverState()
    
    async def start_observation(self, on_change_callback: Optional[Callable] = None) -> None:
        """
        Start observing frame changes after display_instructions is called.
        
        Args:
            on_change_callback: Optional callback to execute when frame change is detected
        """
        logger.info("Start observation, dif screen: %s", bool(self.agent.is_on_different_screen))
        
        if self.state.is_active:
            logger.info("Frame observation already active, stopping previous observer")
            await self.stop_observation()
        
        self.state.is_active = True
        self.state.callback = on_change_callback
        
        # Start the observation task
        self.state.task = asyncio.create_task(self._observe_changes())
        
        # Add to agent's task list for proper cleanup
        if hasattr(self.agent, '_tasks'):
            self.agent._tasks.append(self.state.task)
    
    async def stop_observation(self) -> None:
        """Stop frame change observation"""
        if self.state.is_active:
            self.state.is_active = False
            
            if self.state.task and not self.state.task.done():
                self.state.task.cancel()
                try:
                    await self.state.task
                except asyncio.CancelledError:
                    pass
            
            self.state.task = None
            self.state.callback = None
            logger.info("Stopped frame observation")
    
    async def _observe_changes(self) -> None:
        """Internal method to observe frame changes"""
        logger.info("Started frame observation for display instructions")
        check_count = 0
        
        try:
            while self.state.is_active and check_count < (self.state.max_observation_time * 2):  # Convert to half-second intervals
                await asyncio.sleep(0.5) # Check every half second
                check_count += 1
                
                # Check if agent's is_on_different_screen flag changed to True
                if hasattr(self.agent, 'is_on_different_screen') and self.agent.is_on_different_screen:
                    logger.info("Frame change detected: is_on_different_screen changed to True")
                    
                    # Execute callback if provided - let callback handle all actions
                    if self.state.callback:
                        try:
                            await self.state.callback()
                        except Exception as e:
                            logger.error(f"Error in frame change callback: {e}")
                    
                    # Stop observation after callback execution
                    break
                    
            if check_count >= (self.state.max_observation_time * 2):
                logger.info(f"Frame observation timed out after {self.state.max_observation_time} seconds")
                
                # Do NOT execute callback for timeout scenario - let display_instructions handle timeout
                logger.info("Timeout reached - stopping observation without callback")
                
        except asyncio.CancelledError:
            logger.info("Frame observation cancelled")
        except Exception as e:
            logger.error(f"Error in frame observation: {e}")
        finally:
            self.state.is_active = False
            self.state.task = None
    
    @property 
    def is_active(self) -> bool:
        """Check if observation is currently active"""
        return self.state.is_active
    
    def set_max_observation_time(self, seconds: int) -> None:
        """Set maximum observation time"""
        self.state.max_observation_time = seconds
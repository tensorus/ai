"""
Tensorus Agents
=============

Autonomous agents for Tensorus.
"""

import abc
import logging
import time
from typing import Any, Dict, List, Optional

from tensorus.storage.client import TensorusStorageClient


class Agent(abc.ABC):
    """Base class for Tensorus agents."""
    
    def __init__(
        self,
        storage_client: TensorusStorageClient,
        name: str,
        description: str,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize the agent.
        
        Args:
            storage_client: Storage client for interacting with the Tensorus storage engine.
            name: Name of the agent.
            description: Description of the agent.
            logger: Logger for the agent.
        """
        self.storage_client = storage_client
        self.name = name
        self.description = description
        self.logger = logger or logging.getLogger(f"tensorus.agents.{name}")
        self.start_time = time.time()
        
    @abc.abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """Run the agent.
        
        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
            
        Returns:
            Agent-specific results.
        """
        pass
    
    def log_event(self, event_type: str, message: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log an event.
        
        Args:
            event_type: Type of the event.
            message: Event message.
            metadata: Additional metadata.
        """
        metadata = metadata or {}
        event = {
            "agent": self.name,
            "type": event_type,
            "message": message,
            "timestamp": time.time(),
            "metadata": metadata,
        }
        if event_type == "error":
            self.logger.error(f"{self.name}: {message}", extra={"event": event})
        elif event_type == "warning":
            self.logger.warning(f"{self.name}: {message}", extra={"event": event})
        else:
            self.logger.info(f"{self.name}: {message}", extra={"event": event})
    
    def uptime(self) -> float:
        """Get the uptime of the agent.
        
        Returns:
            Uptime in seconds.
        """
        return time.time() - self.start_time
    
    def status(self) -> Dict[str, Any]:
        """Get the status of the agent.
        
        Returns:
            Agent status.
        """
        return {
            "name": self.name,
            "description": self.description,
            "uptime": self.uptime(),
        } 
"""
Message models for agent communication.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional
import uuid


class MessageType(Enum):
    """Types of messages that can be sent between agents."""
    MARKET_DATA = "MARKET_DATA"
    MARKET_DATA_UPDATE = "MARKET_DATA_UPDATE"
    TRADING_PROPOSAL = "TRADING_PROPOSAL"
    VOTE_REQUEST = "VOTE_REQUEST"
    VOTE_RESPONSE = "VOTE_RESPONSE"
    CONSENSUS_RESULT = "CONSENSUS_RESULT"
    EXECUTION_ORDER = "EXECUTION_ORDER"
    EXECUTION_RESULT = "EXECUTION_RESULT"
    HEALTH_CHECK = "HEALTH_CHECK"
    AGENT_HEARTBEAT = "AGENT_HEARTBEAT"
    SYSTEM_ALERT = "SYSTEM_ALERT"
    PORTFOLIO_UPDATE = "PORTFOLIO_UPDATE"
    OPTIMIZATION_REQUEST = "OPTIMIZATION_REQUEST"
    REBALANCING_REQUEST = "REBALANCING_REQUEST"
    EFFICIENT_FRONTIER_REQUEST = "EFFICIENT_FRONTIER_REQUEST"


@dataclass
class Message:
    """Base message structure for agent communication."""
    message_id: str
    message_type: MessageType
    sender_id: str
    recipient_id: Optional[str]  # None for broadcast messages
    payload: Dict[str, Any]
    timestamp: datetime
    correlation_id: Optional[str] = None
    
    def __post_init__(self):
        """Generate message ID if not provided."""
        if not self.message_id:
            self.message_id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create instance from dictionary."""
        return cls(
            message_id=data["message_id"],
            message_type=MessageType(data["message_type"]),
            sender_id=data["sender_id"],
            recipient_id=data.get("recipient_id"),
            payload=data["payload"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            correlation_id=data.get("correlation_id")
        )


@dataclass
class AgentResponse:
    """Response from an agent after processing a message."""
    response_id: str
    original_message_id: str
    agent_id: str
    success: bool
    data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    processing_time_ms: Optional[float] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        """Set timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if not self.response_id:
            self.response_id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "response_id": self.response_id,
            "original_message_id": self.original_message_id,
            "agent_id": self.agent_id,
            "success": self.success,
            "data": self.data,
            "error_message": self.error_message,
            "processing_time_ms": self.processing_time_ms,
            "timestamp": self.timestamp.isoformat()
        }
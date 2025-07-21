"""
Configuration models for agents and system components.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from enum import Enum


class AgentType(Enum):
    """Types of agents in the system."""
    MARKET_ANALYST = "MARKET_ANALYST"
    RISK_MANAGER = "RISK_MANAGER"
    PORTFOLIO_OPTIMIZER = "PORTFOLIO_OPTIMIZER"
    EXECUTION_AGENT = "EXECUTION_AGENT"


@dataclass
class AgentConfig:
    """Configuration for individual agents."""
    agent_id: str
    agent_type: AgentType
    model_version: str
    parameters: Dict[str, Any]
    resource_limits: Dict[str, Any]
    message_queue_config: Dict[str, Any]
    health_check_interval: int = 30  # seconds
    max_retry_attempts: int = 3
    timeout_seconds: int = 10
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "model_version": self.model_version,
            "parameters": self.parameters,
            "resource_limits": self.resource_limits,
            "message_queue_config": self.message_queue_config,
            "health_check_interval": self.health_check_interval,
            "max_retry_attempts": self.max_retry_attempts,
            "timeout_seconds": self.timeout_seconds
        }


@dataclass
class SystemConfig:
    """System-wide configuration."""
    environment: str  # dev, staging, prod
    consensus_threshold: float = 75.0
    max_disagreement_variance: float = 40.0
    message_broker_url: str = "redis://localhost:6379"
    database_url: str = "postgresql://localhost:5432/trading_system"
    time_series_db_url: str = "influxdb://localhost:8086"
    model_registry_url: str = "http://localhost:5000"
    monitoring_enabled: bool = True
    logging_level: str = "INFO"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "environment": self.environment,
            "consensus_threshold": self.consensus_threshold,
            "max_disagreement_variance": self.max_disagreement_variance,
            "message_broker_url": self.message_broker_url,
            "database_url": self.database_url,
            "time_series_db_url": self.time_series_db_url,
            "model_registry_url": self.model_registry_url,
            "monitoring_enabled": self.monitoring_enabled,
            "logging_level": self.logging_level
        }
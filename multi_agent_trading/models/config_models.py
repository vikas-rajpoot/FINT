"""
Configuration models for agents and system components.
"""

import os
import json
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
from enum import Enum
from pathlib import Path


class ValidationError(Exception):
    """Exception raised for configuration validation errors."""
    pass


class Environment(Enum):
    """Deployment environments."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class AgentType(Enum):
    """Types of agents in the system."""
    MARKET_ANALYST = "MARKET_ANALYST"
    RISK_MANAGER = "RISK_MANAGER"
    PORTFOLIO_OPTIMIZER = "PORTFOLIO_OPTIMIZER"
    EXECUTION_AGENT = "EXECUTION_AGENT"


@dataclass
class ResourceLimits:
    """Resource limits for agents."""
    max_memory_mb: int = 1024
    max_cpu_percent: float = 50.0
    max_disk_mb: int = 500
    max_network_connections: int = 100
    
    def validate(self) -> None:
        """Validate resource limits."""
        if self.max_memory_mb <= 0:
            raise ValidationError("max_memory_mb must be positive")
        if not (0 < self.max_cpu_percent <= 100):
            raise ValidationError("max_cpu_percent must be between 0 and 100")
        if self.max_disk_mb <= 0:
            raise ValidationError("max_disk_mb must be positive")
        if self.max_network_connections <= 0:
            raise ValidationError("max_network_connections must be positive")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_memory_mb": self.max_memory_mb,
            "max_cpu_percent": self.max_cpu_percent,
            "max_disk_mb": self.max_disk_mb,
            "max_network_connections": self.max_network_connections
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResourceLimits':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class MessageQueueConfig:
    """Message queue configuration."""
    redis_url: str = "redis://localhost:6379"
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    dead_letter_ttl: int = 86400  # 24 hours
    handler_timeout: float = 30.0
    connection_pool_size: int = 10
    
    def validate(self) -> None:
        """Validate message queue configuration."""
        if not self.redis_url:
            raise ValidationError("redis_url cannot be empty")
        if self.max_retries < 0:
            raise ValidationError("max_retries must be non-negative")
        if self.base_delay <= 0:
            raise ValidationError("base_delay must be positive")
        if self.max_delay <= 0:
            raise ValidationError("max_delay must be positive")
        if self.dead_letter_ttl <= 0:
            raise ValidationError("dead_letter_ttl must be positive")
        if self.handler_timeout <= 0:
            raise ValidationError("handler_timeout must be positive")
        if self.connection_pool_size <= 0:
            raise ValidationError("connection_pool_size must be positive")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "redis_url": self.redis_url,
            "max_retries": self.max_retries,
            "base_delay": self.base_delay,
            "max_delay": self.max_delay,
            "dead_letter_ttl": self.dead_letter_ttl,
            "handler_timeout": self.handler_timeout,
            "connection_pool_size": self.connection_pool_size
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MessageQueueConfig':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class AgentConfig:
    """Configuration for individual agents."""
    agent_id: str
    agent_type: AgentType
    model_version: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)
    message_queue_config: MessageQueueConfig = field(default_factory=MessageQueueConfig)
    health_check_interval: int = 30  # seconds
    max_retry_attempts: int = 3
    timeout_seconds: int = 10
    environment: Environment = Environment.DEVELOPMENT
    enabled: bool = True
    log_level: str = "INFO"
    model_path: str = ".models"
    
    def __post_init__(self):
        """Post-initialization validation."""
        self.validate()
    
    def validate(self) -> None:
        """Validate agent configuration."""
        if not self.agent_id or not isinstance(self.agent_id, str):
            raise ValidationError("agent_id must be a non-empty string")
        
        if not isinstance(self.agent_type, AgentType):
            raise ValidationError("agent_type must be an AgentType enum")
        
        if not self.model_version or not isinstance(self.model_version, str):
            raise ValidationError("model_version must be a non-empty string")
        
        if not isinstance(self.parameters, dict):
            raise ValidationError("parameters must be a dictionary")
        
        if self.health_check_interval <= 0:
            raise ValidationError("health_check_interval must be positive")
        
        if self.max_retry_attempts < 0:
            raise ValidationError("max_retry_attempts must be non-negative")
        
        if self.timeout_seconds <= 0:
            raise ValidationError("timeout_seconds must be positive")
        
        if not isinstance(self.environment, Environment):
            raise ValidationError("environment must be an Environment enum")
        
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValidationError("log_level must be a valid logging level")
        
        # Validate nested configurations
        self.resource_limits.validate()
        self.message_queue_config.validate()
    
    def get_environment_specific_config(self) -> Dict[str, Any]:
        """Get configuration specific to the current environment."""
        base_config = self.to_dict()
        
        # Environment-specific overrides
        env_overrides = {
            Environment.DEVELOPMENT: {
                "log_level": "DEBUG",
                "health_check_interval": 10,
                "timeout_seconds": 30
            },
            Environment.TESTING: {
                "log_level": "WARNING",
                "health_check_interval": 5,
                "timeout_seconds": 5
            },
            Environment.STAGING: {
                "log_level": "INFO",
                "health_check_interval": 20,
                "timeout_seconds": 15
            },
            Environment.PRODUCTION: {
                "log_level": "WARNING",
                "health_check_interval": 60,
                "timeout_seconds": 10
            }
        }
        
        # Apply environment-specific overrides
        if self.environment in env_overrides:
            base_config.update(env_overrides[self.environment])
        
        return base_config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "model_version": self.model_version,
            "parameters": self.parameters,
            "resource_limits": self.resource_limits.to_dict(),
            "message_queue_config": self.message_queue_config.to_dict(),
            "health_check_interval": self.health_check_interval,
            "max_retry_attempts": self.max_retry_attempts,
            "timeout_seconds": self.timeout_seconds,
            "environment": self.environment.value,
            "enabled": self.enabled,
            "log_level": self.log_level,
            "model_path": self.model_path
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentConfig':
        """Create instance from dictionary."""
        # Convert nested objects
        if "resource_limits" in data and isinstance(data["resource_limits"], dict):
            data["resource_limits"] = ResourceLimits.from_dict(data["resource_limits"])
        
        if "message_queue_config" in data and isinstance(data["message_queue_config"], dict):
            data["message_queue_config"] = MessageQueueConfig.from_dict(data["message_queue_config"])
        
        if "agent_type" in data and isinstance(data["agent_type"], str):
            data["agent_type"] = AgentType(data["agent_type"])
        
        if "environment" in data and isinstance(data["environment"], str):
            data["environment"] = Environment(data["environment"])
        
        return cls(**data)
    
    @classmethod
    def from_file(cls, config_path: str) -> 'AgentConfig':
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
            return cls.from_dict(data)
        except FileNotFoundError:
            raise ValidationError(f"Configuration file not found: {config_path}")
        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON in configuration file: {e}")
    
    def save_to_file(self, config_path: str) -> None:
        """Save configuration to JSON file."""
        try:
            # Create directory if it doesn't exist
            Path(config_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
        except Exception as e:
            raise ValidationError(f"Error saving configuration: {e}")


@dataclass
class SystemConfig:
    """System-wide configuration."""
    environment: Environment = Environment.DEVELOPMENT
    consensus_threshold: float = 75.0
    max_disagreement_variance: float = 40.0
    message_broker_url: str = "redis://localhost:6379"
    database_url: str = "postgresql://localhost:5432/trading_system"
    time_series_db_url: str = "influxdb://localhost:8086"
    model_registry_url: str = "http://localhost:5000"
    monitoring_enabled: bool = True
    logging_level: str = "INFO"
    max_agents: int = 10
    agent_startup_timeout: int = 60
    consensus_timeout: int = 30
    
    def __post_init__(self):
        """Post-initialization validation."""
        self.validate()
    
    def validate(self) -> None:
        """Validate system configuration."""
        if not isinstance(self.environment, Environment):
            raise ValidationError("environment must be an Environment enum")
        
        if not (0 <= self.consensus_threshold <= 100):
            raise ValidationError("consensus_threshold must be between 0 and 100")
        
        if not (0 <= self.max_disagreement_variance <= 100):
            raise ValidationError("max_disagreement_variance must be between 0 and 100")
        
        if not self.message_broker_url:
            raise ValidationError("message_broker_url cannot be empty")
        
        if not self.database_url:
            raise ValidationError("database_url cannot be empty")
        
        if not self.time_series_db_url:
            raise ValidationError("time_series_db_url cannot be empty")
        
        if self.logging_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValidationError("logging_level must be a valid logging level")
        
        if self.max_agents <= 0:
            raise ValidationError("max_agents must be positive")
        
        if self.agent_startup_timeout <= 0:
            raise ValidationError("agent_startup_timeout must be positive")
        
        if self.consensus_timeout <= 0:
            raise ValidationError("consensus_timeout must be positive")
    
    def get_environment_config(self) -> Dict[str, Any]:
        """Get environment-specific configuration."""
        base_config = self.to_dict()
        
        # Environment-specific overrides
        env_overrides = {
            Environment.DEVELOPMENT: {
                "logging_level": "DEBUG",
                "monitoring_enabled": True,
                "consensus_timeout": 60
            },
            Environment.TESTING: {
                "logging_level": "WARNING",
                "monitoring_enabled": False,
                "consensus_timeout": 10
            },
            Environment.STAGING: {
                "logging_level": "INFO",
                "monitoring_enabled": True,
                "consensus_timeout": 20
            },
            Environment.PRODUCTION: {
                "logging_level": "WARNING",
                "monitoring_enabled": True,
                "consensus_timeout": 15
            }
        }
        
        # Apply environment-specific overrides
        if self.environment in env_overrides:
            base_config.update(env_overrides[self.environment])
        
        return base_config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "environment": self.environment.value,
            "consensus_threshold": self.consensus_threshold,
            "max_disagreement_variance": self.max_disagreement_variance,
            "message_broker_url": self.message_broker_url,
            "database_url": self.database_url,
            "time_series_db_url": self.time_series_db_url,
            "model_registry_url": self.model_registry_url,
            "monitoring_enabled": self.monitoring_enabled,
            "logging_level": self.logging_level,
            "max_agents": self.max_agents,
            "agent_startup_timeout": self.agent_startup_timeout,
            "consensus_timeout": self.consensus_timeout
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemConfig':
        """Create instance from dictionary."""
        if "environment" in data and isinstance(data["environment"], str):
            data["environment"] = Environment(data["environment"])
        
        return cls(**data)
    
    @classmethod
    def from_file(cls, config_path: str) -> 'SystemConfig':
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
            return cls.from_dict(data)
        except FileNotFoundError:
            raise ValidationError(f"Configuration file not found: {config_path}")
        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON in configuration file: {e}")
    
    def save_to_file(self, config_path: str) -> None:
        """Save configuration to JSON file."""
        try:
            # Create directory if it doesn't exist
            Path(config_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
        except Exception as e:
            raise ValidationError(f"Error saving configuration: {e}")
    
    @classmethod
    def from_environment(cls) -> 'SystemConfig':
        """Create configuration from environment variables."""
        return cls(
            environment=Environment(os.getenv('TRADING_ENV', 'development')),
            consensus_threshold=float(os.getenv('CONSENSUS_THRESHOLD', '75.0')),
            max_disagreement_variance=float(os.getenv('MAX_DISAGREEMENT_VARIANCE', '40.0')),
            message_broker_url=os.getenv('MESSAGE_BROKER_URL', 'redis://localhost:6379'),
            database_url=os.getenv('DATABASE_URL', 'postgresql://localhost:5432/trading_system'),
            time_series_db_url=os.getenv('TIME_SERIES_DB_URL', 'influxdb://localhost:8086'),
            model_registry_url=os.getenv('MODEL_REGISTRY_URL', 'http://localhost:5000'),
            monitoring_enabled=os.getenv('MONITORING_ENABLED', 'true').lower() == 'true',
            logging_level=os.getenv('LOGGING_LEVEL', 'INFO'),
            max_agents=int(os.getenv('MAX_AGENTS', '10')),
            agent_startup_timeout=int(os.getenv('AGENT_STARTUP_TIMEOUT', '60')),
            consensus_timeout=int(os.getenv('CONSENSUS_TIMEOUT', '30'))
        )
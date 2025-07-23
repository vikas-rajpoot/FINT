"""
Unit tests for configuration models.
"""

import json
import pytest
import tempfile
import os
from pathlib import Path

from multi_agent_trading.models.config_models import (
    AgentConfig, SystemConfig, ResourceLimits, MessageQueueConfig,
    AgentType, Environment, ValidationError
)


class TestResourceLimits:
    """Test ResourceLimits configuration."""
    
    def test_default_values(self):
        """Test default resource limits."""
        limits = ResourceLimits()
        
        assert limits.max_memory_mb == 1024
        assert limits.max_cpu_percent == 50.0
        assert limits.max_disk_mb == 500
        assert limits.max_network_connections == 100
    
    def test_validation_success(self):
        """Test successful validation."""
        limits = ResourceLimits(
            max_memory_mb=2048,
            max_cpu_percent=75.0,
            max_disk_mb=1000,
            max_network_connections=200
        )
        
        # Should not raise exception
        limits.validate()
    
    def test_validation_errors(self):
        """Test validation errors."""
        # Test negative memory
        with pytest.raises(ValidationError, match="max_memory_mb must be positive"):
            limits = ResourceLimits(max_memory_mb=-1)
            limits.validate()
        
        # Test invalid CPU percent
        with pytest.raises(ValidationError, match="max_cpu_percent must be between 0 and 100"):
            limits = ResourceLimits(max_cpu_percent=150.0)
            limits.validate()
        
        # Test zero CPU percent
        with pytest.raises(ValidationError, match="max_cpu_percent must be between 0 and 100"):
            limits = ResourceLimits(max_cpu_percent=0.0)
            limits.validate()
        
        # Test negative disk
        with pytest.raises(ValidationError, match="max_disk_mb must be positive"):
            limits = ResourceLimits(max_disk_mb=-1)
            limits.validate()
        
        # Test negative connections
        with pytest.raises(ValidationError, match="max_network_connections must be positive"):
            limits = ResourceLimits(max_network_connections=-1)
            limits.validate()
    
    def test_serialization(self):
        """Test to_dict and from_dict methods."""
        limits = ResourceLimits(
            max_memory_mb=2048,
            max_cpu_percent=75.0,
            max_disk_mb=1000,
            max_network_connections=200
        )
        
        data = limits.to_dict()
        restored = ResourceLimits.from_dict(data)
        
        assert restored.max_memory_mb == limits.max_memory_mb
        assert restored.max_cpu_percent == limits.max_cpu_percent
        assert restored.max_disk_mb == limits.max_disk_mb
        assert restored.max_network_connections == limits.max_network_connections


class TestMessageQueueConfig:
    """Test MessageQueueConfig configuration."""
    
    def test_default_values(self):
        """Test default message queue configuration."""
        config = MessageQueueConfig()
        
        assert config.redis_url == "redis://localhost:6379"
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.dead_letter_ttl == 86400
        assert config.handler_timeout == 30.0
        assert config.connection_pool_size == 10
    
    def test_validation_success(self):
        """Test successful validation."""
        config = MessageQueueConfig(
            redis_url="redis://remote:6379",
            max_retries=5,
            base_delay=2.0,
            max_delay=120.0,
            dead_letter_ttl=172800,
            handler_timeout=60.0,
            connection_pool_size=20
        )
        
        # Should not raise exception
        config.validate()
    
    def test_validation_errors(self):
        """Test validation errors."""
        # Test empty redis_url
        with pytest.raises(ValidationError, match="redis_url cannot be empty"):
            config = MessageQueueConfig(redis_url="")
            config.validate()
        
        # Test negative max_retries
        with pytest.raises(ValidationError, match="max_retries must be non-negative"):
            config = MessageQueueConfig(max_retries=-1)
            config.validate()
        
        # Test invalid base_delay
        with pytest.raises(ValidationError, match="base_delay must be positive"):
            config = MessageQueueConfig(base_delay=0.0)
            config.validate()
        
        # Test invalid max_delay
        with pytest.raises(ValidationError, match="max_delay must be positive"):
            config = MessageQueueConfig(max_delay=-1.0)
            config.validate()
        
        # Test invalid dead_letter_ttl
        with pytest.raises(ValidationError, match="dead_letter_ttl must be positive"):
            config = MessageQueueConfig(dead_letter_ttl=0)
            config.validate()
        
        # Test invalid handler_timeout
        with pytest.raises(ValidationError, match="handler_timeout must be positive"):
            config = MessageQueueConfig(handler_timeout=0.0)
            config.validate()
        
        # Test invalid connection_pool_size
        with pytest.raises(ValidationError, match="connection_pool_size must be positive"):
            config = MessageQueueConfig(connection_pool_size=0)
            config.validate()
    
    def test_serialization(self):
        """Test to_dict and from_dict methods."""
        config = MessageQueueConfig(
            redis_url="redis://remote:6379",
            max_retries=5,
            base_delay=2.0,
            max_delay=120.0
        )
        
        data = config.to_dict()
        restored = MessageQueueConfig.from_dict(data)
        
        assert restored.redis_url == config.redis_url
        assert restored.max_retries == config.max_retries
        assert restored.base_delay == config.base_delay
        assert restored.max_delay == config.max_delay


class TestAgentConfig:
    """Test AgentConfig configuration."""
    
    def test_default_values(self):
        """Test agent configuration with default values."""
        config = AgentConfig(
            agent_id="test_agent",
            agent_type=AgentType.MARKET_ANALYST,
            model_version="v1.0"
        )
        
        assert config.agent_id == "test_agent"
        assert config.agent_type == AgentType.MARKET_ANALYST
        assert config.model_version == "v1.0"
        assert config.health_check_interval == 30
        assert config.max_retry_attempts == 3
        assert config.timeout_seconds == 10
        assert config.environment == Environment.DEVELOPMENT
        assert config.enabled is True
        assert config.log_level == "INFO"
        assert isinstance(config.resource_limits, ResourceLimits)
        assert isinstance(config.message_queue_config, MessageQueueConfig)
    
    def test_validation_success(self):
        """Test successful validation."""
        config = AgentConfig(
            agent_id="test_agent",
            agent_type=AgentType.RISK_MANAGER,
            model_version="v2.1",
            parameters={"param1": "value1"},
            health_check_interval=60,
            max_retry_attempts=5,
            timeout_seconds=20,
            environment=Environment.PRODUCTION,
            log_level="WARNING"
        )
        
        # Should not raise exception during initialization
        assert config.agent_id == "test_agent"
    
    def test_validation_errors(self):
        """Test validation errors."""
        # Test empty agent_id
        with pytest.raises(ValidationError, match="agent_id must be a non-empty string"):
            AgentConfig(
                agent_id="",
                agent_type=AgentType.MARKET_ANALYST,
                model_version="v1.0"
            )
        
        # Test invalid agent_type
        with pytest.raises(ValidationError, match="agent_type must be an AgentType enum"):
            AgentConfig(
                agent_id="test",
                agent_type="invalid",
                model_version="v1.0"
            )
        
        # Test empty model_version
        with pytest.raises(ValidationError, match="model_version must be a non-empty string"):
            AgentConfig(
                agent_id="test",
                agent_type=AgentType.MARKET_ANALYST,
                model_version=""
            )
        
        # Test invalid parameters type
        with pytest.raises(ValidationError, match="parameters must be a dictionary"):
            AgentConfig(
                agent_id="test",
                agent_type=AgentType.MARKET_ANALYST,
                model_version="v1.0",
                parameters="invalid"
            )
        
        # Test invalid health_check_interval
        with pytest.raises(ValidationError, match="health_check_interval must be positive"):
            AgentConfig(
                agent_id="test",
                agent_type=AgentType.MARKET_ANALYST,
                model_version="v1.0",
                health_check_interval=0
            )
        
        # Test invalid log_level
        with pytest.raises(ValidationError, match="log_level must be a valid logging level"):
            AgentConfig(
                agent_id="test",
                agent_type=AgentType.MARKET_ANALYST,
                model_version="v1.0",
                log_level="INVALID"
            )
    
    def test_environment_specific_config(self):
        """Test environment-specific configuration."""
        config = AgentConfig(
            agent_id="test_agent",
            agent_type=AgentType.MARKET_ANALYST,
            model_version="v1.0",
            environment=Environment.DEVELOPMENT
        )
        
        env_config = config.get_environment_specific_config()
        
        # Development environment should have DEBUG log level
        assert env_config["log_level"] == "DEBUG"
        assert env_config["health_check_interval"] == 10
        assert env_config["timeout_seconds"] == 30
    
    def test_serialization(self):
        """Test to_dict and from_dict methods."""
        config = AgentConfig(
            agent_id="test_agent",
            agent_type=AgentType.PORTFOLIO_OPTIMIZER,
            model_version="v1.5",
            parameters={"param1": "value1", "param2": 42},
            environment=Environment.STAGING,
            log_level="WARNING"
        )
        
        data = config.to_dict()
        restored = AgentConfig.from_dict(data)
        
        assert restored.agent_id == config.agent_id
        assert restored.agent_type == config.agent_type
        assert restored.model_version == config.model_version
        assert restored.parameters == config.parameters
        assert restored.environment == config.environment
        assert restored.log_level == config.log_level
    
    def test_file_operations(self):
        """Test saving to and loading from file."""
        config = AgentConfig(
            agent_id="test_agent",
            agent_type=AgentType.EXECUTION_AGENT,
            model_version="v2.0",
            parameters={"test": True}
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save to file
            config.save_to_file(temp_path)
            
            # Load from file
            restored = AgentConfig.from_file(temp_path)
            
            assert restored.agent_id == config.agent_id
            assert restored.agent_type == config.agent_type
            assert restored.model_version == config.model_version
            assert restored.parameters == config.parameters
            
        finally:
            os.unlink(temp_path)
    
    def test_file_not_found(self):
        """Test loading from non-existent file."""
        with pytest.raises(ValidationError, match="Configuration file not found"):
            AgentConfig.from_file("non_existent_file.json")


class TestSystemConfig:
    """Test SystemConfig configuration."""
    
    def test_default_values(self):
        """Test system configuration with default values."""
        config = SystemConfig()
        
        assert config.environment == Environment.DEVELOPMENT
        assert config.consensus_threshold == 75.0
        assert config.max_disagreement_variance == 40.0
        assert config.message_broker_url == "redis://localhost:6379"
        assert config.monitoring_enabled is True
        assert config.logging_level == "INFO"
        assert config.max_agents == 10
        assert config.agent_startup_timeout == 60
        assert config.consensus_timeout == 30
    
    def test_validation_success(self):
        """Test successful validation."""
        config = SystemConfig(
            environment=Environment.PRODUCTION,
            consensus_threshold=80.0,
            max_disagreement_variance=30.0,
            message_broker_url="redis://prod:6379",
            monitoring_enabled=True,
            logging_level="WARNING",
            max_agents=20
        )
        
        # Should not raise exception during initialization
        assert config.environment == Environment.PRODUCTION
    
    def test_validation_errors(self):
        """Test validation errors."""
        # Test invalid consensus_threshold
        with pytest.raises(ValidationError, match="consensus_threshold must be between 0 and 100"):
            SystemConfig(consensus_threshold=150.0)
        
        # Test invalid max_disagreement_variance
        with pytest.raises(ValidationError, match="max_disagreement_variance must be between 0 and 100"):
            SystemConfig(max_disagreement_variance=-10.0)
        
        # Test empty message_broker_url
        with pytest.raises(ValidationError, match="message_broker_url cannot be empty"):
            SystemConfig(message_broker_url="")
        
        # Test empty database_url
        with pytest.raises(ValidationError, match="database_url cannot be empty"):
            SystemConfig(database_url="")
        
        # Test invalid logging_level
        with pytest.raises(ValidationError, match="logging_level must be a valid logging level"):
            SystemConfig(logging_level="INVALID")
        
        # Test invalid max_agents
        with pytest.raises(ValidationError, match="max_agents must be positive"):
            SystemConfig(max_agents=0)
    
    def test_environment_config(self):
        """Test environment-specific configuration."""
        config = SystemConfig(environment=Environment.TESTING)
        
        env_config = config.get_environment_config()
        
        # Testing environment should have specific settings
        assert env_config["logging_level"] == "WARNING"
        assert env_config["monitoring_enabled"] is False
        assert env_config["consensus_timeout"] == 10
    
    def test_serialization(self):
        """Test to_dict and from_dict methods."""
        config = SystemConfig(
            environment=Environment.STAGING,
            consensus_threshold=85.0,
            max_disagreement_variance=25.0,
            monitoring_enabled=False,
            max_agents=15
        )
        
        data = config.to_dict()
        restored = SystemConfig.from_dict(data)
        
        assert restored.environment == config.environment
        assert restored.consensus_threshold == config.consensus_threshold
        assert restored.max_disagreement_variance == config.max_disagreement_variance
        assert restored.monitoring_enabled == config.monitoring_enabled
        assert restored.max_agents == config.max_agents
    
    def test_from_environment(self):
        """Test creating configuration from environment variables."""
        # Set environment variables
        env_vars = {
            'TRADING_ENV': 'production',
            'CONSENSUS_THRESHOLD': '85.0',
            'MAX_DISAGREEMENT_VARIANCE': '35.0',
            'MESSAGE_BROKER_URL': 'redis://prod:6379',
            'MONITORING_ENABLED': 'false',
            'LOGGING_LEVEL': 'ERROR',
            'MAX_AGENTS': '25'
        }
        
        # Temporarily set environment variables
        original_env = {}
        for key, value in env_vars.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = value
        
        try:
            config = SystemConfig.from_environment()
            
            assert config.environment == Environment.PRODUCTION
            assert config.consensus_threshold == 85.0
            assert config.max_disagreement_variance == 35.0
            assert config.message_broker_url == 'redis://prod:6379'
            assert config.monitoring_enabled is False
            assert config.logging_level == 'ERROR'
            assert config.max_agents == 25
            
        finally:
            # Restore original environment
            for key, value in original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    
    def test_file_operations(self):
        """Test saving to and loading from file."""
        config = SystemConfig(
            environment=Environment.PRODUCTION,
            consensus_threshold=90.0,
            monitoring_enabled=False
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save to file
            config.save_to_file(temp_path)
            
            # Load from file
            restored = SystemConfig.from_file(temp_path)
            
            assert restored.environment == config.environment
            assert restored.consensus_threshold == config.consensus_threshold
            assert restored.monitoring_enabled == config.monitoring_enabled
            
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__])
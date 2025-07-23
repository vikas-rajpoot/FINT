"""
Unit tests for message bus implementation.
"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from multi_agent_trading.services.message_bus import MessageBus, MessageBusError, RetryConfig
from multi_agent_trading.models.message_models import Message, MessageType, AgentResponse


class TestRetryConfig:
    """Test cases for RetryConfig."""
    
    def test_default_config(self):
        """Test default retry configuration."""
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
    
    def test_custom_config(self):
        """Test custom retry configuration."""
        config = RetryConfig(max_retries=5, base_delay=2.0, max_delay=120.0)
        assert config.max_retries == 5
        assert config.base_delay == 2.0
        assert config.max_delay == 120.0
    
    def test_exponential_backoff(self):
        """Test exponential backoff delay calculation."""
        config = RetryConfig(base_delay=1.0, max_delay=10.0)
        
        assert config.get_delay(0) == 1.0  # 1.0 * 2^0
        assert config.get_delay(1) == 2.0  # 1.0 * 2^1
        assert config.get_delay(2) == 4.0  # 1.0 * 2^2
        assert config.get_delay(3) == 8.0  # 1.0 * 2^3
        assert config.get_delay(4) == 10.0  # Capped at max_delay


class TestMessageBus:
    """Test cases for MessageBus."""
    
    @pytest.fixture
    def message_bus_config(self):
        """Message bus configuration for testing."""
        return {
            'redis_url': 'redis://localhost:6379',
            'max_retries': 3,
            'base_delay': 0.1,
            'max_delay': 1.0,
            'dead_letter_ttl': 3600,
            'handler_timeout': 5.0
        }
    
    @pytest.fixture
    def sample_message(self):
        """Sample message for testing."""
        return Message(
            message_id="test_msg_123",
            message_type=MessageType.MARKET_DATA,
            sender_id="market_analyst",
            recipient_id="risk_manager",
            payload={"symbol": "AAPL", "price": 150.0},
            timestamp=datetime.utcnow()
        )
    
    @pytest.fixture
    def sample_response(self):
        """Sample agent response for testing."""
        return AgentResponse(
            response_id="resp_123",
            original_message_id="test_msg_123",
            agent_id="risk_manager",
            success=True,
            result={"risk_score": 0.3}
        )
    
    def test_message_bus_initialization(self, message_bus_config):
        """Test message bus initialization."""
        bus = MessageBus(message_bus_config)
        
        assert bus.config == message_bus_config
        assert not bus.is_connected()
        assert bus._retry_config.max_retries == 3
        assert bus._retry_config.base_delay == 0.1
        assert bus._dead_letter_ttl == 3600
    
    @pytest.mark.asyncio
    async def test_connection_success(self, message_bus_config):
        """Test successful connection to Redis."""
        bus = MessageBus(message_bus_config)
        
        with patch('redis.asyncio.from_url') as mock_redis:
            mock_client = AsyncMock()
            mock_redis.return_value = mock_client
            mock_client.ping = AsyncMock()
            mock_client.pubsub.return_value = AsyncMock()
            
            await bus.connect()
            
            assert bus.is_connected()
            mock_client.ping.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_connection_failure(self, message_bus_config):
        """Test connection failure handling."""
        bus = MessageBus(message_bus_config)
        
        with patch('redis.asyncio.from_url') as mock_redis:
            mock_client = AsyncMock()
            mock_redis.return_value = mock_client
            mock_client.ping.side_effect = Exception("Connection failed")
            
            with pytest.raises(MessageBusError, match="Connection failed"):
                await bus.connect()
            
            assert not bus.is_connected()
    
    @pytest.mark.asyncio
    async def test_publish_success(self, message_bus_config, sample_message):
        """Test successful message publishing."""
        bus = MessageBus(message_bus_config)
        
        with patch('redis.asyncio.from_url') as mock_redis:
            mock_client = AsyncMock()
            mock_redis.return_value = mock_client
            mock_client.ping = AsyncMock()
            mock_client.pubsub.return_value = AsyncMock()
            mock_client.publish = AsyncMock(return_value=1)
            mock_client.setex = AsyncMock()
            
            await bus.connect()
            result = await bus.publish("test_channel", sample_message)
            
            assert result is True
            mock_client.publish.assert_called_once()
            mock_client.setex.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_publish_not_connected(self, message_bus_config, sample_message):
        """Test publishing when not connected."""
        bus = MessageBus(message_bus_config)
        
        with pytest.raises(MessageBusError, match="Message bus not connected"):
            await bus.publish("test_channel", sample_message)
    
    @pytest.mark.asyncio
    async def test_publish_with_retry_success_after_failure(self, message_bus_config, sample_message):
        """Test successful publishing after initial failure."""
        bus = MessageBus(message_bus_config)
        
        with patch('redis.asyncio.from_url') as mock_redis:
            mock_client = AsyncMock()
            mock_redis.return_value = mock_client
            mock_client.ping = AsyncMock()
            mock_client.pubsub.return_value = AsyncMock()
            
            # First call fails, second succeeds
            mock_client.publish.side_effect = [Exception("Network error"), 1]
            mock_client.setex = AsyncMock()
            
            await bus.connect()
            
            with patch('asyncio.sleep', new_callable=AsyncMock):
                result = await bus.publish("test_channel", sample_message)
            
            assert result is True
            assert mock_client.publish.call_count == 2
    
    @pytest.mark.asyncio
    async def test_publish_retry_exhausted(self, message_bus_config, sample_message):
        """Test publishing when all retries are exhausted."""
        bus = MessageBus(message_bus_config)
        
        with patch('redis.asyncio.from_url') as mock_redis:
            mock_client = AsyncMock()
            mock_redis.return_value = mock_client
            mock_client.ping = AsyncMock()
            mock_client.pubsub.return_value = AsyncMock()
            mock_client.publish.side_effect = Exception("Persistent error")
            mock_client.setex = AsyncMock()  # For dead letter queue
            
            await bus.connect()
            
            with patch('asyncio.sleep', new_callable=AsyncMock):
                result = await bus.publish("test_channel", sample_message)
            
            assert result is False
            assert mock_client.publish.call_count == 4  # Initial + 3 retries
            # Should have called setex for dead letter queue
            mock_client.setex.assert_called()
    
    @pytest.mark.asyncio
    async def test_subscribe_success(self, message_bus_config):
        """Test successful subscription to channel."""
        bus = MessageBus(message_bus_config)
        
        async def mock_handler(message: Message) -> AgentResponse:
            return AgentResponse("", message.message_id, "test_agent", True)
        
        with patch('redis.asyncio.from_url') as mock_redis:
            mock_client = AsyncMock()
            mock_pubsub = AsyncMock()
            mock_redis.return_value = mock_client
            mock_client.ping = AsyncMock()
            mock_client.pubsub = MagicMock(return_value=mock_pubsub)
            mock_pubsub.subscribe = AsyncMock()
            
            # Mock the message handler task creation
            with patch('asyncio.create_task') as mock_create_task:
                mock_task = AsyncMock()
                mock_create_task.return_value = mock_task
                
                await bus.connect()
                await bus.subscribe("test_channel", mock_handler)
                
                assert "test_channel" in bus._subscribers
                assert "test_channel" in bus._message_handlers
                mock_pubsub.subscribe.assert_called_once_with("test_channel")
    
    @pytest.mark.asyncio
    async def test_subscribe_not_connected(self, message_bus_config):
        """Test subscription when not connected."""
        bus = MessageBus(message_bus_config)
        
        async def mock_handler(message: Message) -> AgentResponse:
            return AgentResponse("", message.message_id, "test_agent", True)
        
        with pytest.raises(MessageBusError, match="Message bus not connected"):
            await bus.subscribe("test_channel", mock_handler)
    
    @pytest.mark.asyncio
    async def test_unsubscribe_success(self, message_bus_config):
        """Test successful unsubscription from channel."""
        bus = MessageBus(message_bus_config)
        
        async def mock_handler(message: Message) -> AgentResponse:
            return AgentResponse("", message.message_id, "test_agent", True)
        
        with patch('redis.asyncio.from_url') as mock_redis:
            mock_client = AsyncMock()
            mock_pubsub = AsyncMock()
            mock_redis.return_value = mock_client
            mock_client.ping = AsyncMock()
            mock_client.pubsub = MagicMock(return_value=mock_pubsub)
            mock_pubsub.subscribe = AsyncMock()
            mock_pubsub.unsubscribe = AsyncMock()
            
            with patch('asyncio.create_task') as mock_create_task:
                mock_task = AsyncMock()
                mock_create_task.return_value = mock_task
                
                await bus.connect()
                await bus.subscribe("test_channel", mock_handler)
                
                await bus.unsubscribe("test_channel")
                
                assert "test_channel" not in bus._subscribers
                assert "test_channel" not in bus._message_handlers
                mock_task.cancel.assert_called_once()
                mock_pubsub.unsubscribe.assert_called_once_with("test_channel")
    
    @pytest.mark.asyncio
    async def test_disconnect_cleanup(self, message_bus_config):
        """Test proper cleanup during disconnect."""
        bus = MessageBus(message_bus_config)
        
        with patch('redis.asyncio.from_url') as mock_redis:
            mock_client = AsyncMock()
            mock_pubsub = AsyncMock()
            mock_redis.return_value = mock_client
            mock_client.ping = AsyncMock()
            mock_client.pubsub = MagicMock(return_value=mock_pubsub)
            mock_client.close = AsyncMock()
            mock_pubsub.close = AsyncMock()
            
            await bus.connect()
            
            # Add mock message handler with proper mock
            mock_task = MagicMock()
            mock_task.cancel = MagicMock()
            bus._message_handlers["test_channel"] = mock_task
            
            with patch('asyncio.gather', new_callable=AsyncMock) as mock_gather:
                await bus.disconnect()
            
            assert not bus.is_connected()
            mock_task.cancel.assert_called_once()
            mock_pubsub.close.assert_called_once()
            mock_client.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_dead_letter_messages(self, message_bus_config):
        """Test retrieving dead letter queue messages."""
        bus = MessageBus(message_bus_config)
        
        with patch('redis.asyncio.from_url') as mock_redis:
            mock_client = AsyncMock()
            mock_redis.return_value = mock_client
            mock_client.ping = AsyncMock()
            mock_client.pubsub.return_value = AsyncMock()
            
            # Mock dead letter queue data
            dlq_data = {
                "original_channel": "test_channel",
                "message_id": "msg_123",
                "error": "Test error",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            mock_client.keys.return_value = ["dlq:msg_123"]
            mock_client.get.return_value = json.dumps(dlq_data)
            
            await bus.connect()
            messages = await bus.get_dead_letter_messages()
            
            assert len(messages) == 1
            assert messages[0]["message_id"] == "msg_123"
            assert messages[0]["error"] == "Test error"
    
    @pytest.mark.asyncio
    async def test_reprocess_dead_letter_message_success(self, message_bus_config):
        """Test successful reprocessing of dead letter message."""
        bus = MessageBus(message_bus_config)
        
        with patch('redis.asyncio.from_url') as mock_redis:
            mock_client = AsyncMock()
            mock_redis.return_value = mock_client
            mock_client.ping = AsyncMock()
            mock_client.pubsub.return_value = AsyncMock()
            
            # Mock dead letter queue data
            dlq_data = {
                "original_channel": "test_channel",
                "message_data": '{"message_id": "msg_123"}',
                "message_id": "msg_123"
            }
            
            mock_client.get.return_value = json.dumps(dlq_data)
            mock_client.publish.return_value = 1
            mock_client.setex = AsyncMock()
            mock_client.delete = AsyncMock()
            
            await bus.connect()
            result = await bus.reprocess_dead_letter_message("msg_123")
            
            assert result is True
            mock_client.delete.assert_called_once_with("dlq:msg_123")
    
    @pytest.mark.asyncio
    async def test_reprocess_dead_letter_message_not_found(self, message_bus_config):
        """Test reprocessing when dead letter message not found."""
        bus = MessageBus(message_bus_config)
        
        with patch('redis.asyncio.from_url') as mock_redis:
            mock_client = AsyncMock()
            mock_redis.return_value = mock_client
            mock_client.ping = AsyncMock()
            mock_client.pubsub.return_value = AsyncMock()
            mock_client.get.return_value = None  # Message not found
            
            await bus.connect()
            result = await bus.reprocess_dead_letter_message("msg_123")
            
            assert result is False
    
    @pytest.mark.asyncio
    async def test_message_processing_timeout(self, message_bus_config, sample_message):
        """Test message handler timeout handling."""
        bus = MessageBus(message_bus_config)
        
        async def slow_handler(message: Message) -> AgentResponse:
            await asyncio.sleep(10)  # Longer than timeout
            return AgentResponse("", message.message_id, "test_agent", True)
        
        with patch('redis.asyncio.from_url') as mock_redis:
            mock_client = AsyncMock()
            mock_redis.return_value = mock_client
            mock_client.ping = AsyncMock()
            mock_client.pubsub.return_value = AsyncMock()
            mock_client.publish = AsyncMock()
            
            await bus.connect()
            
            response = await bus._call_handler_safely(slow_handler, sample_message)
            
            assert not response.success
            assert "timeout" in response.error_message.lower()
    
    @pytest.mark.asyncio
    async def test_message_processing_error(self, message_bus_config, sample_message):
        """Test message handler error handling."""
        bus = MessageBus(message_bus_config)
        
        async def error_handler(message: Message) -> AgentResponse:
            raise ValueError("Handler error")
        
        with patch('redis.asyncio.from_url') as mock_redis:
            mock_client = AsyncMock()
            mock_redis.return_value = mock_client
            mock_client.ping = AsyncMock()
            mock_client.pubsub.return_value = AsyncMock()
            
            await bus.connect()
            
            response = await bus._call_handler_safely(error_handler, sample_message)
            
            assert not response.success
            assert "Handler error" in response.error_message
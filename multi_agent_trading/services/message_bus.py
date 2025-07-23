"""
Message bus implementation for agent communication.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, Callable, Optional, List
from datetime import datetime, timedelta
import redis.asyncio as redis
import pika
from pika.adapters.asyncio_connection import AsyncioConnection
from ..models.message_models import Message, AgentResponse, MessageType


class MessageBusError(Exception):
    """Custom exception for message bus errors."""
    pass


class RetryConfig:
    """Configuration for retry logic."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
    
    def get_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay."""
        delay = self.base_delay * (2 ** attempt)
        return min(delay, self.max_delay)


class MessageBus:
    """
    Message bus for reliable communication between agents.
    
    Provides publish/subscribe functionality with message acknowledgment
    and retry logic using Redis for high-performance messaging.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize message bus.
        
        Args:
            config: Message bus configuration containing:
                - redis_url: Redis connection URL
                - retry_config: Retry configuration
                - dead_letter_ttl: TTL for dead letter queue messages
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._redis_client: Optional[redis.Redis] = None
        self._pubsub: Optional[redis.client.PubSub] = None
        self._subscribers: Dict[str, Callable] = {}
        self._is_connected = False
        self._retry_config = RetryConfig(
            max_retries=config.get('max_retries', 3),
            base_delay=config.get('base_delay', 1.0),
            max_delay=config.get('max_delay', 60.0)
        )
        self._dead_letter_ttl = config.get('dead_letter_ttl', 86400)  # 24 hours
        self._message_handlers: Dict[str, asyncio.Task] = {}
    
    async def connect(self) -> None:
        """Connect to Redis message broker."""
        try:
            redis_url = self.config.get('redis_url', 'redis://localhost:6379')
            self._redis_client = redis.from_url(redis_url, decode_responses=True)
            
            # Test connection
            await self._redis_client.ping()
            
            self._pubsub = self._redis_client.pubsub()
            self._is_connected = True
            self.logger.info("Message bus connected to Redis")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            raise MessageBusError(f"Connection failed: {e}")
    
    async def disconnect(self) -> None:
        """Disconnect from message broker."""
        try:
            # Cancel all message handlers
            for task in self._message_handlers.values():
                task.cancel()
            
            # Wait for handlers to complete
            if self._message_handlers:
                await asyncio.gather(*self._message_handlers.values(), return_exceptions=True)
            
            # Close pubsub and redis connections
            if self._pubsub:
                await self._pubsub.close()
            
            if self._redis_client:
                await self._redis_client.close()
            
            self._is_connected = False
            self.logger.info("Message bus disconnected")
            
        except Exception as e:
            self.logger.error(f"Error during disconnect: {e}")
    
    async def publish(self, channel: str, message: Message, with_retry: bool = True) -> bool:
        """
        Publish message to channel with retry logic.
        
        Args:
            channel: Channel to publish to
            message: Message to publish
            with_retry: Whether to use retry logic
            
        Returns:
            True if successful, False otherwise
        """
        if not self._is_connected or not self._redis_client:
            raise MessageBusError("Message bus not connected")
        
        message_data = json.dumps(message.to_dict())
        
        if with_retry:
            return await self._publish_with_retry(channel, message_data, message.message_id)
        else:
            return await self._publish_once(channel, message_data, message.message_id)
    
    async def _publish_with_retry(self, channel: str, message_data: str, message_id: str) -> bool:
        """Publish message with exponential backoff retry."""
        last_error = None
        
        for attempt in range(self._retry_config.max_retries + 1):
            try:
                success = await self._publish_once(channel, message_data, message_id)
                if success:
                    if attempt > 0:
                        self.logger.info(f"Message {message_id} published successfully after {attempt} retries")
                    return True
                    
            except Exception as e:
                last_error = e
                self.logger.warning(f"Publish attempt {attempt + 1} failed for message {message_id}: {e}")
            
            if attempt < self._retry_config.max_retries:
                delay = self._retry_config.get_delay(attempt)
                await asyncio.sleep(delay)
        
        # All retries failed, send to dead letter queue
        await self._send_to_dead_letter_queue(channel, message_data, message_id, str(last_error))
        return False
    
    async def _publish_once(self, channel: str, message_data: str, message_id: str) -> bool:
        """Attempt to publish message once."""
        try:
            # Publish to main channel
            result = await self._redis_client.publish(channel, message_data)
            
            # Store message for acknowledgment tracking
            await self._redis_client.setex(
                f"msg:{message_id}",
                300,  # 5 minutes TTL
                message_data
            )
            
            self.logger.debug(f"Published message {message_id} to channel {channel}, {result} subscribers")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to publish message {message_id}: {e}")
            raise
    
    async def _send_to_dead_letter_queue(self, channel: str, message_data: str, message_id: str, error: str) -> None:
        """Send failed message to dead letter queue."""
        try:
            dead_letter_data = {
                "original_channel": channel,
                "message_data": message_data,
                "message_id": message_id,
                "error": error,
                "timestamp": datetime.utcnow().isoformat(),
                "retry_count": self._retry_config.max_retries
            }
            
            await self._redis_client.setex(
                f"dlq:{message_id}",
                self._dead_letter_ttl,
                json.dumps(dead_letter_data)
            )
            
            self.logger.error(f"Message {message_id} sent to dead letter queue after {self._retry_config.max_retries} retries")
            
        except Exception as e:
            self.logger.critical(f"Failed to send message {message_id} to dead letter queue: {e}")
    
    async def subscribe(self, channel: str, handler: Callable[[Message], AgentResponse]) -> None:
        """
        Subscribe to channel with message handler.
        
        Args:
            channel: Channel to subscribe to
            handler: Message handler function
        """
        if not self._is_connected or not self._pubsub:
            raise MessageBusError("Message bus not connected")
        
        try:
            await self._pubsub.subscribe(channel)
            self._subscribers[channel] = handler
            
            # Start message handler task
            task = asyncio.create_task(self._handle_messages(channel, handler))
            self._message_handlers[channel] = task
            
            self.logger.info(f"Subscribed to channel {channel}")
            
        except Exception as e:
            self.logger.error(f"Failed to subscribe to channel {channel}: {e}")
            raise MessageBusError(f"Subscription failed: {e}")
    
    async def _handle_messages(self, channel: str, handler: Callable[[Message], AgentResponse]) -> None:
        """Handle incoming messages for a subscribed channel."""
        try:
            async for message in self._pubsub.listen():
                if message['type'] == 'message' and message['channel'] == channel:
                    await self._process_message(message['data'], handler)
                    
        except asyncio.CancelledError:
            self.logger.info(f"Message handler for channel {channel} cancelled")
        except Exception as e:
            self.logger.error(f"Error in message handler for channel {channel}: {e}")
    
    async def _process_message(self, message_data: str, handler: Callable[[Message], AgentResponse]) -> None:
        """Process individual message with error handling."""
        try:
            # Parse message
            message_dict = json.loads(message_data)
            message = Message.from_dict(message_dict)
            
            # Record processing start time
            start_time = time.time()
            
            # Call handler
            response = await self._call_handler_safely(handler, message)
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            response.processing_time_ms = processing_time
            
            # Send acknowledgment
            await self._send_acknowledgment(message, response)
            
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            # Send negative acknowledgment
            error_response = AgentResponse(
                response_id="",
                original_message_id=message_dict.get('message_id', 'unknown'),
                agent_id="message_bus",
                success=False,
                error_message=str(e)
            )
            await self._send_acknowledgment(None, error_response)
    
    async def _call_handler_safely(self, handler: Callable[[Message], AgentResponse], message: Message) -> AgentResponse:
        """Call message handler with timeout and error handling."""
        try:
            # Set timeout for handler execution
            timeout = self.config.get('handler_timeout', 30.0)
            response = await asyncio.wait_for(handler(message), timeout=timeout)
            return response
            
        except asyncio.TimeoutError:
            self.logger.error(f"Handler timeout for message {message.message_id}")
            return AgentResponse(
                response_id="",
                original_message_id=message.message_id,
                agent_id="unknown",
                success=False,
                error_message="Handler timeout"
            )
        except Exception as e:
            self.logger.error(f"Handler error for message {message.message_id}: {e}")
            return AgentResponse(
                response_id="",
                original_message_id=message.message_id,
                agent_id="unknown",
                success=False,
                error_message=str(e)
            )
    
    async def _send_acknowledgment(self, original_message: Optional[Message], response: AgentResponse) -> None:
        """Send acknowledgment for processed message."""
        try:
            ack_channel = "acknowledgments"
            ack_message = Message(
                message_id="",
                message_type=MessageType.SYSTEM_ALERT,
                sender_id="message_bus",
                recipient_id=original_message.sender_id if original_message else None,
                payload=response.to_dict(),
                timestamp=datetime.utcnow()
            )
            
            await self.publish(ack_channel, ack_message, with_retry=False)
            
        except Exception as e:
            self.logger.error(f"Failed to send acknowledgment: {e}")
    
    async def unsubscribe(self, channel: str) -> None:
        """
        Unsubscribe from channel.
        
        Args:
            channel: Channel to unsubscribe from
        """
        try:
            if channel in self._message_handlers:
                # Cancel message handler task
                self._message_handlers[channel].cancel()
                del self._message_handlers[channel]
            
            if self._pubsub and channel in self._subscribers:
                await self._pubsub.unsubscribe(channel)
                del self._subscribers[channel]
                self.logger.info(f"Unsubscribed from channel {channel}")
                
        except Exception as e:
            self.logger.error(f"Error unsubscribing from channel {channel}: {e}")
    
    def is_connected(self) -> bool:
        """Check if message bus is connected."""
        return self._is_connected
    
    async def get_dead_letter_messages(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve messages from dead letter queue.
        
        Args:
            limit: Maximum number of messages to retrieve
            
        Returns:
            List of dead letter queue messages
        """
        if not self._redis_client:
            return []
        
        try:
            # Get all dead letter queue keys
            dlq_keys = await self._redis_client.keys("dlq:*")
            
            messages = []
            for key in dlq_keys[:limit]:
                message_data = await self._redis_client.get(key)
                if message_data:
                    messages.append(json.loads(message_data))
            
            return messages
            
        except Exception as e:
            self.logger.error(f"Error retrieving dead letter messages: {e}")
            return []
    
    async def reprocess_dead_letter_message(self, message_id: str) -> bool:
        """
        Reprocess a message from the dead letter queue.
        
        Args:
            message_id: ID of the message to reprocess
            
        Returns:
            True if successful, False otherwise
        """
        if not self._redis_client:
            return False
        
        try:
            # Get message from dead letter queue
            dlq_key = f"dlq:{message_id}"
            dlq_data = await self._redis_client.get(dlq_key)
            
            if not dlq_data:
                self.logger.warning(f"Dead letter message {message_id} not found")
                return False
            
            dlq_message = json.loads(dlq_data)
            
            # Attempt to republish
            success = await self._publish_once(
                dlq_message['original_channel'],
                dlq_message['message_data'],
                message_id
            )
            
            if success:
                # Remove from dead letter queue
                await self._redis_client.delete(dlq_key)
                self.logger.info(f"Successfully reprocessed dead letter message {message_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error reprocessing dead letter message {message_id}: {e}")
            return False
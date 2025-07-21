"""
Base agent class for all trading agents.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Dict, Any

from ..models.config_models import AgentConfig
from ..models.message_models import Message, AgentResponse
from ..models.trading_models import TradingProposal, Vote, Experience
from ..services.message_bus import MessageBus
from ..services.model_manager import ModelManager
from ..services.metrics_collector import MetricsCollector


class BaseAgent(ABC):
    """
    Base class for all trading agents in the multi-agent system.
    
    Provides common functionality for message processing, health checks,
    metrics collection, and model management.
    """
    
    def __init__(self, agent_id: str, config: AgentConfig):
        """
        Initialize the base agent.
        
        Args:
            agent_id: Unique identifier for this agent
            config: Agent configuration parameters
        """
        self.agent_id = agent_id
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{agent_id}")
        
        # Initialize core services
        self.message_bus = MessageBus(config.message_queue_config)
        self.model_manager = ModelManager(agent_id, config.model_version)
        self.metrics_collector = MetricsCollector(agent_id)
        
        # Agent state
        self._is_running = False
        self._health_status = "HEALTHY"
        self._last_heartbeat = datetime.utcnow()
        
        self.logger.info(f"Initialized agent {agent_id} of type {config.agent_type}")
    
    async def start(self) -> None:
        """Start the agent and begin processing messages."""
        if self._is_running:
            self.logger.warning("Agent is already running")
            return
        
        self.logger.info(f"Starting agent {self.agent_id}")
        self._is_running = True
        
        # Start message processing
        await self.message_bus.connect()
        await self.message_bus.subscribe(self.agent_id, self._handle_message)
        
        # Start health check loop
        asyncio.create_task(self._health_check_loop())
        
        self.logger.info(f"Agent {self.agent_id} started successfully")
    
    async def stop(self) -> None:
        """Stop the agent and cleanup resources."""
        if not self._is_running:
            self.logger.warning("Agent is not running")
            return
        
        self.logger.info(f"Stopping agent {self.agent_id}")
        self._is_running = False
        
        # Cleanup resources
        await self.message_bus.disconnect()
        await self.model_manager.cleanup()
        await self.metrics_collector.flush()
        
        self.logger.info(f"Agent {self.agent_id} stopped successfully")
    
    async def _handle_message(self, message: Message) -> AgentResponse:
        """
        Handle incoming messages and route to appropriate processors.
        
        Args:
            message: Incoming message to process
            
        Returns:
            Response from message processing
        """
        start_time = datetime.utcnow()
        
        try:
            self.logger.debug(f"Processing message {message.message_id} of type {message.message_type}")
            
            # Update heartbeat
            self._last_heartbeat = datetime.utcnow()
            
            # Process the message
            response = await self.process_message(message)
            
            # Record metrics
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            await self.metrics_collector.record_message_processed(
                message.message_type.value, 
                processing_time, 
                success=response.success
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing message {message.message_id}: {str(e)}")
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return AgentResponse(
                response_id="",
                original_message_id=message.message_id,
                agent_id=self.agent_id,
                success=False,
                error_message=str(e),
                processing_time_ms=processing_time
            )
    
    @abstractmethod
    async def process_message(self, message: Message) -> AgentResponse:
        """
        Process incoming messages. Must be implemented by subclasses.
        
        Args:
            message: Message to process
            
        Returns:
            Agent response
        """
        pass
    
    @abstractmethod
    async def vote_on_decision(self, proposal: TradingProposal) -> Vote:
        """
        Cast vote on trading proposal. Must be implemented by subclasses.
        
        Args:
            proposal: Trading proposal to vote on
            
        Returns:
            Vote with confidence score and rationale
        """
        pass
    
    def update_experience(self, experience: Experience) -> None:
        """
        Add experience to shared replay buffer for learning.
        
        Args:
            experience: Experience data to store
        """
        try:
            # This will be implemented when we create the experience replay system
            self.logger.debug(f"Recording experience {experience.experience_id}")
            # TODO: Implement experience storage
            
        except Exception as e:
            self.logger.error(f"Error updating experience: {str(e)}")
    
    async def _health_check_loop(self) -> None:
        """Periodic health check loop."""
        while self._is_running:
            try:
                await self._perform_health_check()
                await asyncio.sleep(self.config.health_check_interval)
            except Exception as e:
                self.logger.error(f"Health check failed: {str(e)}")
                self._health_status = "UNHEALTHY"
    
    async def _perform_health_check(self) -> None:
        """Perform health check and update status."""
        try:
            # Check if agent is responsive
            time_since_heartbeat = (datetime.utcnow() - self._last_heartbeat).total_seconds()
            
            if time_since_heartbeat > self.config.timeout_seconds * 2:
                self._health_status = "UNHEALTHY"
                self.logger.warning(f"Agent {self.agent_id} appears unresponsive")
            else:
                self._health_status = "HEALTHY"
            
            # Record health metrics
            await self.metrics_collector.record_health_check(self._health_status)
            
        except Exception as e:
            self.logger.error(f"Health check error: {str(e)}")
            self._health_status = "UNHEALTHY"
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get current health status of the agent.
        
        Returns:
            Health status information
        """
        return {
            "agent_id": self.agent_id,
            "status": self._health_status,
            "is_running": self._is_running,
            "last_heartbeat": self._last_heartbeat.isoformat(),
            "uptime_seconds": (datetime.utcnow() - self._last_heartbeat).total_seconds()
        }
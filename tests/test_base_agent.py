"""
Unit tests for BaseAgent class functionality.
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

from multi_agent_trading.agents.base_agent import BaseAgent
from multi_agent_trading.models.config_models import AgentConfig, AgentType
from multi_agent_trading.models.message_models import Message, MessageType, AgentResponse
from multi_agent_trading.models.trading_models import TradingProposal, Vote, Experience, TradeAction, RiskMetrics


class MockBaseAgent(BaseAgent):
    """Test implementation of BaseAgent for testing purposes."""
    
    async def process_message(self, message: Message) -> AgentResponse:
        """Test implementation of message processing."""
        return AgentResponse(
            response_id="test_response",
            original_message_id=message.message_id,
            agent_id=self.agent_id,
            success=True,
            result={"processed": True}
        )
    
    async def vote_on_decision(self, proposal: TradingProposal) -> Vote:
        """Test implementation of voting."""
        return Vote(
            agent_id=self.agent_id,
            proposal_id=proposal.proposal_id,
            score=75,
            confidence=0.8,
            rationale="Test vote",
            timestamp=datetime.utcnow()
        )


@pytest.fixture
def agent_config():
    """Create test agent configuration."""
    from multi_agent_trading.models.config_models import ResourceLimits, MessageQueueConfig
    
    return AgentConfig(
        agent_id="test_agent",
        agent_type=AgentType.MARKET_ANALYST,
        model_version="v1.0",
        parameters={"param1": "value1"},
        resource_limits=ResourceLimits(max_memory_mb=1024),
        message_queue_config=MessageQueueConfig(
            redis_url="redis://localhost:6379",
            max_retries=3
        ),
        health_check_interval=5,
        timeout_seconds=10
    )


@pytest.fixture
def test_agent(agent_config):
    """Create test agent instance."""
    return MockBaseAgent("test_agent", agent_config)


@pytest.fixture
def sample_message():
    """Create sample message for testing."""
    return Message(
        message_id="test_msg_123",
        message_type=MessageType.MARKET_DATA,
        sender_id="market_feed",
        recipient_id="test_agent",
        payload={"symbol": "AAPL", "price": 150.0},
        timestamp=datetime.utcnow()
    )


@pytest.fixture
def sample_trading_proposal():
    """Create sample trading proposal for testing."""
    risk_metrics = RiskMetrics(
        var_95=-0.05,
        cvar_95=-0.08,
        sharpe_ratio=1.2,
        max_drawdown=0.15,
        volatility=0.25
    )
    
    return TradingProposal(
        proposal_id="prop_123",
        symbol="AAPL",
        action=TradeAction.BUY,
        quantity=100,
        price_target=155.0,
        rationale="Strong technical indicators",
        confidence=0.85,
        risk_metrics=risk_metrics,
        timestamp=datetime.utcnow()
    )


class TestBaseAgentInitialization:
    """Test BaseAgent initialization."""
    
    def test_agent_initialization(self, agent_config):
        """Test that agent initializes correctly."""
        agent = MockBaseAgent("test_agent", agent_config)
        
        assert agent.agent_id == "test_agent"
        assert agent.config == agent_config
        assert agent._is_running is False
        assert agent._health_status == "HEALTHY"
        assert agent.message_bus is not None
        assert agent.model_manager is not None
        assert agent.metrics_collector is not None
    
    def test_agent_logger_setup(self, agent_config):
        """Test that logger is set up correctly."""
        agent = MockBaseAgent("test_agent", agent_config)
        
        assert agent.logger.name == "MockBaseAgent.test_agent"


class TestAgentLifecycle:
    """Test agent lifecycle management."""
    
    @pytest.mark.asyncio
    async def test_agent_start(self, test_agent):
        """Test agent start functionality."""
        # Mock the message bus and model manager methods
        test_agent.message_bus.connect = AsyncMock()
        test_agent.message_bus.subscribe = AsyncMock()
        
        await test_agent.start()
        
        assert test_agent._is_running is True
        test_agent.message_bus.connect.assert_called_once()
        test_agent.message_bus.subscribe.assert_called_once_with(
            test_agent.agent_id, 
            test_agent._handle_message
        )
    
    @pytest.mark.asyncio
    async def test_agent_start_already_running(self, test_agent):
        """Test starting agent when already running."""
        test_agent._is_running = True
        test_agent.message_bus.connect = AsyncMock()
        
        await test_agent.start()
        
        # Should not call connect again
        test_agent.message_bus.connect.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_agent_stop(self, test_agent):
        """Test agent stop functionality."""
        test_agent._is_running = True
        test_agent.message_bus.disconnect = AsyncMock()
        test_agent.model_manager.cleanup = AsyncMock()
        test_agent.metrics_collector.flush = AsyncMock()
        
        await test_agent.stop()
        
        assert test_agent._is_running is False
        test_agent.message_bus.disconnect.assert_called_once()
        test_agent.model_manager.cleanup.assert_called_once()
        test_agent.metrics_collector.flush.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_agent_stop_not_running(self, test_agent):
        """Test stopping agent when not running."""
        test_agent._is_running = False
        test_agent.message_bus.disconnect = AsyncMock()
        
        await test_agent.stop()
        
        # Should not call disconnect
        test_agent.message_bus.disconnect.assert_not_called()


class TestMessageHandling:
    """Test message handling functionality."""
    
    @pytest.mark.asyncio
    async def test_handle_message_success(self, test_agent, sample_message):
        """Test successful message handling."""
        test_agent.metrics_collector.record_message_processed = AsyncMock()
        
        response = await test_agent._handle_message(sample_message)
        
        assert response.success is True
        assert response.original_message_id == sample_message.message_id
        assert response.agent_id == test_agent.agent_id
        assert response.result == {"processed": True}
        assert response.processing_time_ms is not None
        
        # Verify metrics were recorded
        test_agent.metrics_collector.record_message_processed.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_handle_message_error(self, test_agent, sample_message):
        """Test message handling with error."""
        # Mock process_message to raise an exception
        with patch.object(test_agent, 'process_message', side_effect=Exception("Test error")):
            response = await test_agent._handle_message(sample_message)
        
        assert response.success is False
        assert response.error_message == "Test error"
        assert response.original_message_id == sample_message.message_id
        assert response.processing_time_ms is not None
    
    @pytest.mark.asyncio
    async def test_handle_message_updates_heartbeat(self, test_agent, sample_message):
        """Test that message handling updates heartbeat."""
        old_heartbeat = test_agent._last_heartbeat
        
        # Wait a small amount to ensure timestamp difference
        await asyncio.sleep(0.01)
        
        await test_agent._handle_message(sample_message)
        
        assert test_agent._last_heartbeat > old_heartbeat


class TestHealthChecks:
    """Test health check functionality."""
    
    def test_get_health_status(self, test_agent):
        """Test health status retrieval."""
        test_agent._is_running = True
        test_agent._health_status = "HEALTHY"
        
        status = test_agent.get_health_status()
        
        assert status["agent_id"] == test_agent.agent_id
        assert status["status"] == "HEALTHY"
        assert status["is_running"] is True
        assert "last_heartbeat" in status
        assert "uptime_seconds" in status
    
    @pytest.mark.asyncio
    async def test_perform_health_check_healthy(self, test_agent):
        """Test health check when agent is healthy."""
        test_agent.metrics_collector.record_health_check = AsyncMock()
        test_agent._last_heartbeat = datetime.utcnow()
        
        await test_agent._perform_health_check()
        
        assert test_agent._health_status == "HEALTHY"
        test_agent.metrics_collector.record_health_check.assert_called_once_with("HEALTHY")
    
    @pytest.mark.asyncio
    async def test_perform_health_check_unhealthy(self, test_agent):
        """Test health check when agent is unresponsive."""
        test_agent.metrics_collector.record_health_check = AsyncMock()
        # Set heartbeat to old timestamp
        test_agent._last_heartbeat = datetime.utcnow() - timedelta(seconds=30)
        
        await test_agent._perform_health_check()
        
        assert test_agent._health_status == "UNHEALTHY"
        test_agent.metrics_collector.record_health_check.assert_called_once_with("UNHEALTHY")
    
    @pytest.mark.asyncio
    async def test_perform_health_check_error(self, test_agent):
        """Test health check with error."""
        test_agent.metrics_collector.record_health_check = AsyncMock(
            side_effect=Exception("Metrics error")
        )
        
        await test_agent._perform_health_check()
        
        assert test_agent._health_status == "UNHEALTHY"
    
    @pytest.mark.asyncio
    async def test_health_check_loop(self, test_agent):
        """Test health check loop functionality."""
        test_agent.config.health_check_interval = 0.1  # Very short interval for testing
        test_agent._perform_health_check = AsyncMock()
        test_agent._is_running = True  # Set running state
        
        # Start the health check loop
        task = asyncio.create_task(test_agent._health_check_loop())
        
        # Let it run for a short time
        await asyncio.sleep(0.25)
        
        # Stop the agent to exit the loop
        test_agent._is_running = False
        
        # Wait for the task to complete
        await asyncio.sleep(0.15)
        
        try:
            await task
        except asyncio.CancelledError:
            pass
        
        # Verify health check was called multiple times
        assert test_agent._perform_health_check.call_count >= 2


class TestVotingAndExperience:
    """Test voting and experience functionality."""
    
    @pytest.mark.asyncio
    async def test_vote_on_decision(self, test_agent, sample_trading_proposal):
        """Test voting on trading proposal."""
        vote = await test_agent.vote_on_decision(sample_trading_proposal)
        
        assert vote.agent_id == test_agent.agent_id
        assert vote.proposal_id == sample_trading_proposal.proposal_id
        assert vote.score == 75
        assert vote.confidence == 0.8
        assert vote.rationale == "Test vote"
        assert isinstance(vote.timestamp, datetime)
    
    def test_update_experience(self, test_agent):
        """Test experience update functionality."""
        experience = Experience(
            experience_id="exp_123",
            state={"market_state": "bullish"},
            action={"action": "buy", "quantity": 100},
            reward=0.05,
            next_state={"market_state": "neutral"},
            agent_contributions={"test_agent": 0.8},
            timestamp=datetime.utcnow()
        )
        
        # Should not raise an exception
        test_agent.update_experience(experience)


class TestErrorHandling:
    """Test error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_message_processing_timeout(self, test_agent, sample_message):
        """Test message processing timeout handling."""
        # Mock process_message to take too long
        async def slow_process(message):
            await asyncio.sleep(2)  # Longer than typical timeout
            return AgentResponse(
                response_id="",
                original_message_id=message.message_id,
                agent_id=test_agent.agent_id,
                success=True
            )
        
        with patch.object(test_agent, 'process_message', side_effect=slow_process):
            # This should complete quickly due to timeout handling in message bus
            response = await test_agent._handle_message(sample_message)
            
            # Response should still be generated
            assert response is not None
            assert response.original_message_id == sample_message.message_id
    
    @pytest.mark.asyncio
    async def test_health_check_exception_handling(self, test_agent):
        """Test health check exception handling."""
        # Mock metrics collector to raise exception
        test_agent.metrics_collector.record_health_check = AsyncMock(
            side_effect=Exception("Metrics failure")
        )
        
        # Should not raise exception
        await test_agent._perform_health_check()
        
        # Health status should be set to unhealthy
        assert test_agent._health_status == "UNHEALTHY"


class TestIntegration:
    """Integration tests for BaseAgent."""
    
    @pytest.mark.asyncio
    async def test_full_agent_lifecycle(self, test_agent, sample_message):
        """Test complete agent lifecycle with message processing."""
        # Mock external dependencies
        test_agent.message_bus.connect = AsyncMock()
        test_agent.message_bus.subscribe = AsyncMock()
        test_agent.message_bus.disconnect = AsyncMock()
        test_agent.model_manager.cleanup = AsyncMock()
        test_agent.metrics_collector.flush = AsyncMock()
        test_agent.metrics_collector.record_message_processed = AsyncMock()
        test_agent.metrics_collector.record_health_check = AsyncMock()
        
        # Start agent
        await test_agent.start()
        assert test_agent._is_running is True
        
        # Process message
        response = await test_agent._handle_message(sample_message)
        assert response.success is True
        
        # Check health
        health = test_agent.get_health_status()
        assert health["is_running"] is True
        
        # Stop agent
        await test_agent.stop()
        assert test_agent._is_running is False
        
        # Verify all cleanup methods were called
        test_agent.message_bus.disconnect.assert_called_once()
        test_agent.model_manager.cleanup.assert_called_once()
        test_agent.metrics_collector.flush.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
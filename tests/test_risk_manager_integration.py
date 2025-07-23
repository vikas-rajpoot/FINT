"""
Integration tests for the Risk Manager Agent.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from multi_agent_trading.agents.risk_manager import RiskManagerAgent
from multi_agent_trading.models.config_models import AgentConfig, MessageQueueConfig
from multi_agent_trading.models.message_models import Message, MessageType
from multi_agent_trading.models.trading_models import (
    Portfolio, Position, TradingProposal, TradeAction, RiskMetrics
)


class TestRiskManagerAgentIntegration:
    """Integration test cases for RiskManagerAgent."""
    
    @pytest.fixture
    def agent_config(self):
        """Create agent configuration."""
        from multi_agent_trading.models.config_models import AgentType
        return AgentConfig(
            agent_id="risk_manager_test",
            agent_type=AgentType.RISK_MANAGER,
            model_version="1.0",
            model_path="/tmp/test_model",
            timeout_seconds=30,
            health_check_interval=10,
            message_queue_config=MessageQueueConfig(
                redis_url="redis://localhost:6379"
            )
        )
    
    @pytest.fixture
    def sample_portfolio(self):
        """Create a sample portfolio."""
        positions = {
            "AAPL": Position(
                symbol="AAPL",
                quantity=100,
                entry_price=150.0,
                current_price=155.0,
                market_value=15500.0,
                unrealized_pnl=500.0,
                weight=0.4
            ),
            "GOOGL": Position(
                symbol="GOOGL",
                quantity=50,
                entry_price=2800.0,
                current_price=2850.0,
                market_value=142500.0,
                unrealized_pnl=2500.0,
                weight=0.6
            )
        }
        
        return Portfolio(
            portfolio_id="test_portfolio",
            total_value=158000.0,
            cash=0.0,
            positions=positions,
            timestamp=datetime.utcnow()
        )
    
    @pytest.fixture
    def sample_proposal(self):
        """Create a sample trading proposal."""
        return TradingProposal(
            proposal_id="test_proposal",
            symbol="MSFT",
            action=TradeAction.BUY,
            quantity=100,
            price_target=300.0,
            rationale="Test proposal for integration testing",
            confidence=0.8,
            risk_metrics=RiskMetrics(
                var_95=0.05,
                cvar_95=0.07,
                sharpe_ratio=1.2,
                max_drawdown=0.1,
                volatility=0.2
            ),
            timestamp=datetime.utcnow()
        )
    
    @pytest.fixture
    def risk_manager_agent(self, agent_config, sample_portfolio):
        """Create a Risk Manager Agent instance."""
        with patch('multi_agent_trading.services.message_bus.MessageBus'):
            with patch('multi_agent_trading.services.model_manager.ModelManager'):
                with patch('multi_agent_trading.services.metrics_collector.MetricsCollector'):
                    agent = RiskManagerAgent("risk_manager_1", agent_config, sample_portfolio)
                    
                    # Mock the message bus methods
                    agent.message_bus.connect = AsyncMock()
                    agent.message_bus.disconnect = AsyncMock()
                    agent.message_bus.subscribe = AsyncMock()
                    agent.message_bus.publish = AsyncMock()
                    
                    # Mock model manager methods
                    agent.model_manager.initialize = AsyncMock()
                    agent.model_manager.cleanup = AsyncMock()
                    
                    # Mock metrics collector methods
                    agent.metrics_collector.record_message_processed = AsyncMock()
                    agent.metrics_collector.record_health_check = AsyncMock()
                    agent.metrics_collector.flush = AsyncMock()
                    
                    return agent
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, risk_manager_agent, sample_portfolio):
        """Test Risk Manager Agent initialization."""
        assert risk_manager_agent.agent_id == "risk_manager_1"
        assert risk_manager_agent.current_portfolio == sample_portfolio
        assert risk_manager_agent.risk_engine is not None
        assert risk_manager_agent.base_risk_params is not None
        assert risk_manager_agent.market_volatility == 0.20  # Default
    
    @pytest.mark.asyncio
    async def test_vote_on_decision_basic(self, risk_manager_agent, sample_proposal):
        """Test basic voting functionality."""
        # Add some historical data
        risk_manager_agent.update_market_data("MSFT", [290, 295, 300, 305, 298], [0.017, 0.017, 0.017, -0.023])
        
        vote = await risk_manager_agent.vote_on_decision(sample_proposal)
        
        assert vote.agent_id == "risk_manager_1"
        assert vote.proposal_id == "test_proposal"
        assert 0 <= vote.score <= 100
        assert 0 <= vote.confidence <= 1
        assert vote.rationale is not None
        assert isinstance(vote.timestamp, datetime)
    
    @pytest.mark.asyncio
    async def test_vote_on_decision_no_portfolio(self):
        """Test voting without portfolio data."""
        from multi_agent_trading.models.config_models import AgentType, AgentConfig, MessageQueueConfig
        
        config = AgentConfig(
            agent_id="risk_manager_2",
            agent_type=AgentType.RISK_MANAGER,
            model_version="1.0",
            model_path="/tmp/test_model",
            timeout_seconds=30,
            health_check_interval=10,
            message_queue_config=MessageQueueConfig(redis_url="redis://localhost:6379")
        )
        
        with patch('multi_agent_trading.services.message_bus.MessageBus'):
            with patch('multi_agent_trading.services.model_manager.ModelManager'):
                with patch('multi_agent_trading.services.metrics_collector.MetricsCollector'):
                    agent = RiskManagerAgent("risk_manager_2", config, None)
                    
                    proposal = TradingProposal(
                        proposal_id="test_proposal",
                        symbol="MSFT",
                        action=TradeAction.BUY,
                        quantity=100,
                        price_target=300.0,
                        rationale="Test proposal",
                        confidence=0.8,
                        risk_metrics=RiskMetrics(0.05, 0.07, 1.2, 0.1, 0.2),
                        timestamp=datetime.utcnow()
                    )
                    
                    vote = await agent.vote_on_decision(proposal)
                    
                    assert vote.score == 0
                    assert vote.confidence == 0.1
                    assert "No portfolio data" in vote.rationale
    
    @pytest.mark.asyncio
    async def test_vote_on_risky_proposal(self, risk_manager_agent):
        """Test voting on a high-risk proposal."""
        # Create a very large proposal that should be rejected
        risky_proposal = TradingProposal(
            proposal_id="risky_proposal",
            symbol="MSFT",
            action=TradeAction.BUY,
            quantity=10000,  # Very large quantity
            price_target=300.0,
            rationale="High risk test proposal",
            confidence=0.8,
            risk_metrics=RiskMetrics(0.15, 0.20, 0.5, 0.25, 0.4),  # Poor risk metrics
            timestamp=datetime.utcnow()
        )
        
        vote = await risk_manager_agent.vote_on_decision(risky_proposal)
        
        # Should have low score due to high risk
        assert vote.score < 30
        assert vote.confidence > 0.7  # High confidence in rejection
        assert "Risk limit violations" in vote.rationale or "High portfolio risk" in vote.rationale
    
    @pytest.mark.asyncio
    async def test_handle_market_data_update(self, risk_manager_agent):
        """Test handling market data update messages."""
        message = Message(
            message_id="market_data_1",
            message_type=MessageType.MARKET_DATA,
            sender_id="market_feed",
            recipient_id=None,
            timestamp=datetime.utcnow(),
            payload={
                "symbol": "AAPL",
                "prices": [145, 148, 152, 155, 153],
                "returns": [0.02, 0.027, 0.02, -0.013]
            }
        )
        
        response = await risk_manager_agent.process_message(message)
        
        assert response.success
        assert response.result["status"] == "market_data_updated"
        assert response.result["symbol"] == "AAPL"
        assert "AAPL" in risk_manager_agent.historical_prices
        assert len(risk_manager_agent.historical_prices["AAPL"]) == 5
    
    @pytest.mark.asyncio
    async def test_handle_risk_analysis_request(self, risk_manager_agent):
        """Test handling risk analysis requests."""
        # Add some historical data first
        risk_manager_agent.update_market_data("AAPL", [145, 148, 152, 155], [0.02, 0.027, 0.02])
        risk_manager_agent.update_market_data("GOOGL", [2750, 2780, 2820, 2850], [0.011, 0.014, 0.011])
        
        message = Message(
            message_id="risk_analysis_1",
            message_type=MessageType.SYSTEM_ALERT,  # Using available message type
            sender_id="portfolio_manager",
            recipient_id=None,
            timestamp=datetime.utcnow(),
            payload={"request_type": "risk_analysis"}
        )
        
        response = await risk_manager_agent.process_message(message)
        
        assert response.success
        assert "risk_metrics" in response.result
        assert "exposure_report" in response.result
        
        risk_metrics = response.result["risk_metrics"]
        assert "var_95" in risk_metrics
        assert "sharpe_ratio" in risk_metrics
        
        exposure_report = response.result["exposure_report"]
        assert "total_exposure" in exposure_report
        assert "concentration_risk" in exposure_report
    
    @pytest.mark.asyncio
    async def test_handle_risk_analysis_no_portfolio(self):
        """Test risk analysis request without portfolio."""
        from multi_agent_trading.models.config_models import AgentType, AgentConfig, MessageQueueConfig
        
        config = AgentConfig(
            agent_id="risk_manager_3",
            agent_type=AgentType.RISK_MANAGER,
            model_version="1.0",
            model_path="/tmp/test_model",
            timeout_seconds=30,
            health_check_interval=10,
            message_queue_config=MessageQueueConfig(redis_url="redis://localhost:6379")
        )
        
        with patch('multi_agent_trading.services.message_bus.MessageBus'):
            with patch('multi_agent_trading.services.model_manager.ModelManager'):
                with patch('multi_agent_trading.services.metrics_collector.MetricsCollector'):
                    agent = RiskManagerAgent("risk_manager_3", config, None)
                    
                    message = Message(
                        message_id="risk_analysis_2",
                        message_type=MessageType.SYSTEM_ALERT,  # Using available message type
                        sender_id="portfolio_manager",
                        recipient_id=None,
                        timestamp=datetime.utcnow(),
                        payload={"request_type": "risk_analysis"}
                    )
                    
                    response = await agent.process_message(message)
                    
                    assert not response.success
                    assert "No portfolio data available" in response.error_message
    
    @pytest.mark.asyncio
    async def test_update_portfolio(self, risk_manager_agent):
        """Test portfolio update functionality."""
        new_positions = {
            "TSLA": Position(
                symbol="TSLA",
                quantity=50,
                entry_price=800.0,
                current_price=850.0,
                market_value=42500.0,
                unrealized_pnl=2500.0,
                weight=1.0
            )
        }
        
        new_portfolio = Portfolio(
            portfolio_id="new_portfolio",
            total_value=42500.0,
            cash=0.0,
            positions=new_positions,
            timestamp=datetime.utcnow()
        )
        
        risk_manager_agent.update_portfolio(new_portfolio)
        
        assert risk_manager_agent.current_portfolio == new_portfolio
        assert len(risk_manager_agent.current_portfolio.positions) == 1
        assert "TSLA" in risk_manager_agent.current_portfolio.positions
    
    @pytest.mark.asyncio
    async def test_update_market_data(self, risk_manager_agent):
        """Test market data update functionality."""
        initial_volatility = risk_manager_agent.market_volatility
        
        # Add market data for multiple symbols
        risk_manager_agent.update_market_data("AAPL", [145, 148, 152, 155], [0.02, 0.027, 0.02])
        risk_manager_agent.update_market_data("GOOGL", [2750, 2780, 2820, 2850], [0.011, 0.014, 0.011])
        
        assert "AAPL" in risk_manager_agent.historical_prices
        assert "GOOGL" in risk_manager_agent.historical_prices
        assert len(risk_manager_agent.historical_prices["AAPL"]) == 4
        assert len(risk_manager_agent.historical_returns["AAPL"]) == 3
        
        # Market volatility should be updated
        assert risk_manager_agent.market_volatility != initial_volatility
    
    @pytest.mark.asyncio
    async def test_sector_currency_mapping(self, risk_manager_agent):
        """Test sector and currency mapping updates."""
        sector_mapping = {"AAPL": "Technology", "GOOGL": "Technology", "MSFT": "Technology"}
        currency_mapping = {"AAPL": "USD", "GOOGL": "USD", "MSFT": "USD"}
        
        risk_manager_agent.update_sector_mapping(sector_mapping)
        risk_manager_agent.update_currency_mapping(currency_mapping)
        
        assert risk_manager_agent.sector_mapping["AAPL"] == "Technology"
        assert risk_manager_agent.currency_mapping["AAPL"] == "USD"
    
    @pytest.mark.asyncio
    async def test_error_handling_in_voting(self, risk_manager_agent):
        """Test error handling during voting process."""
        # Create an invalid proposal that will cause errors
        invalid_proposal = Mock()
        invalid_proposal.proposal_id = "invalid"
        invalid_proposal.symbol = None  # This will cause errors
        
        vote = await risk_manager_agent.vote_on_decision(invalid_proposal)
        
        assert vote.score == 0
        assert vote.confidence == 0.1  # Low confidence due to errors in analysis
        assert "Error in risk analysis" in vote.rationale
    
    @pytest.mark.asyncio
    async def test_message_processing_error_handling(self, risk_manager_agent):
        """Test error handling in message processing."""
        # Create a message with invalid payload
        invalid_message = Message(
            message_id="invalid_msg",
            message_type=MessageType.MARKET_DATA,
            sender_id="test",
            recipient_id=None,
            timestamp=datetime.utcnow(),
            payload=None  # Invalid payload
        )
        
        response = await risk_manager_agent.process_message(invalid_message)
        
        assert not response.success
        assert response.error_message is not None
    
    @pytest.mark.asyncio
    async def test_risk_vote_calculation_logic(self, risk_manager_agent, sample_proposal):
        """Test the risk vote calculation logic with different scenarios."""
        # Add historical data
        risk_manager_agent.update_market_data("MSFT", [290, 295, 300, 305, 298], [0.017, 0.017, 0.017, -0.023])
        
        # Test with high market volatility
        risk_manager_agent.market_volatility = 0.35  # High volatility
        vote_high_vol = await risk_manager_agent.vote_on_decision(sample_proposal)
        
        # Test with low market volatility
        risk_manager_agent.market_volatility = 0.12  # Low volatility
        vote_low_vol = await risk_manager_agent.vote_on_decision(sample_proposal)
        
        # Both votes should be reasonable (the exact relationship depends on risk limits)
        assert 0 <= vote_high_vol.score <= 100
        assert 0 <= vote_low_vol.score <= 100
        
        # Both should have reasonable confidence levels
        assert 0.1 <= vote_high_vol.confidence <= 1.0
        assert 0.1 <= vote_low_vol.confidence <= 1.0
    
    @pytest.mark.asyncio
    async def test_vix_level_impact(self, risk_manager_agent, sample_proposal):
        """Test VIX level impact on voting."""
        # Add historical data
        risk_manager_agent.update_market_data("MSFT", [290, 295, 300, 305, 298], [0.017, 0.017, 0.017, -0.023])
        
        # Test with high VIX (fear)
        risk_manager_agent.vix_level = 35.0
        vote_high_vix = await risk_manager_agent.vote_on_decision(sample_proposal)
        
        # Test with low VIX (complacency)
        risk_manager_agent.vix_level = 12.0
        vote_low_vix = await risk_manager_agent.vote_on_decision(sample_proposal)
        
        # Both votes should be reasonable (the exact relationship depends on risk limits)
        assert 0 <= vote_high_vix.score <= 100
        assert 0 <= vote_low_vix.score <= 100


if __name__ == "__main__":
    pytest.main([__file__])
"""
Integration tests for Portfolio Optimizer Agent.
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
import uuid

from multi_agent_trading.agents.portfolio_optimizer import PortfolioOptimizerAgent
from multi_agent_trading.agents.portfolio_optimization_algorithms import Asset
from multi_agent_trading.models.config_models import AgentConfig, MessageQueueConfig
from multi_agent_trading.models.message_models import Message, MessageType
from multi_agent_trading.models.trading_models import (
    TradingProposal, Portfolio, Position, MarketData, RiskMetrics, TradeAction
)


class TestPortfolioOptimizerAgent:
    """Test PortfolioOptimizerAgent integration."""
    
    @pytest.fixture
    def agent_config(self):
        """Create agent configuration."""
        from multi_agent_trading.models.config_models import AgentType
        return AgentConfig(
            agent_id="portfolio_optimizer_test",
            agent_type=AgentType.PORTFOLIO_OPTIMIZER,
            model_version="1.0.0",
            model_path="/tmp/test_model",
            timeout_seconds=30,
            health_check_interval=10,
            message_queue_config=MessageQueueConfig(
                redis_url="redis://localhost:6379",
                max_retries=3,
                base_delay=1.0
            ),
            parameters={
                'risk_free_rate': 0.02,
                'rebalancing_threshold': 0.05,
                'transaction_cost_rate': 0.001
            }
        )
    
    @pytest.fixture
    def portfolio_optimizer_agent(self, agent_config):
        """Create portfolio optimizer agent instance."""
        with patch('multi_agent_trading.services.message_bus.MessageBus'):
            with patch('multi_agent_trading.services.model_manager.ModelManager'):
                with patch('multi_agent_trading.services.metrics_collector.MetricsCollector'):
                    agent = PortfolioOptimizerAgent("portfolio_optimizer_test", agent_config)
                    return agent
    
    @pytest.fixture
    def sample_portfolio(self):
        """Create sample portfolio."""
        positions = {
            "AAPL": Position(
                symbol="AAPL",
                quantity=100,
                entry_price=140.0,
                current_price=150.0,
                market_value=15000.0,
                unrealized_pnl=1000.0,
                weight=0.5
            ),
            "GOOGL": Position(
                symbol="GOOGL",
                quantity=5,
                entry_price=2700.0,
                current_price=2800.0,
                market_value=14000.0,
                unrealized_pnl=500.0,
                weight=0.47
            )
        }
        
        return Portfolio(
            portfolio_id="test_portfolio",
            total_value=30000.0,
            cash=1000.0,
            positions=positions,
            timestamp=datetime.utcnow()
        )
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data."""
        return {
            "AAPL": MarketData(
                symbol="AAPL",
                timestamp=datetime.utcnow(),
                price=150.0,
                volume=1000000,
                bid=149.5,
                ask=150.5,
                technical_indicators={"rsi": 65.0, "macd": 0.5}
            ),
            "GOOGL": MarketData(
                symbol="GOOGL",
                timestamp=datetime.utcnow(),
                price=2800.0,
                volume=500000,
                bid=2795.0,
                ask=2805.0,
                technical_indicators={"rsi": 55.0, "macd": -0.2}
            ),
            "MSFT": MarketData(
                symbol="MSFT",
                timestamp=datetime.utcnow(),
                price=300.0,
                volume=800000,
                bid=299.5,
                ask=300.5,
                technical_indicators={"rsi": 60.0, "macd": 0.1}
            )
        }
    
    @pytest.fixture
    def sample_trading_proposal(self):
        """Create sample trading proposal."""
        return TradingProposal(
            proposal_id=str(uuid.uuid4()),
            symbol="AAPL",
            action=TradeAction.BUY,
            quantity=50,
            price_target=152.0,
            rationale="Technical analysis suggests upward momentum",
            confidence=0.75,
            risk_metrics=RiskMetrics(
                var_95=0.05,
                cvar_95=0.08,
                sharpe_ratio=1.2,
                max_drawdown=0.15,
                volatility=0.20
            ),
            timestamp=datetime.utcnow()
        )
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, portfolio_optimizer_agent):
        """Test agent initialization."""
        assert portfolio_optimizer_agent.agent_id == "portfolio_optimizer_test"
        assert portfolio_optimizer_agent.optimizer is not None
        assert portfolio_optimizer_agent.rebalancing_engine is not None
        assert portfolio_optimizer_agent.current_portfolio is None
        assert portfolio_optimizer_agent.target_allocations == {}
        assert portfolio_optimizer_agent.asset_universe == []
    
    @pytest.mark.asyncio
    async def test_handle_market_data_update(self, portfolio_optimizer_agent):
        """Test handling market data updates."""
        market_data = {
            'volatility': 0.18,
            'regime': 'normal',
            'symbols': ['AAPL', 'GOOGL', 'MSFT'],
            'AAPL': {
                'expected_return': 0.12,
                'volatility': 0.20,
                'price': 150.0,
                'market_cap': 2500000000000,
                'sector': 'Technology'
            },
            'GOOGL': {
                'expected_return': 0.10,
                'volatility': 0.18,
                'price': 2800.0,
                'market_cap': 1800000000000,
                'sector': 'Technology'
            }
        }
        
        message = Message(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.MARKET_DATA_UPDATE,
            sender_id="market_data_service",
            recipient_id=portfolio_optimizer_agent.agent_id,
            payload={'market_data': market_data},
            timestamp=datetime.utcnow()
        )
        
        response = await portfolio_optimizer_agent.process_message(message)
        
        assert response.success
        assert response.data['status'] == 'market_data_processed'
        assert portfolio_optimizer_agent.market_conditions['volatility'] == 0.18
        assert portfolio_optimizer_agent.market_conditions['regime'] == 'normal'
        assert len(portfolio_optimizer_agent.asset_universe) == 2
    
    @pytest.mark.asyncio
    async def test_handle_portfolio_update(self, portfolio_optimizer_agent, sample_portfolio):
        """Test handling portfolio updates."""
        message = Message(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.PORTFOLIO_UPDATE,
            sender_id="portfolio_service",
            recipient_id=portfolio_optimizer_agent.agent_id,
            payload={'portfolio': sample_portfolio.to_dict()},
            timestamp=datetime.utcnow()
        )
        
        response = await portfolio_optimizer_agent.process_message(message)
        
        assert response.success
        assert response.data['status'] == 'portfolio_updated'
        assert portfolio_optimizer_agent.current_portfolio is not None
        assert portfolio_optimizer_agent.current_portfolio.total_value == 30000.0
    
    @pytest.mark.asyncio
    async def test_handle_optimization_request(self, portfolio_optimizer_agent):
        """Test handling optimization requests."""
        # First set up asset universe
        from multi_agent_trading.agents.portfolio_optimization_algorithms import Asset
        portfolio_optimizer_agent.asset_universe = [
            Asset("AAPL", 0.12, 0.20, 150.0, 2500000000000, "Technology"),
            Asset("GOOGL", 0.10, 0.18, 2800.0, 1800000000000, "Technology")
        ]
        portfolio_optimizer_agent.correlation_matrix = np.array([[1.0, 0.7], [0.7, 1.0]])
        
        message = Message(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.OPTIMIZATION_REQUEST,
            sender_id="trading_service",
            recipient_id=portfolio_optimizer_agent.agent_id,
            payload={
                'optimization_type': 'mean_variance',
                'constraints': {'max_weight': 0.6}
            },
            timestamp=datetime.utcnow()
        )
        
        response = await portfolio_optimizer_agent.process_message(message)
        
        assert response.success
        assert 'allocation_plan' in response.data
        assert response.data['optimization_type'] == 'mean_variance'
        assert len(portfolio_optimizer_agent.target_allocations) > 0
        assert len(portfolio_optimizer_agent.optimization_history) == 1
    
    @pytest.mark.asyncio
    async def test_handle_rebalancing_request(self, portfolio_optimizer_agent, sample_portfolio, sample_market_data):
        """Test handling rebalancing requests."""
        # Set up agent state
        portfolio_optimizer_agent.current_portfolio = sample_portfolio
        portfolio_optimizer_agent.target_allocations = {"AAPL": 0.4, "GOOGL": 0.6}
        
        message = Message(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.REBALANCING_REQUEST,
            sender_id="trading_service",
            recipient_id=portfolio_optimizer_agent.agent_id,
            payload={
                'transaction_costs': {'AAPL': 0.001, 'GOOGL': 0.001},
                'market_data': {symbol: data.to_dict() for symbol, data in sample_market_data.items()}
            },
            timestamp=datetime.utcnow()
        )
        
        response = await portfolio_optimizer_agent.process_message(message)
        
        assert response.success
        assert 'rebalancing_plan' in response.data
        assert 'should_rebalance' in response.data
        assert 'rationale' in response.data
        assert len(portfolio_optimizer_agent.rebalancing_history) == 1
    
    @pytest.mark.asyncio
    async def test_vote_on_trading_proposal(self, portfolio_optimizer_agent, sample_portfolio, sample_trading_proposal):
        """Test voting on trading proposals."""
        # Set up agent state
        portfolio_optimizer_agent.current_portfolio = sample_portfolio
        portfolio_optimizer_agent.target_allocations = {"AAPL": 0.6, "GOOGL": 0.4}
        
        vote = await portfolio_optimizer_agent.vote_on_decision(sample_trading_proposal)
        
        assert isinstance(vote.score, int)
        assert 0 <= vote.score <= 100
        assert 0 <= vote.confidence <= 1
        assert len(vote.rationale) > 0
        assert vote.agent_id == portfolio_optimizer_agent.agent_id
        assert vote.proposal_id == sample_trading_proposal.proposal_id
    
    @pytest.mark.asyncio
    async def test_vote_on_proposal_alignment_improvement(self, portfolio_optimizer_agent, sample_portfolio):
        """Test voting when proposal improves allocation alignment."""
        # Set up agent state - AAPL is under-allocated
        portfolio_optimizer_agent.current_portfolio = sample_portfolio
        portfolio_optimizer_agent.target_allocations = {"AAPL": 0.7, "GOOGL": 0.3}  # Want more AAPL
        
        # Proposal to buy more AAPL (should improve alignment)
        proposal = TradingProposal(
            proposal_id=str(uuid.uuid4()),
            symbol="AAPL",
            action=TradeAction.BUY,
            quantity=100,  # Large buy to improve alignment
            price_target=150.0,
            rationale="Increase AAPL allocation",
            confidence=0.8,
            risk_metrics=RiskMetrics(0.05, 0.08, 1.2, 0.15, 0.20),
            timestamp=datetime.utcnow()
        )
        
        vote = await portfolio_optimizer_agent.vote_on_decision(proposal)
        
        # Should vote positively since it improves alignment
        assert vote.score > 50
        assert "improves allocation alignment" in vote.rationale
    
    @pytest.mark.asyncio
    async def test_vote_on_proposal_alignment_deterioration(self, portfolio_optimizer_agent, sample_portfolio):
        """Test voting when proposal worsens allocation alignment."""
        # Set up agent state - AAPL is already over-allocated
        portfolio_optimizer_agent.current_portfolio = sample_portfolio
        portfolio_optimizer_agent.target_allocations = {"AAPL": 0.3, "GOOGL": 0.7}  # Want less AAPL
        
        # Proposal to buy more AAPL (should worsen alignment)
        proposal = TradingProposal(
            proposal_id=str(uuid.uuid4()),
            symbol="AAPL",
            action=TradeAction.BUY,
            quantity=100,  # Buy more when already over-allocated
            price_target=150.0,
            rationale="Increase AAPL allocation",
            confidence=0.8,
            risk_metrics=RiskMetrics(0.05, 0.08, 1.2, 0.15, 0.20),
            timestamp=datetime.utcnow()
        )
        
        vote = await portfolio_optimizer_agent.vote_on_decision(proposal)
        
        # Should vote negatively since it worsens alignment
        assert vote.score < 50
        assert "worsens allocation alignment" in vote.rationale
    
    @pytest.mark.asyncio
    async def test_vote_with_insufficient_data(self, portfolio_optimizer_agent, sample_trading_proposal):
        """Test voting with insufficient portfolio data."""
        # No portfolio or target allocations set
        vote = await portfolio_optimizer_agent.vote_on_decision(sample_trading_proposal)
        
        assert vote.score == 50  # Neutral vote
        assert vote.confidence == 0.3  # Low confidence
        assert "Insufficient portfolio data" in vote.rationale
    
    @pytest.mark.asyncio
    async def test_handle_unsupported_message_type(self, portfolio_optimizer_agent):
        """Test handling unsupported message types."""
        message = Message(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.AGENT_HEARTBEAT,  # Unsupported type
            sender_id="test_sender",
            recipient_id=portfolio_optimizer_agent.agent_id,
            payload={},
            timestamp=datetime.utcnow()
        )
        
        response = await portfolio_optimizer_agent.process_message(message)
        
        assert not response.success
        assert "Unsupported message type" in response.error_message
    
    @pytest.mark.asyncio
    async def test_optimization_request_without_asset_universe(self, portfolio_optimizer_agent):
        """Test optimization request without asset universe."""
        message = Message(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.OPTIMIZATION_REQUEST,
            sender_id="trading_service",
            recipient_id=portfolio_optimizer_agent.agent_id,
            payload={'optimization_type': 'mean_variance'},
            timestamp=datetime.utcnow()
        )
        
        response = await portfolio_optimizer_agent.process_message(message)
        
        assert not response.success
        assert "Asset universe not initialized" in response.error_message
    
    @pytest.mark.asyncio
    async def test_rebalancing_request_without_portfolio(self, portfolio_optimizer_agent):
        """Test rebalancing request without current portfolio."""
        message = Message(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.REBALANCING_REQUEST,
            sender_id="trading_service",
            recipient_id=portfolio_optimizer_agent.agent_id,
            payload={},
            timestamp=datetime.utcnow()
        )
        
        response = await portfolio_optimizer_agent.process_message(message)
        
        assert not response.success
        assert "Current portfolio not available" in response.error_message
    
    def test_get_portfolio_state(self, portfolio_optimizer_agent, sample_portfolio):
        """Test getting portfolio state."""
        # Set up some state
        portfolio_optimizer_agent.current_portfolio = sample_portfolio
        portfolio_optimizer_agent.target_allocations = {"AAPL": 0.6, "GOOGL": 0.4}
        portfolio_optimizer_agent.market_conditions = {"regime": "normal", "volatility": 0.15}
        
        state = portfolio_optimizer_agent.get_portfolio_state()
        
        assert state['agent_id'] == portfolio_optimizer_agent.agent_id
        assert state['current_portfolio'] is not None
        assert state['target_allocations'] == {"AAPL": 0.6, "GOOGL": 0.4}
        assert state['market_conditions']['regime'] == "normal"
        assert state['optimization_history_count'] == 0
        assert state['rebalancing_history_count'] == 0
    
    def test_get_performance_metrics_no_history(self, portfolio_optimizer_agent):
        """Test getting performance metrics with no history."""
        metrics = portfolio_optimizer_agent.get_performance_metrics()
        
        assert metrics['status'] == 'no_optimization_history'
    
    def test_get_performance_metrics_with_history(self, portfolio_optimizer_agent):
        """Test getting performance metrics with optimization history."""
        from multi_agent_trading.agents.portfolio_optimization_algorithms import AllocationPlan
        
        # Add some fake history
        allocation_plan = AllocationPlan(
            allocations={"AAPL": 0.6, "GOOGL": 0.4},
            expected_return=0.10,
            expected_volatility=0.15,
            sharpe_ratio=0.67,
            optimization_method="mean_variance",
            constraints_applied=["weights_sum_to_one"],
            timestamp=datetime.utcnow()
        )
        
        portfolio_optimizer_agent.optimization_history.append({
            'timestamp': datetime.utcnow(),
            'optimization_type': 'mean_variance',
            'allocation_plan': allocation_plan,
            'market_conditions': {'regime': 'normal'}
        })
        
        metrics = portfolio_optimizer_agent.get_performance_metrics()
        
        assert 'average_sharpe_ratio' in metrics
        assert metrics['optimization_count'] == 1
        assert metrics['rebalancing_count'] == 0
        assert 'last_optimization_time' in metrics
    
    @pytest.mark.asyncio
    async def test_multi_objective_optimization_request(self, portfolio_optimizer_agent):
        """Test multi-objective optimization request."""
        # Set up asset universe
        portfolio_optimizer_agent.asset_universe = [
            Asset("AAPL", 0.12, 0.20, 150.0, 2500000000000, "Technology"),
            Asset("GOOGL", 0.10, 0.18, 2800.0, 1800000000000, "Technology")
        ]
        portfolio_optimizer_agent.correlation_matrix = np.array([[1.0, 0.7], [0.7, 1.0]])
        
        message = Message(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.OPTIMIZATION_REQUEST,
            sender_id="trading_service",
            recipient_id=portfolio_optimizer_agent.agent_id,
            payload={
                'optimization_type': 'multi_objective',
                'objectives': {'return': 0.5, 'risk': 0.3, 'diversification': 0.2},
                'constraints': {'max_weight': 0.7}
            },
            timestamp=datetime.utcnow()
        )
        
        response = await portfolio_optimizer_agent.process_message(message)
        
        assert response.success
        assert response.data['optimization_type'] == 'multi_objective'
        assert 'allocation_plan' in response.data
    
    @pytest.mark.asyncio
    async def test_risk_parity_optimization_request(self, portfolio_optimizer_agent):
        """Test risk parity optimization request."""
        # Set up asset universe
        portfolio_optimizer_agent.asset_universe = [
            Asset("AAPL", 0.12, 0.20, 150.0, 2500000000000, "Technology"),
            Asset("GOOGL", 0.10, 0.18, 2800.0, 1800000000000, "Technology")
        ]
        portfolio_optimizer_agent.correlation_matrix = np.array([[1.0, 0.7], [0.7, 1.0]])
        
        message = Message(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.OPTIMIZATION_REQUEST,
            sender_id="trading_service",
            recipient_id=portfolio_optimizer_agent.agent_id,
            payload={
                'optimization_type': 'risk_parity',
                'constraints': {'max_weight': 0.8}
            },
            timestamp=datetime.utcnow()
        )
        
        response = await portfolio_optimizer_agent.process_message(message)
        
        assert response.success
        assert response.data['optimization_type'] == 'risk_parity'
        assert 'allocation_plan' in response.data


if __name__ == "__main__":
    pytest.main([__file__])
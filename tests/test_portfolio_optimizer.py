"""
Unit tests for Portfolio Optimizer Agent and optimization algorithms.
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from multi_agent_trading.agents.portfolio_optimizer import (
    PortfolioOptimizer, PortfolioOptimizerAgent, Asset, AllocationPlan,
    EfficientFrontier, RebalancingPlan
)
from multi_agent_trading.models.config_models import AgentConfig, MessageQueueConfig
from multi_agent_trading.models.trading_models import (
    TradingProposal, TradeAction, RiskMetrics, Portfolio, Position, MarketData
)
from multi_agent_trading.models.message_models import Message, MessageType


class TestAsset:
    """Test Asset data class."""
    
    def test_asset_creation(self):
        """Test creating a valid asset."""
        asset = Asset(
            symbol="AAPL",
            expected_return=0.12,
            volatility=0.20,
            current_price=150.0,
            market_cap=2500000000000,
            sector="Technology"
        )
        
        assert asset.symbol == "AAPL"
        assert asset.expected_return == 0.12
        assert asset.volatility == 0.20
        assert asset.current_price == 150.0
        assert asset.market_cap == 2500000000000
        assert asset.sector == "Technology"
    
    def test_asset_validation(self):
        """Test asset validation."""
        # Valid asset should not raise
        asset = Asset("AAPL", 0.12, 0.20, 150.0, 2500000000000, "Technology")
        asset.validate()
        
        # Invalid symbol
        with pytest.raises(ValueError, match="Symbol must be a non-empty string"):
            Asset("", 0.12, 0.20, 150.0, 2500000000000, "Technology").validate()
        
        # Invalid expected return
        with pytest.raises(ValueError, match="Expected return must be a number"):
            Asset("AAPL", "invalid", 0.20, 150.0, 2500000000000, "Technology").validate()
        
        # Negative volatility
        with pytest.raises(ValueError, match="Volatility must be a non-negative number"):
            Asset("AAPL", 0.12, -0.20, 150.0, 2500000000000, "Technology").validate()
        
        # Non-positive price
        with pytest.raises(ValueError, match="Current price must be positive"):
            Asset("AAPL", 0.12, 0.20, 0.0, 2500000000000, "Technology").validate()


class TestAllocationPlan:
    """Test AllocationPlan data class."""
    
    def test_allocation_plan_creation(self):
        """Test creating a valid allocation plan."""
        allocations = {"AAPL": 0.4, "GOOGL": 0.3, "MSFT": 0.3}
        plan = AllocationPlan(
            allocations=allocations,
            expected_return=0.10,
            expected_volatility=0.15,
            sharpe_ratio=0.53,
            optimization_method="mean_variance",
            constraints_applied=["weights_sum_to_one"],
            timestamp=datetime.utcnow()
        )
        
        assert plan.allocations == allocations
        assert plan.expected_return == 0.10
        assert plan.expected_volatility == 0.15
        assert plan.sharpe_ratio == 0.53
        assert plan.optimization_method == "mean_variance"
    
    def test_allocation_plan_validation(self):
        """Test allocation plan validation."""
        # Valid plan should not raise
        plan = AllocationPlan(
            allocations={"AAPL": 0.5, "GOOGL": 0.5},
            expected_return=0.10,
            expected_volatility=0.15,
            sharpe_ratio=0.53,
            optimization_method="mean_variance",
            constraints_applied=[],
            timestamp=datetime.utcnow()
        )
        plan.validate()
        
        # Weights don't sum to 1
        with pytest.raises(ValueError, match="Allocation weights must sum to 1.0"):
            AllocationPlan(
                allocations={"AAPL": 0.6, "GOOGL": 0.5},
                expected_return=0.10,
                expected_volatility=0.15,
                sharpe_ratio=0.53,
                optimization_method="mean_variance",
                constraints_applied=[],
                timestamp=datetime.utcnow()
            ).validate()
        
        # Negative weight
        with pytest.raises(ValueError, match="Weight for AAPL must be non-negative"):
            AllocationPlan(
                allocations={"AAPL": -0.1, "GOOGL": 1.1},
                expected_return=0.10,
                expected_volatility=0.15,
                sharpe_ratio=0.53,
                optimization_method="mean_variance",
                constraints_applied=[],
                timestamp=datetime.utcnow()
            ).validate()


class TestPortfolioOptimizer:
    """Test PortfolioOptimizer class."""
    
    @pytest.fixture
    def sample_assets(self):
        """Create sample assets for testing."""
        return [
            Asset("AAPL", 0.12, 0.20, 150.0, 2500000000000, "Technology"),
            Asset("GOOGL", 0.10, 0.18, 2800.0, 1800000000000, "Technology"),
            Asset("MSFT", 0.11, 0.19, 300.0, 2200000000000, "Technology"),
            Asset("JPM", 0.08, 0.25, 140.0, 400000000000, "Financial"),
            Asset("JNJ", 0.06, 0.12, 170.0, 450000000000, "Healthcare")
        ]
    
    @pytest.fixture
    def sample_correlation_matrix(self):
        """Create sample correlation matrix."""
        return np.array([
            [1.00, 0.70, 0.65, 0.30, 0.20],
            [0.70, 1.00, 0.60, 0.25, 0.15],
            [0.65, 0.60, 1.00, 0.35, 0.25],
            [0.30, 0.25, 0.35, 1.00, 0.40],
            [0.20, 0.15, 0.25, 0.40, 1.00]
        ])
    
    @pytest.fixture
    def optimizer(self):
        """Create portfolio optimizer instance."""
        return PortfolioOptimizer(risk_free_rate=0.02)
    
    def test_mean_variance_optimization(self, optimizer, sample_assets, sample_correlation_matrix):
        """Test mean-variance optimization."""
        result = optimizer.mean_variance_optimization(
            sample_assets, sample_correlation_matrix
        )
        
        assert isinstance(result, AllocationPlan)
        assert len(result.allocations) == len(sample_assets)
        assert abs(sum(result.allocations.values()) - 1.0) < 1e-6
        assert result.expected_return > 0
        assert result.expected_volatility > 0
        assert result.optimization_method == "mean_variance"
        
        # All weights should be non-negative
        for weight in result.allocations.values():
            assert weight >= 0
    
    def test_mean_variance_optimization_with_target_return(self, optimizer, sample_assets, sample_correlation_matrix):
        """Test mean-variance optimization with target return."""
        target_return = 0.09
        result = optimizer.mean_variance_optimization(
            sample_assets, sample_correlation_matrix, target_return=target_return
        )
        
        assert isinstance(result, AllocationPlan)
        assert abs(result.expected_return - target_return) < 1e-3
        assert f"target_return_{target_return}" in result.constraints_applied
    
    def test_mean_variance_optimization_with_constraints(self, optimizer, sample_assets, sample_correlation_matrix):
        """Test mean-variance optimization with constraints."""
        constraints = {"max_weight": 0.4, "min_weight": 0.05}
        result = optimizer.mean_variance_optimization(
            sample_assets, sample_correlation_matrix, constraints=constraints
        )
        
        assert isinstance(result, AllocationPlan)
        
        # Check weight constraints
        for weight in result.allocations.values():
            assert weight >= 0.05
            assert weight <= 0.4
        
        assert "max_weight_0.4" in result.constraints_applied
        assert "min_weight_0.05" in result.constraints_applied
    
    def test_mean_variance_optimization_empty_assets(self, optimizer):
        """Test mean-variance optimization with empty assets list."""
        with pytest.raises(ValueError, match="Assets list cannot be empty"):
            optimizer.mean_variance_optimization([], np.array([]))
    
    def test_mean_variance_optimization_invalid_correlation_matrix(self, optimizer, sample_assets):
        """Test mean-variance optimization with invalid correlation matrix."""
        invalid_matrix = np.array([[1.0, 0.5], [0.5, 1.0]])  # Wrong size
        
        with pytest.raises(ValueError, match="Correlation matrix dimensions must match"):
            optimizer.mean_variance_optimization(sample_assets, invalid_matrix)
    
    def test_calculate_efficient_frontier(self, optimizer, sample_assets, sample_correlation_matrix):
        """Test efficient frontier calculation."""
        result = optimizer.calculate_efficient_frontier(
            sample_assets, sample_correlation_matrix, num_points=10
        )
        
        assert isinstance(result, EfficientFrontier)
        assert len(result.returns) <= 10  # May be less due to optimization failures
        assert len(result.volatilities) == len(result.returns)
        assert len(result.sharpe_ratios) == len(result.returns)
        assert len(result.allocations) == len(result.returns)
        
        # Check that returns are generally increasing with volatility
        if len(result.returns) > 1:
            assert result.returns[-1] >= result.returns[0]
        
        # Optimal portfolios should be valid
        assert isinstance(result.optimal_portfolio, dict)
        assert isinstance(result.min_variance_portfolio, dict)
        assert abs(sum(result.optimal_portfolio.values()) - 1.0) < 1e-6
        assert abs(sum(result.min_variance_portfolio.values()) - 1.0) < 1e-6
    
    def test_calculate_efficient_frontier_insufficient_assets(self, optimizer):
        """Test efficient frontier with insufficient assets."""
        single_asset = [Asset("AAPL", 0.12, 0.20, 150.0, 2500000000000, "Technology")]
        
        with pytest.raises(ValueError, match="Need at least 2 assets"):
            optimizer.calculate_efficient_frontier(single_asset, np.array([[1.0]]))
    
    def test_multi_objective_optimization(self, optimizer, sample_assets, sample_correlation_matrix):
        """Test multi-objective optimization."""
        objectives = {"return": 0.6, "risk": 0.3, "diversification": 0.1}
        result = optimizer.multi_objective_optimization(
            sample_assets, sample_correlation_matrix, objectives
        )
        
        assert isinstance(result, AllocationPlan)
        assert len(result.allocations) == len(sample_assets)
        assert abs(sum(result.allocations.values()) - 1.0) < 1e-6
        assert result.optimization_method == "multi_objective"
        assert "multi_objective" in result.constraints_applied
        
        # All weights should be non-negative
        for weight in result.allocations.values():
            assert weight >= 0
    
    def test_multi_objective_optimization_invalid_objectives(self, optimizer, sample_assets, sample_correlation_matrix):
        """Test multi-objective optimization with invalid objectives."""
        # Empty objectives
        with pytest.raises(ValueError, match="Objectives must be specified"):
            optimizer.multi_objective_optimization(
                sample_assets, sample_correlation_matrix, {}
            )
        
        # Zero sum objectives
        with pytest.raises(ValueError, match="Objectives must be specified"):
            optimizer.multi_objective_optimization(
                sample_assets, sample_correlation_matrix, {"return": 0, "risk": 0}
            )
    
    def test_multi_objective_optimization_with_constraints(self, optimizer, sample_assets, sample_correlation_matrix):
        """Test multi-objective optimization with constraints."""
        objectives = {"return": 0.7, "risk": 0.3}
        constraints = {"max_weight": 0.3, "min_weight": 0.1}
        
        result = optimizer.multi_objective_optimization(
            sample_assets, sample_correlation_matrix, objectives, constraints
        )
        
        assert isinstance(result, AllocationPlan)
        
        # Check weight constraints
        for weight in result.allocations.values():
            assert weight >= 0.1
            assert weight <= 0.3
    
    def test_generate_allocation_plan_normal_market(self, optimizer, sample_assets, sample_correlation_matrix):
        """Test allocation plan generation for normal market conditions."""
        market_conditions = {
            'regime': 'normal',
            'volatility': 0.15
        }
        
        result = optimizer.generate_allocation_plan(
            sample_assets, sample_correlation_matrix, market_conditions
        )
        
        assert isinstance(result, AllocationPlan)
        assert result.optimization_method == "multi_objective"
        assert abs(sum(result.allocations.values()) - 1.0) < 1e-6
    
    def test_generate_allocation_plan_high_volatility(self, optimizer, sample_assets, sample_correlation_matrix):
        """Test allocation plan generation for high volatility market."""
        market_conditions = {
            'regime': 'high_volatility',
            'volatility': 0.30
        }
        
        result = optimizer.generate_allocation_plan(
            sample_assets, sample_correlation_matrix, market_conditions
        )
        
        assert isinstance(result, AllocationPlan)
        # In high volatility, should have more conservative allocation
        assert result.optimization_method == "multi_objective"
    
    def test_calculate_rebalancing_plan(self, optimizer, sample_assets):
        """Test rebalancing plan calculation."""
        # Create mock portfolio
        positions = {
            "AAPL": Position("AAPL", 100, 140.0, 150.0, 15000.0, 1000.0, 0.3),
            "GOOGL": Position("GOOGL", 10, 2700.0, 2800.0, 28000.0, 1000.0, 0.56),
            "MSFT": Position("MSFT", 20, 290.0, 300.0, 6000.0, 200.0, 0.12)
        }
        
        portfolio = Portfolio(
            portfolio_id="test_portfolio",
            total_value=50000.0,
            cash=1000.0,
            positions=positions,
            timestamp=datetime.utcnow()
        )
        
        target_allocations = {
            "AAPL": 0.4,
            "GOOGL": 0.4,
            "MSFT": 0.2
        }
        
        transaction_costs = {
            "AAPL": 0.001,
            "GOOGL": 0.001,
            "MSFT": 0.001
        }
        
        market_data = {
            "AAPL": MarketData("AAPL", datetime.utcnow(), 150.0, 1000, 149.5, 150.5, {}),
            "GOOGL": MarketData("GOOGL", datetime.utcnow(), 2800.0, 100, 2799.0, 2801.0, {}),
            "MSFT": MarketData("MSFT", datetime.utcnow(), 300.0, 500, 299.5, 300.5, {})
        }
        
        result = optimizer.calculate_rebalancing_plan(
            portfolio, target_allocations, transaction_costs, market_data
        )
        
        assert isinstance(result, RebalancingPlan)
        assert isinstance(result.current_allocations, dict)
        assert isinstance(result.target_allocations, dict)
        assert isinstance(result.trades_required, dict)
        assert result.rebalancing_cost >= 0
        assert isinstance(result.rationale, str)
        assert len(result.rationale) > 0
    
    def test_analyze_diversification(self, optimizer, sample_assets, sample_correlation_matrix):
        """Test diversification analysis."""
        allocations = {
            "AAPL": 0.3,
            "GOOGL": 0.25,
            "MSFT": 0.25,
            "JPM": 0.1,
            "JNJ": 0.1
        }
        
        result = optimizer.analyze_diversification(
            allocations, sample_assets, sample_correlation_matrix
        )
        
        assert isinstance(result, dict)
        assert 'herfindahl_index' in result
        assert 'effective_assets' in result
        assert 'sector_concentration' in result
        assert 'diversification_ratio' in result
        assert 'sector_weights' in result
        assert 'concentration_risk' in result
        
        assert 0 <= result['herfindahl_index'] <= 1
        assert result['effective_assets'] > 0
        assert result['diversification_ratio'] > 0
        assert result['concentration_risk'] in ['LOW', 'MEDIUM', 'HIGH']
    
    def test_apply_constraints(self, optimizer):
        """Test constraint application."""
        allocations = {
            "AAPL": 0.5,  # Too high
            "GOOGL": 0.05,  # Too low
            "MSFT": 0.45
        }
        
        constraints = {
            "max_weight": 0.4,
            "min_weight": 0.1
        }
        
        adjusted_allocations, violations = optimizer.apply_constraints(allocations, constraints)
        
        assert isinstance(adjusted_allocations, dict)
        assert isinstance(violations, list)
        
        # Check constraints are applied
        for weight in adjusted_allocations.values():
            assert weight <= 0.4
            assert weight >= 0.1 or weight == 0.0
        
        # Check weights still sum to 1
        assert abs(sum(adjusted_allocations.values()) - 1.0) < 1e-6
        
        # Should have violations recorded
        assert len(violations) > 0


class TestPortfolioOptimizerAgent:
    """Test PortfolioOptimizerAgent class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock agent configuration."""
        config = Mock(spec=AgentConfig)
        config.agent_type = "portfolio_optimizer"
        config.model_version = "1.0.0"
        config.model_path = "/path/to/model"
        config.timeout_seconds = 30
        config.health_check_interval = 60
        config.risk_free_rate = 0.02
        config.message_queue_config = Mock(spec=MessageQueueConfig)
        config.message_queue_config.to_dict.return_value = {}
        return config
    
    @pytest.fixture
    def agent(self, mock_config):
        """Create portfolio optimizer agent instance."""
        with patch('multi_agent_trading.agents.base_agent.MessageBus'), \
             patch('multi_agent_trading.agents.base_agent.ModelManager'), \
             patch('multi_agent_trading.agents.base_agent.MetricsCollector'):
            return PortfolioOptimizerAgent("portfolio_optimizer_1", mock_config)
    
    def test_agent_initialization(self, agent, mock_config):
        """Test agent initialization."""
        assert agent.agent_id == "portfolio_optimizer_1"
        assert agent.config == mock_config
        assert isinstance(agent.optimizer, PortfolioOptimizer)
        assert agent.optimizer.risk_free_rate == 0.02
    
    @pytest.mark.asyncio
    async def test_vote_on_decision(self, agent):
        """Test voting on trading proposal."""
        # Create sample proposal
        risk_metrics = RiskMetrics(
            var_95=-0.05,
            cvar_95=-0.08,
            sharpe_ratio=1.2,
            max_drawdown=0.15,
            volatility=0.20
        )
        
        proposal = TradingProposal(
            proposal_id="test_proposal_1",
            symbol="AAPL",
            action=TradeAction.BUY,
            quantity=100,
            price_target=150.0,
            rationale="Technical analysis indicates buy signal",
            confidence=0.8,
            risk_metrics=risk_metrics,
            timestamp=datetime.utcnow()
        )
        
        vote = await agent.vote_on_decision(proposal)
        
        assert vote.agent_id == "portfolio_optimizer_1"
        assert vote.proposal_id == "test_proposal_1"
        assert 0 <= vote.score <= 100
        assert 0 <= vote.confidence <= 1
        assert isinstance(vote.rationale, str)
        assert len(vote.rationale) > 0
        assert isinstance(vote.timestamp, datetime)
    
    @pytest.mark.asyncio
    async def test_process_message_optimization_request(self, agent):
        """Test processing optimization request message."""
        assets_data = [
            {
                'symbol': 'AAPL',
                'expected_return': 0.12,
                'volatility': 0.20,
                'current_price': 150.0,
                'market_cap': 2500000000000,
                'sector': 'Technology'
            },
            {
                'symbol': 'GOOGL',
                'expected_return': 0.10,
                'volatility': 0.18,
                'current_price': 2800.0,
                'market_cap': 1800000000000,
                'sector': 'Technology'
            },
            {
                'symbol': 'MSFT',
                'expected_return': 0.11,
                'volatility': 0.19,
                'current_price': 300.0,
                'market_cap': 2200000000000,
                'sector': 'Technology'
            }
        ]
        
        message = Message(
            message_id="test_msg_1",
            message_type=MessageType.OPTIMIZATION_REQUEST,
            sender_id="test_sender",
            recipient_id="portfolio_optimizer_1",
            data={
                "assets": assets_data, 
                "constraints": {},
                "correlation_matrix": [[1.0, 0.7, 0.6], [0.7, 1.0, 0.5], [0.6, 0.5, 1.0]]
            },
            timestamp=datetime.utcnow()
        )
        
        response = await agent.process_message(message)
        
        assert response.original_message_id == "test_msg_1"
        assert response.agent_id == "portfolio_optimizer_1"
        assert response.success is True
    
    @pytest.mark.asyncio
    async def test_process_message_rebalancing_request(self, agent):
        """Test processing rebalancing request message."""
        # Set up a current portfolio first
        positions = {
            "AAPL": {
                "quantity": 100,
                "entry_price": 140.0,
                "current_price": 150.0,
                "market_value": 15000.0,
                "unrealized_pnl": 1000.0,
                "weight": 0.3
            }
        }
        
        portfolio_data = {
            "portfolio_id": "test_portfolio",
            "total_value": 50000.0,
            "cash": 1000.0,
            "positions": positions,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        message = Message(
            message_id="test_msg_2",
            message_type=MessageType.REBALANCING_REQUEST,
            sender_id="test_sender",
            recipient_id="portfolio_optimizer_1",
            data={
                "current_portfolio": portfolio_data,
                "target_allocations": {"AAPL": 0.4},
                "transaction_costs": {"AAPL": 0.001},
                "market_data": {
                    "AAPL": {
                        "timestamp": datetime.utcnow().isoformat(),
                        "price": 150.0,
                        "volume": 1000,
                        "bid": 149.5,
                        "ask": 150.5,
                        "technical_indicators": {}
                    }
                }
            },
            timestamp=datetime.utcnow()
        )
        
        response = await agent.process_message(message)
        
        assert response.original_message_id == "test_msg_2"
        assert response.agent_id == "portfolio_optimizer_1"
        assert response.success is True
    
    @pytest.mark.asyncio
    async def test_process_message_efficient_frontier_request(self, agent):
        """Test processing efficient frontier request message."""
        assets_data = [
            {
                'symbol': 'AAPL',
                'expected_return': 0.12,
                'volatility': 0.20,
                'current_price': 150.0,
                'market_cap': 2500000000000,
                'sector': 'Technology'
            },
            {
                'symbol': 'GOOGL',
                'expected_return': 0.10,
                'volatility': 0.18,
                'current_price': 2800.0,
                'market_cap': 1800000000000,
                'sector': 'Technology'
            }
        ]
        
        message = Message(
            message_id="test_msg_3",
            message_type=MessageType.EFFICIENT_FRONTIER_REQUEST,
            sender_id="test_sender",
            recipient_id="portfolio_optimizer_1",
            data={"assets": assets_data, "num_points": 10},
            timestamp=datetime.utcnow()
        )
        
        response = await agent.process_message(message)
        
        assert response.original_message_id == "test_msg_3"
        assert response.agent_id == "portfolio_optimizer_1"
        assert response.success is True
    
    @pytest.mark.asyncio
    async def test_process_message_unsupported_type(self, agent):
        """Test processing unsupported message type."""
        message = Message(
            message_id="test_msg_4",
            message_type=MessageType.SYSTEM_ALERT,  # Truly unsupported message type
            sender_id="test_sender",
            recipient_id="portfolio_optimizer_1",
            data={},
            timestamp=datetime.utcnow()
        )
        
        response = await agent.process_message(message)
        
        assert response.original_message_id == "test_msg_4"
        assert response.agent_id == "portfolio_optimizer_1"
        assert response.success is False
        assert "Unsupported message type" in response.error_message


if __name__ == "__main__":
    pytest.main([__file__])
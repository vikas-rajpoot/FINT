"""
Unit tests for portfolio optimization algorithms.
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch

from multi_agent_trading.agents.portfolio_optimization_algorithms import (
    PortfolioOptimizationAlgorithms,
    PortfolioRebalancingEngine,
    Asset,
    AllocationPlan,
    EfficientFrontier,
    RebalancingPlan
)


class TestAsset:
    """Test Asset dataclass."""
    
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
    
    def test_asset_validation_empty_symbol(self):
        """Test asset validation with empty symbol."""
        asset = Asset(
            symbol="",
            expected_return=0.12,
            volatility=0.20,
            current_price=150.0,
            market_cap=2500000000000,
            sector="Technology"
        )
        
        with pytest.raises(ValueError, match="Symbol must be a non-empty string"):
            asset.validate()
    
    def test_asset_validation_negative_volatility(self):
        """Test asset validation with negative volatility."""
        asset = Asset(
            symbol="AAPL",
            expected_return=0.12,
            volatility=-0.20,
            current_price=150.0,
            market_cap=2500000000000,
            sector="Technology"
        )
        
        with pytest.raises(ValueError, match="Volatility must be a non-negative number"):
            asset.validate()
    
    def test_asset_validation_zero_price(self):
        """Test asset validation with zero price."""
        asset = Asset(
            symbol="AAPL",
            expected_return=0.12,
            volatility=0.20,
            current_price=0.0,
            market_cap=2500000000000,
            sector="Technology"
        )
        
        with pytest.raises(ValueError, match="Current price must be positive"):
            asset.validate()


class TestAllocationPlan:
    """Test AllocationPlan dataclass."""
    
    def test_allocation_plan_creation(self):
        """Test creating a valid allocation plan."""
        plan = AllocationPlan(
            allocations={"AAPL": 0.6, "GOOGL": 0.4},
            expected_return=0.10,
            expected_volatility=0.15,
            sharpe_ratio=0.53,
            optimization_method="mean_variance",
            constraints_applied=["weights_sum_to_one"],
            timestamp=datetime.utcnow()
        )
        
        assert plan.allocations == {"AAPL": 0.6, "GOOGL": 0.4}
        assert plan.expected_return == 0.10
        assert plan.sharpe_ratio == 0.53
    
    def test_allocation_plan_validation_weights_sum(self):
        """Test allocation plan validation with weights not summing to 1."""
        plan = AllocationPlan(
            allocations={"AAPL": 0.7, "GOOGL": 0.4},  # Sum = 1.1
            expected_return=0.10,
            expected_volatility=0.15,
            sharpe_ratio=0.53,
            optimization_method="mean_variance",
            constraints_applied=["weights_sum_to_one"],
            timestamp=datetime.utcnow()
        )
        
        with pytest.raises(ValueError, match="Allocation weights must sum to 1.0"):
            plan.validate()
    
    def test_allocation_plan_validation_negative_weight(self):
        """Test allocation plan validation with negative weight."""
        plan = AllocationPlan(
            allocations={"AAPL": 1.2, "GOOGL": -0.2},  # Negative weight
            expected_return=0.10,
            expected_volatility=0.15,
            sharpe_ratio=0.53,
            optimization_method="mean_variance",
            constraints_applied=["weights_sum_to_one"],
            timestamp=datetime.utcnow()
        )
        
        with pytest.raises(ValueError, match="Weight for GOOGL must be non-negative"):
            plan.validate()


# Module-level fixtures for shared use
@pytest.fixture
def sample_assets():
    """Create sample assets for testing."""
    return [
        Asset("AAPL", 0.12, 0.20, 150.0, 2500000000000, "Technology"),
        Asset("GOOGL", 0.10, 0.18, 2800.0, 1800000000000, "Technology"),
        Asset("MSFT", 0.11, 0.19, 300.0, 2200000000000, "Technology"),
        Asset("JPM", 0.08, 0.25, 140.0, 400000000000, "Financial"),
        Asset("JNJ", 0.06, 0.15, 170.0, 450000000000, "Healthcare")
    ]

@pytest.fixture
def sample_correlation_matrix():
    """Create sample correlation matrix."""
    return np.array([
        [1.00, 0.70, 0.65, 0.30, 0.20],
        [0.70, 1.00, 0.60, 0.25, 0.15],
        [0.65, 0.60, 1.00, 0.35, 0.25],
        [0.30, 0.25, 0.35, 1.00, 0.40],
        [0.20, 0.15, 0.25, 0.40, 1.00]
    ])


class TestPortfolioOptimizationAlgorithms:
    """Test PortfolioOptimizationAlgorithms class."""
    
    @pytest.fixture
    def optimizer(self):
        """Create optimizer instance."""
        return PortfolioOptimizationAlgorithms(risk_free_rate=0.02)
    
    def test_mean_variance_optimization_basic(self, optimizer, sample_assets, sample_correlation_matrix):
        """Test basic mean-variance optimization."""
        result = optimizer.mean_variance_optimization(
            sample_assets,
            sample_correlation_matrix
        )
        
        assert isinstance(result, AllocationPlan)
        assert result.optimization_method == "mean_variance"
        assert len(result.allocations) == len(sample_assets)
        assert abs(sum(result.allocations.values()) - 1.0) < 1e-6
        assert all(weight >= 0 for weight in result.allocations.values())
        assert result.expected_return > 0
        assert result.expected_volatility > 0
        assert result.sharpe_ratio > 0
    
    def test_mean_variance_optimization_with_target_return(self, optimizer, sample_assets, sample_correlation_matrix):
        """Test mean-variance optimization with target return."""
        target_return = 0.09
        
        result = optimizer.mean_variance_optimization(
            sample_assets,
            sample_correlation_matrix,
            target_return=target_return
        )
        
        assert isinstance(result, AllocationPlan)
        assert abs(result.expected_return - target_return) < 1e-3
        assert f"target_return_{target_return:.4f}" in result.constraints_applied
    
    def test_mean_variance_optimization_with_constraints(self, optimizer, sample_assets, sample_correlation_matrix):
        """Test mean-variance optimization with constraints."""
        constraints = {
            'max_weight': 0.4,
            'min_weight': 0.05
        }
        
        result = optimizer.mean_variance_optimization(
            sample_assets,
            sample_correlation_matrix,
            constraints=constraints
        )
        
        assert isinstance(result, AllocationPlan)
        assert all(weight <= 0.4 + 1e-6 for weight in result.allocations.values())
        assert all(weight >= 0.05 - 1e-6 or weight == 0 for weight in result.allocations.values())
        assert "max_weight_0.4" in result.constraints_applied
        assert "min_weight_0.05" in result.constraints_applied
    
    def test_mean_variance_optimization_empty_assets(self, optimizer):
        """Test mean-variance optimization with empty assets list."""
        with pytest.raises(ValueError, match="Assets list cannot be empty"):
            optimizer.mean_variance_optimization([], np.array([]))
    
    def test_mean_variance_optimization_mismatched_correlation_matrix(self, optimizer, sample_assets):
        """Test mean-variance optimization with mismatched correlation matrix."""
        wrong_matrix = np.array([[1.0, 0.5], [0.5, 1.0]])  # 2x2 instead of 5x5
        
        with pytest.raises(ValueError, match="Correlation matrix dimensions must match number of assets"):
            optimizer.mean_variance_optimization(sample_assets, wrong_matrix)
    
    def test_calculate_efficient_frontier(self, optimizer, sample_assets, sample_correlation_matrix):
        """Test efficient frontier calculation."""
        result = optimizer.calculate_efficient_frontier(
            sample_assets,
            sample_correlation_matrix,
            num_points=20
        )
        
        assert isinstance(result, EfficientFrontier)
        assert len(result.returns) > 0
        assert len(result.volatilities) > 0
        assert len(result.sharpe_ratios) > 0
        assert len(result.allocations) > 0
        assert len(result.returns) == len(result.volatilities)
        assert len(result.returns) == len(result.sharpe_ratios)
        assert len(result.returns) == len(result.allocations)
        
        # Check that we have reasonable range of returns and volatilities
        assert max(result.returns) > min(result.returns)
        assert max(result.volatilities) > min(result.volatilities)
        
        # Check that optimal portfolios exist
        assert len(result.optimal_portfolio) > 0
        assert len(result.min_variance_portfolio) > 0
        
        # Check that all allocations sum to 1
        for allocation in result.allocations:
            assert abs(sum(allocation.values()) - 1.0) < 1e-6
    
    def test_calculate_efficient_frontier_insufficient_assets(self, optimizer):
        """Test efficient frontier with insufficient assets."""
        single_asset = [Asset("AAPL", 0.12, 0.20, 150.0, 2500000000000, "Technology")]
        correlation_matrix = np.array([[1.0]])
        
        with pytest.raises(ValueError, match="Need at least 2 assets for efficient frontier"):
            optimizer.calculate_efficient_frontier(single_asset, correlation_matrix)
    
    def test_calculate_efficient_frontier_insufficient_points(self, optimizer, sample_assets, sample_correlation_matrix):
        """Test efficient frontier with insufficient points."""
        with pytest.raises(ValueError, match="Need at least 10 points for meaningful frontier"):
            optimizer.calculate_efficient_frontier(
                sample_assets,
                sample_correlation_matrix,
                num_points=5
            )
    
    def test_multi_objective_optimization(self, optimizer, sample_assets, sample_correlation_matrix):
        """Test multi-objective optimization."""
        objectives = {
            'return': 0.6,
            'risk': 0.3,
            'diversification': 0.1
        }
        
        result = optimizer.multi_objective_optimization(
            sample_assets,
            sample_correlation_matrix,
            objectives
        )
        
        assert isinstance(result, AllocationPlan)
        assert result.optimization_method == "multi_objective"
        assert abs(sum(result.allocations.values()) - 1.0) < 1e-6
        assert all(weight >= 0 for weight in result.allocations.values())
        assert any("objectives_" in constraint for constraint in result.constraints_applied)
    
    def test_multi_objective_optimization_empty_objectives(self, optimizer, sample_assets, sample_correlation_matrix):
        """Test multi-objective optimization with empty objectives."""
        with pytest.raises(ValueError, match="Objectives must be specified with non-zero weights"):
            optimizer.multi_objective_optimization(
                sample_assets,
                sample_correlation_matrix,
                {}
            )
    
    def test_multi_objective_optimization_invalid_objectives(self, optimizer, sample_assets, sample_correlation_matrix):
        """Test multi-objective optimization with invalid objectives."""
        objectives = {
            'return': 0.5,
            'invalid_objective': 0.5
        }
        
        with pytest.raises(ValueError, match="Invalid objectives"):
            optimizer.multi_objective_optimization(
                sample_assets,
                sample_correlation_matrix,
                objectives
            )
    
    def test_multi_objective_optimization_with_momentum(self, optimizer, sample_assets, sample_correlation_matrix):
        """Test multi-objective optimization with momentum objective."""
        objectives = {
            'return': 0.4,
            'risk': 0.3,
            'momentum': 0.3
        }
        
        result = optimizer.multi_objective_optimization(
            sample_assets,
            sample_correlation_matrix,
            objectives
        )
        
        assert isinstance(result, AllocationPlan)
        assert result.optimization_method == "multi_objective"
    
    def test_multi_objective_optimization_with_value(self, optimizer, sample_assets, sample_correlation_matrix):
        """Test multi-objective optimization with value objective."""
        objectives = {
            'return': 0.4,
            'risk': 0.3,
            'value': 0.3
        }
        
        result = optimizer.multi_objective_optimization(
            sample_assets,
            sample_correlation_matrix,
            objectives
        )
        
        assert isinstance(result, AllocationPlan)
        assert result.optimization_method == "multi_objective"
    
    def test_calculate_risk_parity_allocation(self, optimizer, sample_assets, sample_correlation_matrix):
        """Test risk parity allocation calculation."""
        result = optimizer.calculate_risk_parity_allocation(
            sample_assets,
            sample_correlation_matrix
        )
        
        assert isinstance(result, AllocationPlan)
        assert result.optimization_method == "risk_parity"
        assert abs(sum(result.allocations.values()) - 1.0) < 1e-6
        assert all(weight >= 0 for weight in result.allocations.values())
        assert "risk_parity" in result.constraints_applied
        
        # In risk parity, lower volatility assets should generally have higher weights
        volatilities = {asset.symbol: asset.volatility for asset in sample_assets}
        
        # Find the lowest and highest volatility assets
        min_vol_asset = min(sample_assets, key=lambda x: x.volatility)
        max_vol_asset = max(sample_assets, key=lambda x: x.volatility)
        
        # The lowest volatility asset should have higher or equal weight than highest volatility asset
        assert result.allocations[min_vol_asset.symbol] >= result.allocations[max_vol_asset.symbol]
    
    def test_optimization_with_singular_covariance_matrix(self, optimizer):
        """Test optimization with singular covariance matrix."""
        # Create assets with identical volatilities and perfect correlation
        identical_assets = [
            Asset("ASSET1", 0.10, 0.20, 100.0, 1000000000, "Sector1"),
            Asset("ASSET2", 0.10, 0.20, 100.0, 1000000000, "Sector1")
        ]
        
        # Perfect correlation matrix (singular)
        perfect_correlation = np.array([[1.0, 1.0], [1.0, 1.0]])
        
        # Should handle singular matrix with regularization
        result = optimizer.mean_variance_optimization(
            identical_assets,
            perfect_correlation
        )
        
        assert isinstance(result, AllocationPlan)
        assert abs(sum(result.allocations.values()) - 1.0) < 1e-6
    
    def test_optimization_with_extreme_parameters(self, optimizer):
        """Test optimization with extreme parameters."""
        extreme_assets = [
            Asset("HIGH_RETURN", 0.50, 0.60, 100.0, 1000000000, "Sector1"),
            Asset("LOW_RETURN", 0.01, 0.05, 100.0, 1000000000, "Sector2"),
            Asset("HIGH_VOL", 0.15, 0.80, 100.0, 1000000000, "Sector3")
        ]
        
        correlation_matrix = np.array([
            [1.0, 0.1, 0.2],
            [0.1, 1.0, 0.1],
            [0.2, 0.1, 1.0]
        ])
        
        result = optimizer.mean_variance_optimization(
            extreme_assets,
            correlation_matrix
        )
        
        assert isinstance(result, AllocationPlan)
        assert abs(sum(result.allocations.values()) - 1.0) < 1e-6
        assert all(weight >= 0 for weight in result.allocations.values())
    
    def test_optimization_numerical_stability(self, optimizer):
        """Test optimization numerical stability with small numbers."""
        small_assets = [
            Asset("ASSET1", 0.001, 0.002, 0.01, 1000, "Sector1"),
            Asset("ASSET2", 0.002, 0.003, 0.02, 2000, "Sector2")
        ]
        
        correlation_matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
        
        result = optimizer.mean_variance_optimization(
            small_assets,
            correlation_matrix
        )
        
        assert isinstance(result, AllocationPlan)
        assert abs(sum(result.allocations.values()) - 1.0) < 1e-6
        assert all(weight >= 0 for weight in result.allocations.values())
        assert not np.isnan(result.expected_return)
        assert not np.isnan(result.expected_volatility)
        assert not np.isnan(result.sharpe_ratio)


if __name__ == "__main__":
    pytest.main([__file__])


class TestRebalancingPlan:
    """Test RebalancingPlan dataclass."""
    
    def test_rebalancing_plan_creation(self):
        """Test creating a valid rebalancing plan."""
        plan = RebalancingPlan(
            current_allocations={"AAPL": 0.7, "GOOGL": 0.3},
            target_allocations={"AAPL": 0.6, "GOOGL": 0.4},
            trades_required={"AAPL": -100, "GOOGL": 50},
            rebalancing_cost=150.0,
            expected_improvement=200.0,
            rationale="Portfolio drift requires rebalancing",
            timestamp=datetime.utcnow()
        )
        
        assert plan.current_allocations == {"AAPL": 0.7, "GOOGL": 0.3}
        assert plan.target_allocations == {"AAPL": 0.6, "GOOGL": 0.4}
        assert plan.trades_required == {"AAPL": -100, "GOOGL": 50}
        assert plan.rebalancing_cost == 150.0
        assert plan.expected_improvement == 200.0


class TestPortfolioRebalancingEngine:
    """Test PortfolioRebalancingEngine class."""
    
    @pytest.fixture
    def rebalancing_engine(self):
        """Create rebalancing engine instance."""
        return PortfolioRebalancingEngine(rebalancing_threshold=0.05, transaction_cost_rate=0.001)
    
    @pytest.fixture
    def sample_portfolio(self):
        """Create sample portfolio for testing."""
        from multi_agent_trading.models.trading_models import Portfolio, Position
        
        positions = {
            "AAPL": Position(
                symbol="AAPL",
                quantity=100,
                entry_price=140.0,
                current_price=150.0,
                market_value=15000.0,
                unrealized_pnl=1000.0,
                weight=0.6
            ),
            "GOOGL": Position(
                symbol="GOOGL",
                quantity=5,
                entry_price=2700.0,
                current_price=2800.0,
                market_value=14000.0,
                unrealized_pnl=500.0,
                weight=0.4
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
        from multi_agent_trading.models.trading_models import MarketData
        
        return {
            "AAPL": MarketData(
                symbol="AAPL",
                timestamp=datetime.utcnow(),
                price=150.0,
                volume=1000000,
                bid=149.5,
                ask=150.5,
                technical_indicators={}
            ),
            "GOOGL": MarketData(
                symbol="GOOGL",
                timestamp=datetime.utcnow(),
                price=2800.0,
                volume=500000,
                bid=2795.0,
                ask=2805.0,
                technical_indicators={}
            )
        }
    
    def test_generate_allocation_plan_normal_market(self, rebalancing_engine, sample_assets, sample_correlation_matrix):
        """Test allocation plan generation for normal market conditions."""
        market_conditions = {
            'regime': 'normal',
            'volatility': 0.15
        }
        
        result = rebalancing_engine.generate_allocation_plan(
            sample_assets,
            sample_correlation_matrix,
            market_conditions
        )
        
        assert isinstance(result, AllocationPlan)
        assert abs(sum(result.allocations.values()) - 1.0) < 1e-6
        assert all(weight >= 0 for weight in result.allocations.values())
    
    def test_generate_allocation_plan_high_volatility(self, rebalancing_engine, sample_assets, sample_correlation_matrix):
        """Test allocation plan generation for high volatility market."""
        market_conditions = {
            'regime': 'high_volatility',
            'volatility': 0.30
        }
        
        result = rebalancing_engine.generate_allocation_plan(
            sample_assets,
            sample_correlation_matrix,
            market_conditions
        )
        
        assert isinstance(result, AllocationPlan)
        # In high volatility, max weight should be reduced
        assert all(weight <= 0.25 + 1e-6 for weight in result.allocations.values())
    
    def test_generate_allocation_plan_bull_market(self, rebalancing_engine, sample_assets, sample_correlation_matrix):
        """Test allocation plan generation for bull market."""
        market_conditions = {
            'regime': 'bull_market',
            'volatility': 0.12
        }
        
        result = rebalancing_engine.generate_allocation_plan(
            sample_assets,
            sample_correlation_matrix,
            market_conditions
        )
        
        assert isinstance(result, AllocationPlan)
        assert result.optimization_method == "multi_objective"
    
    def test_generate_allocation_plan_bear_market(self, rebalancing_engine, sample_assets, sample_correlation_matrix):
        """Test allocation plan generation for bear market."""
        market_conditions = {
            'regime': 'bear_market',
            'volatility': 0.25
        }
        
        result = rebalancing_engine.generate_allocation_plan(
            sample_assets,
            sample_correlation_matrix,
            market_conditions
        )
        
        assert isinstance(result, AllocationPlan)
        assert result.optimization_method == "multi_objective"
    
    def test_calculate_rebalancing_plan(self, rebalancing_engine, sample_portfolio, sample_market_data):
        """Test rebalancing plan calculation."""
        target_allocations = {"AAPL": 0.5, "GOOGL": 0.5}
        transaction_costs = {"AAPL": 0.001, "GOOGL": 0.001}
        
        result = rebalancing_engine.calculate_rebalancing_plan(
            sample_portfolio,
            target_allocations,
            transaction_costs,
            sample_market_data
        )
        
        assert isinstance(result, RebalancingPlan)
        assert len(result.current_allocations) > 0
        assert len(result.target_allocations) > 0
        assert result.rebalancing_cost >= 0
        assert result.expected_improvement >= 0
        assert len(result.rationale) > 0
    
    def test_calculate_rebalancing_plan_zero_portfolio_value(self, rebalancing_engine, sample_market_data):
        """Test rebalancing plan with zero portfolio value."""
        from multi_agent_trading.models.trading_models import Portfolio
        
        zero_portfolio = Portfolio(
            portfolio_id="zero_portfolio",
            total_value=0.0,
            cash=0.0,
            positions={},
            timestamp=datetime.utcnow()
        )
        
        target_allocations = {"AAPL": 0.5, "GOOGL": 0.5}
        transaction_costs = {"AAPL": 0.001, "GOOGL": 0.001}
        
        with pytest.raises(ValueError, match="Portfolio total value must be positive"):
            rebalancing_engine.calculate_rebalancing_plan(
                zero_portfolio,
                target_allocations,
                transaction_costs,
                sample_market_data
            )
    
    def test_analyze_diversification(self, rebalancing_engine, sample_assets, sample_correlation_matrix):
        """Test diversification analysis."""
        allocations = {"AAPL": 0.4, "GOOGL": 0.3, "MSFT": 0.2, "JPM": 0.1}
        
        result = rebalancing_engine.analyze_diversification(
            allocations,
            sample_assets,
            sample_correlation_matrix
        )
        
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
    
    def test_analyze_diversification_single_asset(self, rebalancing_engine, sample_assets, sample_correlation_matrix):
        """Test diversification analysis with single asset."""
        allocations = {"AAPL": 1.0}
        
        result = rebalancing_engine.analyze_diversification(
            allocations,
            sample_assets,
            sample_correlation_matrix
        )
        
        assert result['herfindahl_index'] == 1.0
        assert result['effective_assets'] == 1.0
        assert result['concentration_risk'] == 'HIGH'
    
    def test_apply_constraints_max_weight(self, rebalancing_engine):
        """Test applying maximum weight constraints."""
        allocations = {"AAPL": 0.6, "GOOGL": 0.4}
        constraints = {"max_weight": 0.5}
        
        adjusted, violations = rebalancing_engine.apply_constraints(allocations, constraints)
        
        assert all(weight <= 0.5 + 1e-6 for weight in adjusted.values())
        assert abs(sum(adjusted.values()) - 1.0) < 1e-6
        assert len(violations) > 0
    
    def test_apply_constraints_min_weight(self, rebalancing_engine):
        """Test applying minimum weight constraints."""
        allocations = {"AAPL": 0.95, "GOOGL": 0.05}
        constraints = {"min_weight": 0.1}
        
        adjusted, violations = rebalancing_engine.apply_constraints(allocations, constraints)
        
        assert all(weight >= 0.1 - 1e-6 or weight == 0 for weight in adjusted.values())
        assert abs(sum(adjusted.values()) - 1.0) < 1e-6
        assert len(violations) > 0
    
    def test_should_rebalance_below_threshold(self, rebalancing_engine):
        """Test rebalancing decision below threshold."""
        current_allocations = {"AAPL": 0.51, "GOOGL": 0.49}
        target_allocations = {"AAPL": 0.5, "GOOGL": 0.5}
        
        should_rebalance, rationale = rebalancing_engine.should_rebalance(
            current_allocations, target_allocations, 100.0, 50.0
        )
        
        assert not should_rebalance
        assert "below threshold" in rationale
    
    def test_should_rebalance_high_cost(self, rebalancing_engine):
        """Test rebalancing decision with high cost."""
        current_allocations = {"AAPL": 0.6, "GOOGL": 0.4}
        target_allocations = {"AAPL": 0.5, "GOOGL": 0.5}
        
        should_rebalance, rationale = rebalancing_engine.should_rebalance(
            current_allocations, target_allocations, 1000.0, 100.0
        )
        
        assert not should_rebalance
        assert "does not exceed cost" in rationale
    
    def test_should_rebalance_large_deviation(self, rebalancing_engine):
        """Test rebalancing decision with large deviation."""
        current_allocations = {"AAPL": 0.8, "GOOGL": 0.2}
        target_allocations = {"AAPL": 0.5, "GOOGL": 0.5}
        
        should_rebalance, rationale = rebalancing_engine.should_rebalance(
            current_allocations, target_allocations, 100.0, 50.0
        )
        
        assert should_rebalance
        assert "Large deviation" in rationale
    
    def test_should_rebalance_favorable_ratio(self, rebalancing_engine):
        """Test rebalancing decision with favorable benefit-cost ratio."""
        current_allocations = {"AAPL": 0.6, "GOOGL": 0.4}
        target_allocations = {"AAPL": 0.5, "GOOGL": 0.5}
        
        should_rebalance, rationale = rebalancing_engine.should_rebalance(
            current_allocations, target_allocations, 100.0, 300.0
        )
        
        assert should_rebalance
        assert "Favorable benefit-cost ratio" in rationale
    
    def test_estimate_rebalancing_benefit(self, rebalancing_engine):
        """Test rebalancing benefit estimation."""
        current_allocations = {"AAPL": 0.7, "GOOGL": 0.3}
        target_allocations = {"AAPL": 0.5, "GOOGL": 0.5}
        total_deviation = 0.4
        
        benefit = rebalancing_engine._estimate_rebalancing_benefit(
            current_allocations, target_allocations, total_deviation
        )
        
        assert benefit > 0
        assert isinstance(benefit, float)
    
    def test_generate_rebalancing_rationale(self, rebalancing_engine):
        """Test rebalancing rationale generation."""
        deviations = {"AAPL": 0.1, "GOOGL": -0.1}
        total_deviation = 0.2
        rebalancing_cost = 100.0
        expected_improvement = 200.0
        
        rationale = rebalancing_engine._generate_rebalancing_rationale(
            deviations, total_deviation, rebalancing_cost, expected_improvement
        )
        
        assert isinstance(rationale, str)
        assert len(rationale) > 0
        assert "drifted" in rationale or "deviation" in rationale
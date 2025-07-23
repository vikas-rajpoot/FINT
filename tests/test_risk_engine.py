"""
Unit tests for the risk calculation engine.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from multi_agent_trading.services.risk_engine import RiskCalculationEngine
from multi_agent_trading.models.trading_models import (
    Portfolio, Position, RiskParameters, TradingProposal, 
    TradeAction, RiskMetrics
)


class TestRiskCalculationEngine:
    """Test cases for RiskCalculationEngine."""
    
    @pytest.fixture
    def risk_engine(self):
        """Create a risk calculation engine instance."""
        return RiskCalculationEngine(risk_free_rate=0.02)
    
    @pytest.fixture
    def sample_portfolio(self):
        """Create a sample portfolio for testing."""
        positions = {
            "AAPL": Position(
                symbol="AAPL",
                quantity=100,
                entry_price=150.0,
                current_price=155.0,
                market_value=15500.0,
                unrealized_pnl=500.0,
                weight=0.5
            ),
            "GOOGL": Position(
                symbol="GOOGL",
                quantity=50,
                entry_price=2800.0,
                current_price=2850.0,
                market_value=142500.0,
                unrealized_pnl=2500.0,
                weight=0.5
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
    def sample_risk_params(self):
        """Create sample risk parameters."""
        return RiskParameters(
            stop_loss_pct=0.02,
            take_profit_pct=0.05,
            max_position_size_pct=0.1,
            max_portfolio_risk_pct=0.02,
            volatility_adjustment=0.2
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
            rationale="Test proposal",
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
    def sample_historical_prices(self):
        """Create sample historical price data."""
        return {
            "AAPL": [145.0, 148.0, 152.0, 155.0, 153.0, 157.0],
            "GOOGL": [2750.0, 2780.0, 2820.0, 2850.0, 2830.0, 2860.0],
            "MSFT": [290.0, 295.0, 300.0, 305.0, 298.0, 302.0]
        }
    
    @pytest.fixture
    def sample_historical_returns(self):
        """Create sample historical returns data."""
        return {
            "AAPL": [0.02, 0.027, 0.02, -0.013, 0.026],
            "GOOGL": [0.011, 0.014, 0.011, -0.007, 0.011],
            "MSFT": [0.017, 0.017, 0.017, -0.023, 0.013]
        }
    
    def test_calculate_position_size_basic(self, risk_engine, sample_proposal, 
                                         sample_portfolio, sample_risk_params, 
                                         sample_historical_prices):
        """Test basic position size calculation."""
        position_size = risk_engine.calculate_position_size(
            sample_proposal, sample_portfolio, sample_risk_params, sample_historical_prices
        )
        
        assert position_size.symbol == "MSFT"
        assert position_size.recommended_quantity >= 0
        assert position_size.max_quantity >= position_size.recommended_quantity
        assert position_size.risk_amount >= 0
        assert position_size.position_value >= 0
        assert 0 <= position_size.risk_pct <= 1
        assert position_size.rationale is not None
    
    def test_calculate_position_size_zero_price(self, risk_engine, sample_portfolio, 
                                              sample_risk_params, sample_historical_prices):
        """Test position size calculation with very small price target."""
        proposal = TradingProposal(
            proposal_id="test_proposal",
            symbol="MSFT",
            action=TradeAction.BUY,
            quantity=100,
            price_target=0.01,  # Very small price
            rationale="Test proposal",
            confidence=0.8,
            risk_metrics=RiskMetrics(0.05, 0.07, 1.2, 0.1, 0.2),
            timestamp=datetime.utcnow()
        )
        
        position_size = risk_engine.calculate_position_size(
            proposal, sample_portfolio, sample_risk_params, sample_historical_prices
        )
        
        assert position_size.recommended_quantity >= 0
        assert position_size.max_quantity >= 0
    
    def test_calculate_risk_metrics_basic(self, risk_engine, sample_portfolio, 
                                        sample_historical_returns):
        """Test basic risk metrics calculation."""
        risk_metrics = risk_engine.calculate_risk_metrics(
            sample_portfolio, sample_historical_returns
        )
        
        assert isinstance(risk_metrics.var_95, float)
        assert isinstance(risk_metrics.cvar_95, float)
        assert isinstance(risk_metrics.sharpe_ratio, float)
        assert isinstance(risk_metrics.max_drawdown, float)
        assert isinstance(risk_metrics.volatility, float)
        
        assert risk_metrics.var_95 >= 0
        assert risk_metrics.cvar_95 >= 0
        assert risk_metrics.max_drawdown >= 0
        assert risk_metrics.volatility >= 0
    
    def test_calculate_risk_metrics_empty_portfolio(self, risk_engine):
        """Test risk metrics calculation with empty portfolio."""
        empty_portfolio = Portfolio(
            portfolio_id="empty",
            total_value=100000.0,
            cash=100000.0,
            positions={},
            timestamp=datetime.utcnow()
        )
        
        risk_metrics = risk_engine.calculate_risk_metrics(empty_portfolio, {})
        
        assert risk_metrics.var_95 == 0.0
        assert risk_metrics.cvar_95 == 0.0
        assert risk_metrics.sharpe_ratio == 0.0
        assert risk_metrics.max_drawdown == 0.0
        assert risk_metrics.volatility == 0.0
    
    def test_monitor_portfolio_exposure(self, risk_engine, sample_portfolio, 
                                      sample_historical_returns):
        """Test portfolio exposure monitoring."""
        sector_mapping = {"AAPL": "Technology", "GOOGL": "Technology"}
        currency_mapping = {"AAPL": "USD", "GOOGL": "USD"}
        
        exposure_report = risk_engine.monitor_portfolio_exposure(
            sample_portfolio, sector_mapping, currency_mapping, sample_historical_returns
        )
        
        assert exposure_report.total_exposure > 0
        assert "Technology" in exposure_report.sector_exposure
        assert "USD" in exposure_report.currency_exposure
        assert isinstance(exposure_report.correlation_matrix, dict)
        assert 0 <= exposure_report.concentration_risk <= 1
        assert exposure_report.diversification_ratio >= 0
        assert isinstance(exposure_report.timestamp, datetime)
    
    def test_calculate_volatility(self, risk_engine):
        """Test volatility calculation."""
        prices = [100.0, 102.0, 98.0, 105.0, 103.0, 107.0]
        volatility = risk_engine._calculate_volatility("TEST", prices)
        
        assert isinstance(volatility, float)
        assert volatility > 0
    
    def test_calculate_volatility_insufficient_data(self, risk_engine):
        """Test volatility calculation with insufficient data."""
        prices = [100.0]  # Only one price point
        volatility = risk_engine._calculate_volatility("TEST", prices)
        
        assert volatility == 0.2  # Default volatility
    
    def test_calculate_var(self, risk_engine):
        """Test VaR calculation."""
        returns = [-0.05, -0.02, 0.01, 0.03, -0.01, 0.02, -0.03, 0.04, -0.01, 0.01]
        var = risk_engine._calculate_var(returns, 0.95)
        
        assert isinstance(var, float)
        assert var >= 0
    
    def test_calculate_cvar(self, risk_engine):
        """Test CVaR calculation."""
        returns = [-0.05, -0.02, 0.01, 0.03, -0.01, 0.02, -0.03, 0.04, -0.01, 0.01]
        cvar = risk_engine._calculate_cvar(returns, 0.95)
        
        assert isinstance(cvar, float)
        assert cvar >= 0
    
    def test_calculate_sharpe_ratio(self, risk_engine):
        """Test Sharpe ratio calculation."""
        returns = [0.01, 0.02, -0.01, 0.03, 0.01, -0.02, 0.02, 0.01, 0.03, -0.01]
        sharpe = risk_engine._calculate_sharpe_ratio(returns)
        
        assert isinstance(sharpe, float)
    
    def test_calculate_sharpe_ratio_zero_std(self, risk_engine):
        """Test Sharpe ratio calculation with zero standard deviation."""
        returns = [0.01, 0.01, 0.01, 0.01, 0.01]  # Constant returns
        sharpe = risk_engine._calculate_sharpe_ratio(returns)
        
        assert sharpe == 0.0
    
    def test_calculate_max_drawdown(self, risk_engine):
        """Test maximum drawdown calculation."""
        returns = [0.05, -0.02, -0.03, 0.04, -0.01, 0.02, -0.05, 0.03, 0.01, -0.02]
        max_dd = risk_engine._calculate_max_drawdown(returns)
        
        assert isinstance(max_dd, float)
        assert max_dd >= 0
    
    def test_calculate_correlation_matrix(self, risk_engine, sample_historical_returns):
        """Test correlation matrix calculation."""
        symbols = ["AAPL", "GOOGL"]
        corr_matrix = risk_engine._calculate_correlation_matrix(symbols, sample_historical_returns)
        
        assert isinstance(corr_matrix, dict)
        assert "AAPL" in corr_matrix
        assert "GOOGL" in corr_matrix
        assert corr_matrix["AAPL"]["AAPL"] == 1.0
        assert corr_matrix["GOOGL"]["GOOGL"] == 1.0
        assert -1 <= corr_matrix["AAPL"]["GOOGL"] <= 1
    
    def test_calculate_diversification_ratio(self, risk_engine, sample_portfolio, 
                                           sample_historical_returns):
        """Test diversification ratio calculation."""
        div_ratio = risk_engine._calculate_diversification_ratio(
            sample_portfolio, sample_historical_returns
        )
        
        assert isinstance(div_ratio, float)
        assert div_ratio >= 0
    
    def test_calculate_portfolio_returns(self, risk_engine, sample_portfolio, 
                                       sample_historical_returns):
        """Test portfolio returns calculation."""
        portfolio_returns = risk_engine._calculate_portfolio_returns(
            sample_portfolio, sample_historical_returns
        )
        
        assert isinstance(portfolio_returns, list)
        assert len(portfolio_returns) > 0
        assert all(isinstance(ret, float) for ret in portfolio_returns)
    
    def test_error_handling_position_size(self, risk_engine, sample_portfolio, 
                                        sample_risk_params):
        """Test error handling in position size calculation."""
        # Create invalid proposal
        invalid_proposal = Mock()
        invalid_proposal.symbol = "TEST"  # Valid symbol
        invalid_proposal.price_target = "invalid"  # Invalid price type
        
        position_size = risk_engine.calculate_position_size(
            invalid_proposal, sample_portfolio, sample_risk_params, {}
        )
        
        assert position_size.recommended_quantity == 0
        assert "Error in calculation" in position_size.rationale
    
    def test_error_handling_risk_metrics(self, risk_engine):
        """Test error handling in risk metrics calculation."""
        # Create invalid portfolio
        invalid_portfolio = Mock()
        invalid_portfolio.positions = None  # Invalid positions
        
        risk_metrics = risk_engine.calculate_risk_metrics(invalid_portfolio, {})
        
        assert risk_metrics.var_95 == 0.0
        assert risk_metrics.cvar_95 == 0.0
        assert risk_metrics.sharpe_ratio == 0.0
        assert risk_metrics.max_drawdown == 0.0
        assert risk_metrics.volatility == 0.0
    
    def test_caching_volatility(self, risk_engine):
        """Test volatility calculation caching."""
        prices = [100.0, 102.0, 98.0, 105.0, 103.0, 107.0]
        
        # First calculation
        vol1 = risk_engine._calculate_volatility("TEST", prices)
        
        # Second calculation should use cache
        vol2 = risk_engine._calculate_volatility("TEST", prices)
        
        assert vol1 == vol2
        assert f"TEST_{len(prices)}" in risk_engine._volatility_cache
    
    def test_calculate_stop_loss_take_profit(self, risk_engine, sample_proposal, 
                                           sample_portfolio, sample_historical_prices):
        """Test stop-loss and take-profit calculation."""
        risk_params = risk_engine.calculate_stop_loss_take_profit(
            sample_proposal, sample_portfolio, sample_historical_prices
        )
        
        assert isinstance(risk_params.stop_loss_pct, float)
        assert isinstance(risk_params.take_profit_pct, float)
        assert isinstance(risk_params.max_position_size_pct, float)
        assert isinstance(risk_params.max_portfolio_risk_pct, float)
        assert isinstance(risk_params.volatility_adjustment, float)
        
        assert 0 < risk_params.stop_loss_pct <= 0.1
        assert 0 < risk_params.take_profit_pct <= 0.2
        assert 0.01 <= risk_params.max_position_size_pct <= 0.2
        assert 0.005 <= risk_params.max_portfolio_risk_pct <= 0.05
    
    def test_adjust_risk_for_market_volatility_low_vol(self, risk_engine, sample_risk_params):
        """Test risk adjustment for low volatility market."""
        adjusted_params = risk_engine.adjust_risk_for_market_volatility(
            sample_risk_params, market_volatility=0.10  # Low volatility
        )
        
        # Should allow more risk in low volatility
        assert adjusted_params.stop_loss_pct >= sample_risk_params.stop_loss_pct
        assert adjusted_params.take_profit_pct >= sample_risk_params.take_profit_pct
    
    def test_adjust_risk_for_market_volatility_high_vol(self, risk_engine, sample_risk_params):
        """Test risk adjustment for high volatility market."""
        adjusted_params = risk_engine.adjust_risk_for_market_volatility(
            sample_risk_params, market_volatility=0.40  # High volatility
        )
        
        # Should reduce stop-loss and take-profit in high volatility (more conservative)
        assert adjusted_params.stop_loss_pct <= sample_risk_params.stop_loss_pct
        assert adjusted_params.take_profit_pct <= sample_risk_params.take_profit_pct
    
    def test_adjust_risk_with_vix(self, risk_engine, sample_risk_params):
        """Test risk adjustment with VIX level."""
        # High VIX (fear)
        high_vix_params = risk_engine.adjust_risk_for_market_volatility(
            sample_risk_params, market_volatility=0.20, vix_level=35.0
        )
        
        # Low VIX (complacency)
        low_vix_params = risk_engine.adjust_risk_for_market_volatility(
            sample_risk_params, market_volatility=0.20, vix_level=12.0
        )
        
        # High VIX should be more conservative (smaller stop-loss and take-profit)
        assert high_vix_params.stop_loss_pct < low_vix_params.stop_loss_pct
        assert high_vix_params.take_profit_pct < low_vix_params.take_profit_pct
    
    def test_enforce_risk_limits_pass(self, risk_engine, sample_proposal, 
                                    sample_portfolio, sample_risk_params):
        """Test risk limit enforcement with valid proposal."""
        # Modify proposal to be within limits
        sample_proposal.quantity = 10  # Small quantity
        
        is_allowed, violations = risk_engine.enforce_risk_limits(
            sample_proposal, sample_portfolio, sample_risk_params
        )
        
        assert isinstance(is_allowed, bool)
        assert isinstance(violations, list)
    
    def test_enforce_risk_limits_violation(self, risk_engine, sample_portfolio, sample_risk_params):
        """Test risk limit enforcement with violating proposal."""
        # Create a large proposal that should violate limits
        large_proposal = TradingProposal(
            proposal_id="large_proposal",
            symbol="MSFT",
            action=TradeAction.BUY,
            quantity=10000,  # Very large quantity
            price_target=300.0,
            rationale="Large test proposal",
            confidence=0.8,
            risk_metrics=RiskMetrics(0.05, 0.07, 1.2, 0.1, 0.2),
            timestamp=datetime.utcnow()
        )
        
        is_allowed, violations = risk_engine.enforce_risk_limits(
            large_proposal, sample_portfolio, sample_risk_params
        )
        
        assert not is_allowed
        assert len(violations) > 0
        assert any("Position size" in violation for violation in violations)
    
    def test_get_risk_alerts(self, risk_engine, sample_portfolio, sample_risk_params):
        """Test risk alert generation and retrieval."""
        # Create a violating proposal to generate alerts
        large_proposal = TradingProposal(
            proposal_id="alert_test",
            symbol="MSFT",
            action=TradeAction.BUY,
            quantity=10000,
            price_target=300.0,
            rationale="Alert test proposal",
            confidence=0.8,
            risk_metrics=RiskMetrics(0.05, 0.07, 1.2, 0.1, 0.2),
            timestamp=datetime.utcnow()
        )
        
        # This should generate alerts
        risk_engine.enforce_risk_limits(large_proposal, sample_portfolio, sample_risk_params)
        
        alerts = risk_engine.get_risk_alerts()
        assert len(alerts) > 0
        assert alerts[0]["proposal_id"] == "alert_test"
        assert "violations" in alerts[0]
    
    def test_clear_risk_alerts(self, risk_engine):
        """Test clearing risk alerts."""
        # Add some alerts first (from previous test)
        initial_count = len(risk_engine.get_risk_alerts())
        
        # Clear all alerts
        risk_engine.clear_risk_alerts()
        
        alerts_after_clear = risk_engine.get_risk_alerts()
        assert len(alerts_after_clear) == 0
    
    def test_calculate_current_portfolio_risk(self, risk_engine, sample_portfolio, 
                                            sample_historical_prices):
        """Test current portfolio risk calculation."""
        risk_level = risk_engine._calculate_current_portfolio_risk(
            sample_portfolio, sample_historical_prices
        )
        
        assert isinstance(risk_level, float)
        assert risk_level >= 0
    
    def test_calculate_symbol_exposure(self, risk_engine, sample_portfolio):
        """Test symbol exposure calculation."""
        # Test existing symbol
        aapl_exposure = risk_engine._calculate_symbol_exposure("AAPL", sample_portfolio)
        assert aapl_exposure == 0.5  # From fixture
        
        # Test non-existing symbol
        new_exposure = risk_engine._calculate_symbol_exposure("TSLA", sample_portfolio)
        assert new_exposure == 0.0
    
    def test_error_handling_risk_parameters(self, risk_engine, sample_portfolio):
        """Test error handling in risk parameter calculation."""
        # Create invalid proposal
        invalid_proposal = Mock()
        invalid_proposal.symbol = "TEST"
        invalid_proposal.price_target = None  # Invalid price
        
        risk_params = risk_engine.calculate_stop_loss_take_profit(
            invalid_proposal, sample_portfolio, {}
        )
        
        # Should return default parameters
        assert risk_params.stop_loss_pct == 0.02
        assert risk_params.take_profit_pct == 0.05


if __name__ == "__main__":
    pytest.main([__file__])
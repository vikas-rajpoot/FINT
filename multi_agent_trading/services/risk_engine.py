"""
Risk calculation engine for portfolio risk management.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging

from ..models.trading_models import (
    Portfolio, Position, RiskMetrics, RiskParameters, 
    PositionSize, ExposureReport, TradingProposal
)


class RiskCalculationEngine:
    """
    Risk calculation engine providing comprehensive risk analysis capabilities.
    
    Features:
    - Position sizing based on portfolio risk
    - VaR and CVaR calculations
    - Sharpe ratio and risk-adjusted returns
    - Portfolio exposure monitoring
    - Correlation analysis
    - Dynamic risk parameter setting
    - Risk limit enforcement
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize the risk calculation engine.
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculations
        """
        self.risk_free_rate = risk_free_rate
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Cache for historical data and calculations
        self._price_cache: Dict[str, pd.DataFrame] = {}
        self._correlation_cache: Dict[str, np.ndarray] = {}
        self._volatility_cache: Dict[str, float] = {}
        
        # Risk limit tracking
        self._risk_alerts: List[Dict[str, Any]] = []
        self._risk_limits_breached: Dict[str, bool] = {}
    
    def calculate_position_size(
        self, 
        proposal: TradingProposal, 
        portfolio: Portfolio, 
        risk_params: RiskParameters,
        historical_prices: Dict[str, List[float]]
    ) -> PositionSize:
        """
        Calculate optimal position size based on portfolio risk.
        
        Args:
            proposal: Trading proposal to size
            portfolio: Current portfolio state
            risk_params: Risk management parameters
            historical_prices: Historical price data for volatility calculation
            
        Returns:
            Position sizing recommendation
        """
        try:
            symbol = proposal.symbol
            entry_price = proposal.price_target
            
            # Calculate volatility
            volatility = self._calculate_volatility(symbol, historical_prices.get(symbol, []))
            
            # Calculate maximum risk amount
            max_risk_amount = portfolio.total_value * risk_params.max_portfolio_risk_pct
            
            # Calculate stop loss distance
            stop_loss_distance = entry_price * risk_params.stop_loss_pct
            
            # Position size based on risk amount
            risk_based_quantity = int(max_risk_amount / stop_loss_distance) if stop_loss_distance > 0 else 0
            
            # Position size based on portfolio percentage
            max_position_value = portfolio.total_value * risk_params.max_position_size_pct
            max_quantity_by_value = int(max_position_value / entry_price) if entry_price > 0 else 0
            
            # Volatility adjustment
            volatility_adjustment = min(1.0, risk_params.volatility_adjustment / volatility) if volatility > 0 else 1.0
            adjusted_quantity = int(risk_based_quantity * volatility_adjustment)
            
            # Take the minimum of all constraints
            recommended_quantity = min(adjusted_quantity, max_quantity_by_value)
            max_quantity = max_quantity_by_value
            
            # Calculate risk metrics
            position_value = recommended_quantity * entry_price
            risk_amount = recommended_quantity * stop_loss_distance
            risk_pct = risk_amount / portfolio.total_value if portfolio.total_value > 0 else 0
            
            rationale = (
                f"Position sized based on {risk_params.max_portfolio_risk_pct:.1%} portfolio risk, "
                f"{risk_params.max_position_size_pct:.1%} max position size, "
                f"volatility adjustment: {volatility_adjustment:.2f}"
            )
            
            return PositionSize(
                symbol=symbol,
                recommended_quantity=recommended_quantity,
                max_quantity=max_quantity,
                risk_amount=risk_amount,
                position_value=position_value,
                risk_pct=risk_pct,
                rationale=rationale
            )
            
        except Exception as e:
            symbol = getattr(proposal, 'symbol', 'UNKNOWN')
            self.logger.error(f"Error calculating position size for {symbol}: {str(e)}")
            return PositionSize(
                symbol=symbol if isinstance(symbol, str) and symbol else "UNKNOWN",
                recommended_quantity=0,
                max_quantity=0,
                risk_amount=0.0,
                position_value=0.0,
                risk_pct=0.0,
                rationale=f"Error in calculation: {str(e)}"
            )
    
    def calculate_risk_metrics(
        self, 
        portfolio: Portfolio, 
        historical_returns: Dict[str, List[float]],
        confidence_level: float = 0.95
    ) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics for the portfolio.
        
        Args:
            portfolio: Current portfolio state
            historical_returns: Historical returns for each position
            confidence_level: Confidence level for VaR calculation
            
        Returns:
            Risk metrics including VaR, CVaR, Sharpe ratio
        """
        try:
            if not portfolio.positions:
                return RiskMetrics(
                    var_95=0.0,
                    cvar_95=0.0,
                    sharpe_ratio=0.0,
                    max_drawdown=0.0,
                    volatility=0.0
                )
            
            # Calculate portfolio returns
            portfolio_returns = self._calculate_portfolio_returns(portfolio, historical_returns)
            
            if len(portfolio_returns) == 0:
                return RiskMetrics(
                    var_95=0.0,
                    cvar_95=0.0,
                    sharpe_ratio=0.0,
                    max_drawdown=0.0,
                    volatility=0.0
                )
            
            # Calculate VaR
            var_95 = self._calculate_var(portfolio_returns, confidence_level)
            
            # Calculate CVaR (Expected Shortfall)
            cvar_95 = self._calculate_cvar(portfolio_returns, confidence_level)
            
            # Calculate Sharpe ratio
            sharpe_ratio = self._calculate_sharpe_ratio(portfolio_returns)
            
            # Calculate maximum drawdown
            max_drawdown = self._calculate_max_drawdown(portfolio_returns)
            
            # Calculate volatility
            volatility = np.std(portfolio_returns) * np.sqrt(252)  # Annualized
            
            return RiskMetrics(
                var_95=var_95,
                cvar_95=cvar_95,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                volatility=volatility
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {str(e)}")
            return RiskMetrics(
                var_95=0.0,
                cvar_95=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                volatility=0.0
            )
    
    def monitor_portfolio_exposure(
        self, 
        portfolio: Portfolio,
        sector_mapping: Dict[str, str],
        currency_mapping: Dict[str, str],
        historical_returns: Dict[str, List[float]]
    ) -> ExposureReport:
        """
        Monitor portfolio exposure across sectors, currencies, and correlations.
        
        Args:
            portfolio: Current portfolio state
            sector_mapping: Mapping of symbols to sectors
            currency_mapping: Mapping of symbols to currencies
            historical_returns: Historical returns for correlation analysis
            
        Returns:
            Comprehensive exposure analysis report
        """
        try:
            # Calculate total exposure
            total_exposure = sum(pos.market_value for pos in portfolio.positions.values())
            
            # Calculate sector exposure
            sector_exposure = {}
            for symbol, position in portfolio.positions.items():
                sector = sector_mapping.get(symbol, "Unknown")
                sector_exposure[sector] = sector_exposure.get(sector, 0) + position.weight
            
            # Calculate currency exposure
            currency_exposure = {}
            for symbol, position in portfolio.positions.items():
                currency = currency_mapping.get(symbol, "USD")
                currency_exposure[currency] = currency_exposure.get(currency, 0) + position.weight
            
            # Calculate correlation matrix
            correlation_matrix = self._calculate_correlation_matrix(
                list(portfolio.positions.keys()), 
                historical_returns
            )
            
            # Calculate concentration risk (Herfindahl index)
            concentration_risk = sum(pos.weight ** 2 for pos in portfolio.positions.values())
            
            # Calculate diversification ratio
            diversification_ratio = self._calculate_diversification_ratio(
                portfolio, historical_returns
            )
            
            return ExposureReport(
                total_exposure=total_exposure,
                sector_exposure=sector_exposure,
                currency_exposure=currency_exposure,
                correlation_matrix=correlation_matrix,
                concentration_risk=concentration_risk,
                diversification_ratio=diversification_ratio,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            self.logger.error(f"Error monitoring portfolio exposure: {str(e)}")
            return ExposureReport(
                total_exposure=0.0,
                sector_exposure={},
                currency_exposure={},
                correlation_matrix={},
                concentration_risk=0.0,
                diversification_ratio=0.0,
                timestamp=datetime.utcnow()
            )
    
    def _calculate_volatility(self, symbol: str, prices: List[float]) -> float:
        """Calculate annualized volatility from price series."""
        if len(prices) < 2:
            return 0.2  # Default volatility assumption
        
        # Check cache
        cache_key = f"{symbol}_{len(prices)}"
        if cache_key in self._volatility_cache:
            return self._volatility_cache[cache_key]
        
        # Calculate returns
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] > 0:
                returns.append((prices[i] - prices[i-1]) / prices[i-1])
        
        if len(returns) < 2:
            return 0.2
        
        # Annualized volatility
        volatility = np.std(returns) * np.sqrt(252)
        
        # Cache result
        self._volatility_cache[cache_key] = volatility
        
        return volatility
    
    def _calculate_portfolio_returns(
        self, 
        portfolio: Portfolio, 
        historical_returns: Dict[str, List[float]]
    ) -> List[float]:
        """Calculate portfolio returns from position weights and historical returns."""
        if not portfolio.positions or not historical_returns:
            return []
        
        # Find common length
        min_length = min(len(returns) for returns in historical_returns.values() if returns)
        if min_length == 0:
            return []
        
        portfolio_returns = []
        for i in range(min_length):
            portfolio_return = 0.0
            for symbol, position in portfolio.positions.items():
                if symbol in historical_returns and i < len(historical_returns[symbol]):
                    portfolio_return += position.weight * historical_returns[symbol][i]
            portfolio_returns.append(portfolio_return)
        
        return portfolio_returns
    
    def _calculate_var(self, returns: List[float], confidence_level: float) -> float:
        """Calculate Value at Risk using historical simulation."""
        if not returns:
            return 0.0
        
        sorted_returns = sorted(returns)
        index = int((1 - confidence_level) * len(sorted_returns))
        return abs(sorted_returns[index]) if index < len(sorted_returns) else 0.0
    
    def _calculate_cvar(self, returns: List[float], confidence_level: float) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        if not returns:
            return 0.0
        
        var = self._calculate_var(returns, confidence_level)
        tail_returns = [r for r in returns if r <= -var]
        
        return abs(np.mean(tail_returns)) if tail_returns else 0.0
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio."""
        if not returns or len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Annualized Sharpe ratio
        daily_risk_free = self.risk_free_rate / 252
        excess_return = mean_return - daily_risk_free
        
        return (excess_return * np.sqrt(252)) / (std_return * np.sqrt(252))
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown."""
        if not returns:
            return 0.0
        
        cumulative = np.cumprod([1 + r for r in returns])
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        return abs(np.min(drawdown))
    
    def _calculate_correlation_matrix(
        self, 
        symbols: List[str], 
        historical_returns: Dict[str, List[float]]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate correlation matrix between assets."""
        correlation_matrix = {}
        
        for symbol1 in symbols:
            correlation_matrix[symbol1] = {}
            for symbol2 in symbols:
                if symbol1 == symbol2:
                    correlation_matrix[symbol1][symbol2] = 1.0
                elif (symbol1 in historical_returns and symbol2 in historical_returns and
                      historical_returns[symbol1] and historical_returns[symbol2]):
                    
                    returns1 = historical_returns[symbol1]
                    returns2 = historical_returns[symbol2]
                    
                    # Align lengths
                    min_len = min(len(returns1), len(returns2))
                    if min_len > 1:
                        corr = np.corrcoef(returns1[:min_len], returns2[:min_len])[0, 1]
                        correlation_matrix[symbol1][symbol2] = corr if not np.isnan(corr) else 0.0
                    else:
                        correlation_matrix[symbol1][symbol2] = 0.0
                else:
                    correlation_matrix[symbol1][symbol2] = 0.0
        
        return correlation_matrix
    
    def _calculate_diversification_ratio(
        self, 
        portfolio: Portfolio, 
        historical_returns: Dict[str, List[float]]
    ) -> float:
        """Calculate portfolio diversification ratio."""
        if not portfolio.positions or len(portfolio.positions) < 2:
            return 1.0
        
        try:
            # Calculate weighted average volatility
            weighted_vol = 0.0
            for symbol, position in portfolio.positions.items():
                if symbol in historical_returns and historical_returns[symbol]:
                    vol = self._calculate_volatility(symbol, historical_returns[symbol])
                    weighted_vol += position.weight * vol
            
            # Calculate portfolio volatility
            portfolio_returns = self._calculate_portfolio_returns(portfolio, historical_returns)
            if not portfolio_returns:
                return 1.0
            
            portfolio_vol = np.std(portfolio_returns) * np.sqrt(252)
            
            if portfolio_vol == 0:
                return 1.0
            
            return weighted_vol / portfolio_vol
            
        except Exception as e:
            self.logger.error(f"Error calculating diversification ratio: {str(e)}")
            return 1.0
    
    def calculate_stop_loss_take_profit(
        self, 
        proposal: TradingProposal,
        portfolio: Portfolio,
        historical_prices: Dict[str, List[float]],
        base_stop_loss_pct: float = 0.02,
        base_take_profit_pct: float = 0.05
    ) -> RiskParameters:
        """
        Calculate dynamic stop-loss and take-profit levels based on market conditions.
        
        Args:
            proposal: Trading proposal
            portfolio: Current portfolio state
            historical_prices: Historical price data
            base_stop_loss_pct: Base stop-loss percentage
            base_take_profit_pct: Base take-profit percentage
            
        Returns:
            Risk parameters with calculated stop-loss and take-profit levels
        """
        try:
            symbol = proposal.symbol
            
            # Calculate volatility for dynamic adjustment
            volatility = self._calculate_volatility(symbol, historical_prices.get(symbol, []))
            
            # Adjust stop-loss based on volatility
            # Higher volatility = wider stop-loss to avoid premature exits
            volatility_multiplier = max(0.5, min(2.0, volatility / 0.2))  # Normalize around 20% volatility
            adjusted_stop_loss = base_stop_loss_pct * volatility_multiplier
            
            # Adjust take-profit based on volatility and market conditions
            adjusted_take_profit = base_take_profit_pct * volatility_multiplier
            
            # Calculate position size limits
            max_position_size_pct = min(0.2, 0.1 / volatility)  # Smaller positions for higher volatility
            
            # Portfolio risk adjustment based on current exposure
            current_risk = self._calculate_current_portfolio_risk(portfolio, historical_prices)
            max_portfolio_risk_pct = max(0.01, 0.05 - current_risk)  # Reduce risk if portfolio already risky
            
            return RiskParameters(
                stop_loss_pct=min(adjusted_stop_loss, 0.1),  # Cap at 10%
                take_profit_pct=min(adjusted_take_profit, 0.2),  # Cap at 20%
                max_position_size_pct=max(0.01, max_position_size_pct),  # Minimum 1%
                max_portfolio_risk_pct=max(0.005, max_portfolio_risk_pct),  # Minimum 0.5%
                volatility_adjustment=1.0 / volatility_multiplier
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating risk parameters for {proposal.symbol}: {str(e)}")
            return RiskParameters(
                stop_loss_pct=base_stop_loss_pct,
                take_profit_pct=base_take_profit_pct,
                max_position_size_pct=0.1,
                max_portfolio_risk_pct=0.02,
                volatility_adjustment=1.0
            )
    
    def adjust_risk_for_market_volatility(
        self, 
        base_params: RiskParameters,
        market_volatility: float,
        vix_level: Optional[float] = None
    ) -> RiskParameters:
        """
        Dynamically adjust risk parameters based on market volatility.
        
        Args:
            base_params: Base risk parameters
            market_volatility: Current market volatility (annualized)
            vix_level: VIX level if available
            
        Returns:
            Adjusted risk parameters
        """
        try:
            # Volatility regime detection
            low_vol_threshold = 0.15
            high_vol_threshold = 0.30
            
            if market_volatility < low_vol_threshold:
                # Low volatility regime - can take more risk
                vol_adjustment = 1.2
            elif market_volatility > high_vol_threshold:
                # High volatility regime - reduce risk
                vol_adjustment = 0.7
            else:
                # Normal volatility regime
                vol_adjustment = 1.0
            
            # VIX-based adjustment if available
            if vix_level is not None:
                if vix_level > 30:  # High fear
                    vix_adjustment = 0.6
                elif vix_level < 15:  # Low fear
                    vix_adjustment = 1.3
                else:
                    vix_adjustment = 1.0
                
                vol_adjustment *= vix_adjustment
            
            return RiskParameters(
                stop_loss_pct=base_params.stop_loss_pct * vol_adjustment,
                take_profit_pct=base_params.take_profit_pct * vol_adjustment,
                max_position_size_pct=base_params.max_position_size_pct / vol_adjustment,
                max_portfolio_risk_pct=base_params.max_portfolio_risk_pct / vol_adjustment,
                volatility_adjustment=base_params.volatility_adjustment * vol_adjustment
            )
            
        except Exception as e:
            self.logger.error(f"Error adjusting risk for market volatility: {str(e)}")
            return base_params
    
    def enforce_risk_limits(
        self, 
        proposal: TradingProposal,
        portfolio: Portfolio,
        risk_params: RiskParameters,
        max_drawdown_limit: float = 0.15,
        max_var_limit: float = 0.10
    ) -> Tuple[bool, List[str]]:
        """
        Enforce risk limits and generate alerts for violations.
        
        Args:
            proposal: Trading proposal to check
            portfolio: Current portfolio state
            risk_params: Risk parameters
            max_drawdown_limit: Maximum allowed drawdown
            max_var_limit: Maximum allowed VaR
            
        Returns:
            Tuple of (is_allowed, list_of_violations)
        """
        violations = []
        
        try:
            # Check portfolio-level risk limits
            current_risk_metrics = self.calculate_risk_metrics(portfolio, {})
            
            # Check maximum drawdown
            if current_risk_metrics.max_drawdown > max_drawdown_limit:
                violations.append(f"Portfolio drawdown {current_risk_metrics.max_drawdown:.2%} exceeds limit {max_drawdown_limit:.2%}")
            
            # Check VaR limit
            if current_risk_metrics.var_95 > max_var_limit:
                violations.append(f"Portfolio VaR {current_risk_metrics.var_95:.2%} exceeds limit {max_var_limit:.2%}")
            
            # Check position size limits
            position_value = proposal.quantity * proposal.price_target
            position_pct = position_value / portfolio.total_value if portfolio.total_value > 0 else 0
            
            if position_pct > risk_params.max_position_size_pct:
                violations.append(f"Position size {position_pct:.2%} exceeds limit {risk_params.max_position_size_pct:.2%}")
            
            # Check portfolio risk limit
            estimated_risk = position_pct * risk_params.stop_loss_pct
            if estimated_risk > risk_params.max_portfolio_risk_pct:
                violations.append(f"Position risk {estimated_risk:.2%} exceeds portfolio limit {risk_params.max_portfolio_risk_pct:.2%}")
            
            # Check concentration limits
            symbol_exposure = self._calculate_symbol_exposure(proposal.symbol, portfolio)
            if symbol_exposure + position_pct > 0.25:  # Max 25% in single symbol
                violations.append(f"Symbol concentration would exceed 25% limit")
            
            # Generate alerts for violations
            if violations:
                alert = {
                    "timestamp": datetime.utcnow(),
                    "proposal_id": proposal.proposal_id,
                    "symbol": proposal.symbol,
                    "violations": violations,
                    "severity": "HIGH" if len(violations) > 2 else "MEDIUM"
                }
                self._risk_alerts.append(alert)
                self._risk_limits_breached[proposal.symbol] = True
                
                self.logger.warning(f"Risk limit violations for {proposal.symbol}: {violations}")
            
            return len(violations) == 0, violations
            
        except Exception as e:
            self.logger.error(f"Error enforcing risk limits: {str(e)}")
            return False, [f"Error in risk limit enforcement: {str(e)}"]
    
    def get_risk_alerts(self, since: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Get risk alerts generated since a specific time.
        
        Args:
            since: Get alerts since this timestamp (default: all alerts)
            
        Returns:
            List of risk alerts
        """
        if since is None:
            return self._risk_alerts.copy()
        
        return [alert for alert in self._risk_alerts if alert["timestamp"] >= since]
    
    def clear_risk_alerts(self, before: Optional[datetime] = None) -> None:
        """
        Clear risk alerts before a specific time.
        
        Args:
            before: Clear alerts before this timestamp (default: clear all)
        """
        if before is None:
            self._risk_alerts.clear()
            self._risk_limits_breached.clear()
        else:
            self._risk_alerts = [alert for alert in self._risk_alerts if alert["timestamp"] >= before]
    
    def _calculate_current_portfolio_risk(
        self, 
        portfolio: Portfolio, 
        historical_prices: Dict[str, List[float]]
    ) -> float:
        """Calculate current portfolio risk level."""
        try:
            if not portfolio.positions:
                return 0.0
            
            total_risk = 0.0
            for symbol, position in portfolio.positions.items():
                volatility = self._calculate_volatility(symbol, historical_prices.get(symbol, []))
                position_risk = position.weight * volatility
                total_risk += position_risk
            
            return total_risk
            
        except Exception as e:
            self.logger.error(f"Error calculating current portfolio risk: {str(e)}")
            return 0.0
    
    def _calculate_symbol_exposure(self, symbol: str, portfolio: Portfolio) -> float:
        """Calculate current exposure to a specific symbol."""
        if symbol in portfolio.positions:
            return portfolio.positions[symbol].weight
        return 0.0
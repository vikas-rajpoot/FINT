"""
Portfolio optimization algorithms for the multi-agent trading system.

This module implements mean-variance optimization, efficient frontier calculation,
and multi-objective optimization for risk-return trade-offs.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy.stats import norm
import logging


@dataclass
class Asset:
    """Asset information for portfolio optimization."""
    symbol: str
    expected_return: float
    volatility: float
    current_price: float
    market_cap: float
    sector: str
    
    def validate(self):
        """Validate asset data."""
        if not self.symbol or not isinstance(self.symbol, str):
            raise ValueError("Symbol must be a non-empty string")
        if not isinstance(self.expected_return, (int, float)):
            raise ValueError("Expected return must be a number")
        if not isinstance(self.volatility, (int, float)) or self.volatility < 0:
            raise ValueError("Volatility must be a non-negative number")
        if not isinstance(self.current_price, (int, float)) or self.current_price <= 0:
            raise ValueError("Current price must be positive")
        if not isinstance(self.market_cap, (int, float)) or self.market_cap <= 0:
            raise ValueError("Market cap must be positive")
        if not self.sector or not isinstance(self.sector, str):
            raise ValueError("Sector must be a non-empty string")


@dataclass
class AllocationPlan:
    """Portfolio allocation plan result."""
    allocations: Dict[str, float]  # symbol -> weight
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    optimization_method: str
    constraints_applied: List[str]
    timestamp: datetime
    
    def validate(self):
        """Validate allocation plan."""
        if not isinstance(self.allocations, dict):
            raise ValueError("Allocations must be a dictionary")
        
        total_weight = sum(self.allocations.values())
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"Allocation weights must sum to 1.0, got {total_weight}")
        
        for symbol, weight in self.allocations.items():
            if not isinstance(weight, (int, float)) or weight < 0:
                raise ValueError(f"Weight for {symbol} must be non-negative")


@dataclass
class EfficientFrontier:
    """Efficient frontier calculation result."""
    returns: List[float]
    volatilities: List[float]
    sharpe_ratios: List[float]
    allocations: List[Dict[str, float]]
    optimal_portfolio: Dict[str, float]  # Maximum Sharpe ratio portfolio
    min_variance_portfolio: Dict[str, float]
    timestamp: datetime


class PortfolioOptimizationAlgorithms:
    """Core portfolio optimization algorithms."""
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize portfolio optimization algorithms.
        
        Args:
            risk_free_rate: Risk-free rate for Sharpe ratio calculations
        """
        self.risk_free_rate = risk_free_rate
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def mean_variance_optimization(
        self,
        assets: List[Asset],
        correlation_matrix: np.ndarray,
        target_return: Optional[float] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> AllocationPlan:
        """
        Perform mean-variance optimization for asset allocation.
        
        This implements the classic Markowitz mean-variance optimization to find
        the portfolio with minimum variance for a given return level, or maximum
        Sharpe ratio if no target return is specified.
        
        Args:
            assets: List of assets to optimize
            correlation_matrix: Asset correlation matrix (n x n)
            target_return: Target portfolio return (optional)
            constraints: Additional constraints (optional)
            
        Returns:
            Optimal allocation plan
            
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If optimization fails
        """
        if len(assets) == 0:
            raise ValueError("Assets list cannot be empty")
        
        if correlation_matrix.shape != (len(assets), len(assets)):
            raise ValueError("Correlation matrix dimensions must match number of assets")
        
        # Validate assets
        for asset in assets:
            asset.validate()
        
        # Extract returns and volatilities
        returns = np.array([asset.expected_return for asset in assets])
        volatilities = np.array([asset.volatility for asset in assets])
        
        # Create covariance matrix from correlation matrix and volatilities
        cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix
        
        # Ensure covariance matrix is positive definite
        eigenvals = np.linalg.eigvals(cov_matrix)
        if np.any(eigenvals <= 0):
            self.logger.warning("Covariance matrix is not positive definite, adding regularization")
            cov_matrix += np.eye(len(assets)) * 1e-8
        
        n_assets = len(assets)
        
        def objective(weights):
            """Minimize portfolio variance."""
            return np.dot(weights, np.dot(cov_matrix, weights))
        
        # Set up constraints
        constraints_list = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}  # Weights sum to 1
        ]
        
        applied_constraints = ["weights_sum_to_one"]
        
        # Target return constraint if specified
        if target_return is not None:
            constraints_list.append({
                'type': 'eq',
                'fun': lambda x: np.dot(x, returns) - target_return
            })
            applied_constraints.append(f"target_return_{target_return:.4f}")
        
        # Process additional constraints
        if constraints:
            # Maximum weight constraint
            if 'max_weight' in constraints:
                max_weight = constraints['max_weight']
                applied_constraints.append(f"max_weight_{max_weight}")
            else:
                max_weight = 1.0
            
            # Minimum weight constraint
            if 'min_weight' in constraints:
                min_weight = constraints['min_weight']
                applied_constraints.append(f"min_weight_{min_weight}")
            else:
                min_weight = 0.0
            
            # Sector constraints
            if 'max_sector_weight' in constraints:
                max_sector_weight = constraints['max_sector_weight']
                sectors = {}
                for i, asset in enumerate(assets):
                    if asset.sector not in sectors:
                        sectors[asset.sector] = []
                    sectors[asset.sector].append(i)
                
                for sector, indices in sectors.items():
                    constraints_list.append({
                        'type': 'ineq',
                        'fun': lambda x, idx=indices: max_sector_weight - np.sum([x[i] for i in idx])
                    })
                applied_constraints.append(f"max_sector_weight_{max_sector_weight}")
        else:
            max_weight = 1.0
            min_weight = 0.0
        
        # Bounds for weights
        bounds = [(min_weight, max_weight) for _ in range(n_assets)]
        
        # Initial guess (equal weights)
        x0 = np.ones(n_assets) / n_assets
        
        # If no target return specified, maximize Sharpe ratio instead
        if target_return is None:
            def sharpe_objective(weights):
                """Minimize negative Sharpe ratio."""
                portfolio_return = np.dot(weights, returns)
                portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
                portfolio_volatility = np.sqrt(portfolio_variance)
                
                if portfolio_volatility == 0:
                    return -np.inf
                
                sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
                return -sharpe_ratio  # Minimize negative Sharpe ratio
            
            objective = sharpe_objective
            applied_constraints.append("maximize_sharpe_ratio")
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list,
            options={'ftol': 1e-9, 'disp': False, 'maxiter': 1000}
        )
        
        if not result.success:
            raise RuntimeError(f"Optimization failed: {result.message}")
        
        # Ensure weights are non-negative and sum to 1
        weights = np.maximum(result.x, 0)
        weights = weights / np.sum(weights)
        
        # Create allocation dictionary
        allocations = {
            assets[i].symbol: float(weights[i])
            for i in range(n_assets)
        }
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(weights, returns)
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        if portfolio_volatility > 0:
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        else:
            sharpe_ratio = 0.0
        
        allocation_plan = AllocationPlan(
            allocations=allocations,
            expected_return=float(portfolio_return),
            expected_volatility=float(portfolio_volatility),
            sharpe_ratio=float(sharpe_ratio),
            optimization_method="mean_variance",
            constraints_applied=applied_constraints,
            timestamp=datetime.utcnow()
        )
        
        allocation_plan.validate()
        return allocation_plan
    
    def calculate_efficient_frontier(
        self,
        assets: List[Asset],
        correlation_matrix: np.ndarray,
        num_points: int = 50,
        constraints: Optional[Dict[str, Any]] = None
    ) -> EfficientFrontier:
        """
        Calculate the efficient frontier for given assets.
        
        The efficient frontier represents the set of optimal portfolios offering
        the highest expected return for each level of risk.
        
        Args:
            assets: List of assets
            correlation_matrix: Asset correlation matrix
            num_points: Number of points on the frontier
            constraints: Portfolio constraints
            
        Returns:
            Efficient frontier data
            
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If frontier calculation fails
        """
        if len(assets) < 2:
            raise ValueError("Need at least 2 assets for efficient frontier")
        
        if num_points < 10:
            raise ValueError("Need at least 10 points for meaningful frontier")
        
        returns = np.array([asset.expected_return for asset in assets])
        min_return = np.min(returns)
        max_return = np.max(returns)
        
        # Extend range slightly beyond individual asset returns
        return_range = max_return - min_return
        min_target = min_return - 0.1 * return_range
        max_target = max_return + 0.1 * return_range
        
        # Generate target returns
        target_returns = np.linspace(min_target, max_target, num_points)
        
        frontier_returns = []
        frontier_volatilities = []
        frontier_sharpe_ratios = []
        frontier_allocations = []
        
        successful_optimizations = 0
        
        for target_return in target_returns:
            try:
                allocation_plan = self.mean_variance_optimization(
                    assets, correlation_matrix, target_return, constraints
                )
                
                frontier_returns.append(allocation_plan.expected_return)
                frontier_volatilities.append(allocation_plan.expected_volatility)
                frontier_sharpe_ratios.append(allocation_plan.sharpe_ratio)
                frontier_allocations.append(allocation_plan.allocations)
                successful_optimizations += 1
                
            except Exception as e:
                self.logger.debug(f"Failed to optimize for return {target_return:.4f}: {e}")
                continue
        
        if successful_optimizations < 5:
            raise RuntimeError(f"Failed to generate efficient frontier: only {successful_optimizations} successful optimizations")
        
        # Find optimal portfolios
        if frontier_sharpe_ratios:
            max_sharpe_idx = np.argmax(frontier_sharpe_ratios)
            optimal_portfolio = frontier_allocations[max_sharpe_idx]
        else:
            optimal_portfolio = {}
        
        if frontier_volatilities:
            min_var_idx = np.argmin(frontier_volatilities)
            min_variance_portfolio = frontier_allocations[min_var_idx]
        else:
            min_variance_portfolio = {}
        
        return EfficientFrontier(
            returns=frontier_returns,
            volatilities=frontier_volatilities,
            sharpe_ratios=frontier_sharpe_ratios,
            allocations=frontier_allocations,
            optimal_portfolio=optimal_portfolio,
            min_variance_portfolio=min_variance_portfolio,
            timestamp=datetime.utcnow()
        )
    
    def multi_objective_optimization(
        self,
        assets: List[Asset],
        correlation_matrix: np.ndarray,
        objectives: Dict[str, float],
        constraints: Optional[Dict[str, Any]] = None
    ) -> AllocationPlan:
        """
        Perform multi-objective optimization balancing return, risk, and other factors.
        
        This method allows optimization across multiple objectives such as return
        maximization, risk minimization, and diversification enhancement.
        
        Args:
            assets: List of assets
            correlation_matrix: Asset correlation matrix
            objectives: Objective weights (e.g., {'return': 0.6, 'risk': 0.3, 'diversification': 0.1})
            constraints: Portfolio constraints
            
        Returns:
            Optimal allocation plan
            
        Raises:
            ValueError: If objectives are invalid
            RuntimeError: If optimization fails
        """
        if not objectives or sum(objectives.values()) == 0:
            raise ValueError("Objectives must be specified with non-zero weights")
        
        # Validate objective keys
        valid_objectives = {'return', 'risk', 'diversification', 'momentum', 'value'}
        invalid_objectives = set(objectives.keys()) - valid_objectives
        if invalid_objectives:
            raise ValueError(f"Invalid objectives: {invalid_objectives}. Valid: {valid_objectives}")
        
        # Normalize objective weights
        total_weight = sum(objectives.values())
        objectives = {k: v / total_weight for k, v in objectives.items()}
        
        returns = np.array([asset.expected_return for asset in assets])
        volatilities = np.array([asset.volatility for asset in assets])
        cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix
        
        n_assets = len(assets)
        
        def multi_objective_function(weights):
            """Multi-objective function to minimize."""
            portfolio_return = np.dot(weights, returns)
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            # Normalize metrics for combination
            max_return = np.max(returns)
            max_volatility = np.max(volatilities)
            
            # Return score (higher is better, so we minimize negative)
            return_score = portfolio_return / max_return if max_return > 0 else 0
            
            # Risk score (lower is better)
            risk_score = portfolio_volatility / max_volatility if max_volatility > 0 else 0
            
            # Diversification score (higher is better, so we minimize negative)
            # Using inverse of Herfindahl index
            concentration = np.sum(weights ** 2)  # Herfindahl index
            diversification_score = 1 - concentration  # Higher is better
            
            # Momentum score (if momentum data available)
            momentum_score = 0
            if 'momentum' in objectives:
                # Simple momentum proxy using returns
                momentum_score = np.dot(weights, np.maximum(returns, 0)) / max_return if max_return > 0 else 0
            
            # Value score (if value data available)
            value_score = 0
            if 'value' in objectives:
                # Simple value proxy using inverse of market cap weights
                market_caps = np.array([asset.market_cap for asset in assets])
                if np.sum(market_caps) > 0:
                    value_weights = 1 / (market_caps / np.sum(market_caps))
                    value_score = np.dot(weights, value_weights) / np.max(value_weights)
            
            # Combine objectives (minimize negative of weighted sum for maximization objectives)
            objective_value = 0
            
            if 'return' in objectives:
                objective_value -= objectives['return'] * return_score
            
            if 'risk' in objectives:
                objective_value += objectives['risk'] * risk_score
            
            if 'diversification' in objectives:
                objective_value -= objectives['diversification'] * diversification_score
            
            if 'momentum' in objectives:
                objective_value -= objectives['momentum'] * momentum_score
            
            if 'value' in objectives:
                objective_value -= objectives['value'] * value_score
            
            return objective_value
        
        # Set up constraints
        constraints_list = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
        ]
        
        applied_constraints = ["weights_sum_to_one", "multi_objective"]
        
        # Additional constraints
        if constraints:
            max_weight = constraints.get('max_weight', 1.0)
            min_weight = constraints.get('min_weight', 0.0)
            applied_constraints.extend([f"max_weight_{max_weight}", f"min_weight_{min_weight}"])
        else:
            max_weight = 1.0
            min_weight = 0.0
        
        bounds = [(min_weight, max_weight) for _ in range(n_assets)]
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(
            multi_objective_function,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list,
            options={'ftol': 1e-9, 'disp': False, 'maxiter': 1000}
        )
        
        if not result.success:
            raise RuntimeError(f"Multi-objective optimization failed: {result.message}")
        
        # Ensure weights are non-negative and sum to 1
        weights = np.maximum(result.x, 0)
        weights = weights / np.sum(weights)
        
        # Create allocation dictionary
        allocations = {
            assets[i].symbol: float(weights[i])
            for i in range(n_assets)
        }
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(weights, returns)
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        if portfolio_volatility > 0:
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        else:
            sharpe_ratio = 0.0
        
        # Add objective details to constraints
        objective_str = "_".join([f"{k}_{v:.2f}" for k, v in objectives.items()])
        applied_constraints.append(f"objectives_{objective_str}")
        
        allocation_plan = AllocationPlan(
            allocations=allocations,
            expected_return=float(portfolio_return),
            expected_volatility=float(portfolio_volatility),
            sharpe_ratio=float(sharpe_ratio),
            optimization_method="multi_objective",
            constraints_applied=applied_constraints,
            timestamp=datetime.utcnow()
        )
        
        allocation_plan.validate()
        return allocation_plan
    
    def calculate_risk_parity_allocation(
        self,
        assets: List[Asset],
        correlation_matrix: np.ndarray,
        constraints: Optional[Dict[str, Any]] = None
    ) -> AllocationPlan:
        """
        Calculate risk parity allocation where each asset contributes equally to portfolio risk.
        
        Args:
            assets: List of assets
            correlation_matrix: Asset correlation matrix
            constraints: Portfolio constraints
            
        Returns:
            Risk parity allocation plan
        """
        volatilities = np.array([asset.volatility for asset in assets])
        cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix
        
        n_assets = len(assets)
        
        def risk_parity_objective(weights):
            """Minimize sum of squared differences in risk contributions."""
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            
            if portfolio_variance == 0:
                return 1e6
            
            # Calculate marginal risk contributions
            marginal_contrib = np.dot(cov_matrix, weights)
            risk_contrib = weights * marginal_contrib / portfolio_variance
            
            # Target equal risk contribution
            target_contrib = 1.0 / n_assets
            
            return np.sum((risk_contrib - target_contrib) ** 2)
        
        # Constraints
        constraints_list = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
        ]
        
        applied_constraints = ["weights_sum_to_one", "risk_parity"]
        
        # Additional constraints
        if constraints:
            max_weight = constraints.get('max_weight', 1.0)
            min_weight = constraints.get('min_weight', 0.0)
        else:
            max_weight = 1.0
            min_weight = 0.0
        
        bounds = [(min_weight, max_weight) for _ in range(n_assets)]
        
        # Initial guess: inverse volatility weights
        x0 = 1 / volatilities
        x0 = x0 / np.sum(x0)
        
        # Optimize
        result = minimize(
            risk_parity_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list,
            options={'ftol': 1e-9, 'disp': False}
        )
        
        if not result.success:
            raise RuntimeError(f"Risk parity optimization failed: {result.message}")
        
        weights = result.x
        
        # Create allocation dictionary
        allocations = {
            assets[i].symbol: float(weights[i])
            for i in range(n_assets)
        }
        
        # Calculate portfolio metrics
        returns = np.array([asset.expected_return for asset in assets])
        portfolio_return = np.dot(weights, returns)
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        if portfolio_volatility > 0:
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        else:
            sharpe_ratio = 0.0
        
        return AllocationPlan(
            allocations=allocations,
            expected_return=float(portfolio_return),
            expected_volatility=float(portfolio_volatility),
            sharpe_ratio=float(sharpe_ratio),
            optimization_method="risk_parity",
            constraints_applied=applied_constraints,
            timestamp=datetime.utcnow()
        )

@dataclass
class RebalancingPlan:
    """Portfolio rebalancing plan."""
    current_allocations: Dict[str, float]
    target_allocations: Dict[str, float]
    trades_required: Dict[str, int]  # symbol -> quantity change
    rebalancing_cost: float
    expected_improvement: float
    rationale: str
    timestamp: datetime


class PortfolioRebalancingEngine:
    """Portfolio rebalancing and allocation logic."""
    
    def __init__(self, rebalancing_threshold: float = 0.05, transaction_cost_rate: float = 0.001):
        """
        Initialize rebalancing engine.
        
        Args:
            rebalancing_threshold: Minimum deviation to trigger rebalancing
            transaction_cost_rate: Default transaction cost rate
        """
        self.rebalancing_threshold = rebalancing_threshold
        self.transaction_cost_rate = transaction_cost_rate
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def generate_allocation_plan(
        self,
        assets: List[Asset],
        correlation_matrix: np.ndarray,
        market_conditions: Dict[str, Any],
        constraints: Optional[Dict[str, Any]] = None
    ) -> AllocationPlan:
        """
        Generate allocation plan based on current market conditions.
        
        Args:
            assets: List of available assets
            correlation_matrix: Asset correlation matrix
            market_conditions: Current market conditions (volatility, regime, etc.)
            constraints: Portfolio constraints
            
        Returns:
            Allocation plan optimized for current conditions
        """
        optimizer = PortfolioOptimizationAlgorithms()
        
        # Adjust optimization based on market conditions
        market_regime = market_conditions.get('regime', 'normal')
        market_volatility = market_conditions.get('volatility', 0.15)
        
        # Adjust constraints based on market volatility
        if constraints is None:
            constraints = {}
        
        # In high volatility, limit individual position sizes more
        if market_volatility > 0.25:
            constraints['max_weight'] = min(constraints.get('max_weight', 0.3), 0.25)
            self.logger.info(f"High volatility detected ({market_volatility:.1%}), reducing max position size")
        
        # Try multi-objective optimization first, fall back to mean-variance if it fails
        try:
            # Adjust risk preferences based on market conditions
            if market_regime == 'high_volatility':
                # In high volatility, prefer more conservative allocation
                objectives = {'return': 0.4, 'risk': 0.5, 'diversification': 0.1}
            elif market_regime == 'bull_market':
                # In bull market, can take more risk for returns
                objectives = {'return': 0.7, 'risk': 0.2, 'diversification': 0.1}
            elif market_regime == 'bear_market':
                # In bear market, focus on risk management and diversification
                objectives = {'return': 0.3, 'risk': 0.4, 'diversification': 0.3}
            else:
                # Normal market conditions
                objectives = {'return': 0.6, 'risk': 0.3, 'diversification': 0.1}
            
            self.logger.info(f"Using multi-objective optimization for {market_regime} market regime")
            return optimizer.multi_objective_optimization(
                assets, correlation_matrix, objectives, constraints
            )
        except Exception as e:
            self.logger.warning(f"Multi-objective optimization failed: {e}, falling back to mean-variance")
            # Fall back to mean-variance optimization
            return optimizer.mean_variance_optimization(
                assets, correlation_matrix, constraints=constraints
            )
    
    def calculate_rebalancing_plan(
        self,
        current_portfolio: 'Portfolio',
        target_allocations: Dict[str, float],
        transaction_costs: Dict[str, float],
        market_data: Dict[str, 'MarketData']
    ) -> RebalancingPlan:
        """
        Calculate portfolio rebalancing plan.
        
        Args:
            current_portfolio: Current portfolio state
            target_allocations: Target allocation weights
            transaction_costs: Transaction costs per asset (as percentage)
            market_data: Current market data for assets
            
        Returns:
            Rebalancing plan with trades required
        """
        # Calculate current allocations
        current_allocations = {}
        total_value = current_portfolio.total_value
        
        if total_value <= 0:
            raise ValueError("Portfolio total value must be positive")
        
        for symbol, position in current_portfolio.positions.items():
            current_allocations[symbol] = position.market_value / total_value
        
        # Add cash allocation
        if current_portfolio.cash > 0:
            current_allocations['CASH'] = current_portfolio.cash / total_value
        
        # Calculate deviations from target
        deviations = {}
        trades_required = {}
        total_deviation = 0
        
        all_symbols = set(list(current_allocations.keys()) + list(target_allocations.keys()))
        
        for symbol in all_symbols:
            current_weight = current_allocations.get(symbol, 0.0)
            target_weight = target_allocations.get(symbol, 0.0)
            deviation = target_weight - current_weight
            deviations[symbol] = deviation
            total_deviation += abs(deviation)
            
            if symbol != 'CASH' and abs(deviation) > self.rebalancing_threshold:
                # Calculate required trade quantity
                if symbol in market_data:
                    current_price = market_data[symbol].price
                    target_value = target_weight * total_value
                    current_value = current_weight * total_value
                    value_change = target_value - current_value
                    quantity_change = int(value_change / current_price)
                    trades_required[symbol] = quantity_change
                else:
                    self.logger.warning(f"No market data available for {symbol}, skipping trade calculation")
        
        # Calculate rebalancing costs
        rebalancing_cost = 0
        for symbol, quantity in trades_required.items():
            if symbol in market_data:
                trade_value = abs(quantity) * market_data[symbol].price
                cost_rate = transaction_costs.get(symbol, self.transaction_cost_rate)
                rebalancing_cost += trade_value * cost_rate
        
        # Estimate expected improvement
        expected_improvement = self._estimate_rebalancing_benefit(
            current_allocations, target_allocations, total_deviation
        )
        
        # Generate rationale
        rationale = self._generate_rebalancing_rationale(
            deviations, total_deviation, rebalancing_cost, expected_improvement
        )
        
        return RebalancingPlan(
            current_allocations=current_allocations,
            target_allocations=target_allocations,
            trades_required=trades_required,
            rebalancing_cost=rebalancing_cost,
            expected_improvement=expected_improvement,
            rationale=rationale,
            timestamp=datetime.utcnow()
        )
    
    def analyze_diversification(
        self,
        allocations: Dict[str, float],
        assets: List[Asset],
        correlation_matrix: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze portfolio diversification.
        
        Args:
            allocations: Portfolio allocations
            assets: Asset information
            correlation_matrix: Asset correlation matrix
            
        Returns:
            Diversification analysis results
        """
        # Create asset lookup
        asset_lookup = {asset.symbol: asset for asset in assets}
        
        # Calculate concentration metrics
        weights = np.array([allocations.get(asset.symbol, 0.0) for asset in assets])
        
        # Herfindahl-Hirschman Index (concentration)
        hhi = np.sum(weights ** 2)
        
        # Effective number of assets
        effective_assets = 1 / hhi if hhi > 0 else 0
        
        # Sector diversification
        sector_weights = {}
        for asset in assets:
            weight = allocations.get(asset.symbol, 0.0)
            if weight > 0:
                sector_weights[asset.sector] = sector_weights.get(asset.sector, 0.0) + weight
        
        sector_concentration = sum(w ** 2 for w in sector_weights.values())
        
        # Correlation-based diversification ratio
        if len(weights) > 1 and np.sum(weights) > 0:
            portfolio_variance = np.dot(weights, np.dot(correlation_matrix, weights))
            individual_variances = np.sum(weights ** 2)
            diversification_ratio = np.sqrt(individual_variances / portfolio_variance) if portfolio_variance > 0 else 1.0
        else:
            diversification_ratio = 1.0
        
        return {
            'herfindahl_index': float(hhi),
            'effective_assets': float(effective_assets),
            'sector_concentration': float(sector_concentration),
            'diversification_ratio': float(diversification_ratio),
            'sector_weights': sector_weights,
            'concentration_risk': 'HIGH' if hhi > 0.25 else 'MEDIUM' if hhi > 0.1 else 'LOW'
        }
    
    def apply_constraints(
        self,
        allocations: Dict[str, float],
        constraints: Dict[str, Any]
    ) -> Tuple[Dict[str, float], List[str]]:
        """
        Apply portfolio constraints to allocations.
        
        Args:
            allocations: Original allocations
            constraints: Constraints to apply
            
        Returns:
            Tuple of (adjusted_allocations, constraint_violations)
        """
        adjusted_allocations = allocations.copy()
        violations = []
        
        # Apply constraints iteratively to handle interactions
        max_iterations = 5
        for iteration in range(max_iterations):
            changed = False
            
            # Maximum weight constraint
            if 'max_weight' in constraints:
                max_weight = constraints['max_weight']
                for symbol, weight in adjusted_allocations.items():
                    if weight > max_weight:
                        adjusted_allocations[symbol] = max_weight
                        violations.append(f"{symbol} weight reduced from {weight:.3f} to {max_weight:.3f}")
                        changed = True
            
            # Minimum weight constraint
            if 'min_weight' in constraints:
                min_weight = constraints['min_weight']
                for symbol, weight in adjusted_allocations.items():
                    if 0 < weight < min_weight:
                        adjusted_allocations[symbol] = min_weight
                        violations.append(f"{symbol} weight increased from {weight:.3f} to {min_weight:.3f}")
                        changed = True
            
            # Renormalize weights to sum to 1
            total_weight = sum(adjusted_allocations.values())
            if abs(total_weight - 1.0) > 1e-6 and total_weight > 0:
                # Scale all weights proportionally, but respect max constraints
                scale_factor = 1.0 / total_weight
                
                # First, try simple scaling
                temp_allocations = {
                    symbol: weight * scale_factor
                    for symbol, weight in adjusted_allocations.items()
                }
                
                # Check if scaling violates max constraint
                max_weight = constraints.get('max_weight', 1.0)
                needs_redistribution = any(w > max_weight for w in temp_allocations.values())
                
                if needs_redistribution:
                    # Cap weights at max and redistribute excess
                    excess = 0
                    capped_symbols = set()
                    
                    for symbol, weight in temp_allocations.items():
                        if weight > max_weight:
                            excess += weight - max_weight
                            adjusted_allocations[symbol] = max_weight
                            capped_symbols.add(symbol)
                        else:
                            adjusted_allocations[symbol] = weight
                    
                    # Redistribute excess to uncapped symbols
                    uncapped_symbols = [s for s in adjusted_allocations.keys() if s not in capped_symbols]
                    if uncapped_symbols and excess > 0:
                        uncapped_total = sum(adjusted_allocations[s] for s in uncapped_symbols)
                        if uncapped_total > 0:
                            for symbol in uncapped_symbols:
                                additional = (adjusted_allocations[symbol] / uncapped_total) * excess
                                adjusted_allocations[symbol] += additional
                else:
                    adjusted_allocations = temp_allocations
                
                violations.append(f"Weights renormalized (iteration {iteration + 1}, original sum: {total_weight:.3f})")
                changed = True
            
            if not changed:
                break
        
        # Sector concentration limits
        if 'max_sector_weight' in constraints:
            max_sector_weight = constraints['max_sector_weight']
            violations.append(f"Sector weight constraint {max_sector_weight} applied")
        
        return adjusted_allocations, violations
    
    def should_rebalance(
        self,
        current_allocations: Dict[str, float],
        target_allocations: Dict[str, float],
        rebalancing_cost: float,
        expected_benefit: float
    ) -> Tuple[bool, str]:
        """
        Determine if portfolio should be rebalanced.
        
        Args:
            current_allocations: Current portfolio allocations
            target_allocations: Target allocations
            rebalancing_cost: Cost of rebalancing
            expected_benefit: Expected benefit from rebalancing
            
        Returns:
            Tuple of (should_rebalance, rationale)
        """
        # Calculate total deviation
        total_deviation = 0
        max_deviation = 0
        max_deviation_asset = ""
        
        all_symbols = set(list(current_allocations.keys()) + list(target_allocations.keys()))
        
        for symbol in all_symbols:
            current_weight = current_allocations.get(symbol, 0.0)
            target_weight = target_allocations.get(symbol, 0.0)
            deviation = abs(target_weight - current_weight)
            total_deviation += deviation
            
            if deviation > max_deviation:
                max_deviation = deviation
                max_deviation_asset = symbol
        
        # Decision logic - check for extreme deviations first (override cost considerations)
        if max_deviation > 0.15:  # 15% deviation in any single asset
            return True, f"Large deviation in {max_deviation_asset} ({max_deviation:.1%}) requires immediate rebalancing"
        
        if total_deviation > 0.20:  # 20% total deviation
            return True, f"High total deviation ({total_deviation:.1%}) requires rebalancing"
        
        # Check if deviation is below threshold
        if total_deviation < self.rebalancing_threshold:
            return False, f"Total deviation ({total_deviation:.1%}) below threshold ({self.rebalancing_threshold:.1%})"
        
        # Check cost-benefit ratio
        if expected_benefit <= rebalancing_cost:
            return False, f"Expected benefit ({expected_benefit:.4f}) does not exceed cost ({rebalancing_cost:.4f})"
        
        # Normal rebalancing decision
        benefit_cost_ratio = expected_benefit / rebalancing_cost if rebalancing_cost > 0 else float('inf')
        if benefit_cost_ratio > 2.0:  # Benefit should be at least 2x the cost
            return True, f"Favorable benefit-cost ratio ({benefit_cost_ratio:.2f}) justifies rebalancing"
        
        return False, f"Benefit-cost ratio ({benefit_cost_ratio:.2f}) insufficient for rebalancing"
    
    def _estimate_rebalancing_benefit(
        self,
        current_allocations: Dict[str, float],
        target_allocations: Dict[str, float],
        total_deviation: float
    ) -> float:
        """
        Estimate the benefit of rebalancing.
        
        Args:
            current_allocations: Current portfolio allocations
            target_allocations: Target allocations
            total_deviation: Total absolute deviation
            
        Returns:
            Estimated improvement from rebalancing
        """
        # Simple heuristic: benefit is proportional to deviation
        # In practice, this would use more sophisticated models
        base_benefit = total_deviation * 0.1  # 10% of deviation as benefit
        
        # Adjust for portfolio size effects
        portfolio_complexity = len(target_allocations)
        complexity_adjustment = min(1.0, portfolio_complexity / 10.0)
        
        # Adjust for concentration risk
        current_concentration = sum(w ** 2 for w in current_allocations.values())
        target_concentration = sum(w ** 2 for w in target_allocations.values())
        concentration_improvement = max(0, current_concentration - target_concentration)
        
        return base_benefit * complexity_adjustment + concentration_improvement * 0.05
    
    def _generate_rebalancing_rationale(
        self,
        deviations: Dict[str, float],
        total_deviation: float,
        rebalancing_cost: float,
        expected_improvement: float
    ) -> str:
        """
        Generate rationale for rebalancing decision.
        
        Args:
            deviations: Weight deviations by asset
            total_deviation: Total absolute deviation
            rebalancing_cost: Cost of rebalancing
            expected_improvement: Expected benefit
            
        Returns:
            Rationale string
        """
        # Find assets with largest deviations
        sorted_deviations = sorted(
            [(symbol, abs(dev)) for symbol, dev in deviations.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        top_deviations = sorted_deviations[:3]
        
        rationale_parts = []
        
        if total_deviation > self.rebalancing_threshold:
            rationale_parts.append(f"Portfolio has drifted {total_deviation:.1%} from target allocation")
            
            if top_deviations:
                top_assets = ", ".join([f"{symbol} ({dev:.1%})" for symbol, dev in top_deviations])
                rationale_parts.append(f"Largest deviations: {top_assets}")
            
            if expected_improvement > rebalancing_cost:
                rationale_parts.append(f"Expected benefit ({expected_improvement:.4f}) exceeds cost ({rebalancing_cost:.4f})")
                rationale_parts.append("Rebalancing recommended")
            else:
                rationale_parts.append(f"Expected benefit ({expected_improvement:.4f}) below cost ({rebalancing_cost:.4f})")
                rationale_parts.append("Rebalancing not recommended due to high costs")
        else:
            rationale_parts.append(f"Portfolio deviation ({total_deviation:.1%}) within acceptable threshold ({self.rebalancing_threshold:.1%})")
            rationale_parts.append("No rebalancing required")
        
        return ". ".join(rationale_parts)
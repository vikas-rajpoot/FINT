"""
Portfolio Optimizer Agent for multi-agent trading system.

This agent implements portfolio optimization algorithms and integrates with the
multi-agent system to provide optimization-based voting for trading proposals.
"""

import asyncio
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
import uuid

from .base_agent import BaseAgent
from .portfolio_optimization_algorithms import (
    PortfolioOptimizationAlgorithms,
    PortfolioRebalancingEngine,
    Asset,
    AllocationPlan,
    RebalancingPlan
)
from ..models.trading_models import (
    TradingProposal, Vote, Portfolio, Position, MarketData, RiskMetrics, TradeAction
)
from ..models.message_models import Message, AgentResponse, MessageType
from ..models.config_models import AgentConfig


class PortfolioOptimizerAgent(BaseAgent):
    """
    Portfolio Optimizer Agent that provides optimization-based trading decisions.
    
    This agent specializes in portfolio optimization, asset allocation, and rebalancing
    decisions. It uses modern portfolio theory and multi-objective optimization to
    make informed trading recommendations.
    """
    
    def __init__(self, agent_id: str, config: AgentConfig):
        """
        Initialize the Portfolio Optimizer Agent.
        
        Args:
            agent_id: Unique identifier for this agent
            config: Agent configuration parameters
        """
        super().__init__(agent_id, config)
        
        # Initialize optimization engines
        self.optimizer = PortfolioOptimizationAlgorithms(
            risk_free_rate=config.parameters.get('risk_free_rate', 0.02)
        )
        self.rebalancing_engine = PortfolioRebalancingEngine(
            rebalancing_threshold=config.parameters.get('rebalancing_threshold', 0.05),
            transaction_cost_rate=config.parameters.get('transaction_cost_rate', 0.001)
        )
        
        # Portfolio state management
        self.current_portfolio: Optional[Portfolio] = None
        self.target_allocations: Dict[str, float] = {}
        self.asset_universe: List[Asset] = []
        self.correlation_matrix: Optional[np.ndarray] = None
        self.market_conditions: Dict[str, Any] = {}
        
        # Performance tracking
        self.optimization_history: List[Dict[str, Any]] = []
        self.rebalancing_history: List[RebalancingPlan] = []
        
        self.logger.info(f"Portfolio Optimizer Agent {agent_id} initialized")
    
    async def process_message(self, message: Message) -> AgentResponse:
        """
        Process incoming messages and route to appropriate handlers.
        
        Args:
            message: Incoming message to process
            
        Returns:
            Agent response
        """
        try:
            if message.message_type == MessageType.MARKET_DATA_UPDATE:
                return await self._handle_market_data_update(message)
            elif message.message_type == MessageType.PORTFOLIO_UPDATE:
                return await self._handle_portfolio_update(message)
            elif message.message_type == MessageType.OPTIMIZATION_REQUEST:
                return await self._handle_optimization_request(message)
            elif message.message_type == MessageType.REBALANCING_REQUEST:
                return await self._handle_rebalancing_request(message)
            elif message.message_type == MessageType.TRADING_PROPOSAL:
                return await self._handle_trading_proposal(message)
            else:
                return AgentResponse(
                    response_id=str(uuid.uuid4()),
                    original_message_id=message.message_id,
                    agent_id=self.agent_id,
                    success=False,
                    error_message=f"Unsupported message type: {message.message_type}"
                )
                
        except Exception as e:
            self.logger.error(f"Error processing message {message.message_id}: {str(e)}")
            return AgentResponse(
                response_id=str(uuid.uuid4()),
                original_message_id=message.message_id,
                agent_id=self.agent_id,
                success=False,
                error_message=str(e)
            )
    
    async def vote_on_decision(self, proposal: TradingProposal) -> Vote:
        """
        Cast vote on trading proposal based on portfolio optimization analysis.
        
        Args:
            proposal: Trading proposal to vote on
            
        Returns:
            Vote with confidence score and rationale
        """
        try:
            # Analyze the proposal from portfolio optimization perspective
            vote_score, confidence, rationale = await self._analyze_trading_proposal(proposal)
            
            vote = Vote(
                agent_id=self.agent_id,
                proposal_id=proposal.proposal_id,
                score=vote_score,
                confidence=confidence,
                rationale=rationale,
                timestamp=datetime.utcnow()
            )
            
            self.logger.info(f"Voted {vote_score} on proposal {proposal.proposal_id} with confidence {confidence:.2f}")
            return vote
            
        except Exception as e:
            self.logger.error(f"Error voting on proposal {proposal.proposal_id}: {str(e)}")
            # Return neutral vote in case of error
            return Vote(
                agent_id=self.agent_id,
                proposal_id=proposal.proposal_id,
                score=50,
                confidence=0.1,
                rationale=f"Error in analysis: {str(e)}",
                timestamp=datetime.utcnow()
            )
    
    async def _handle_market_data_update(self, message: Message) -> AgentResponse:
        """Handle market data updates."""
        try:
            market_data = message.payload.get('market_data', {})
            
            # Update market conditions
            self._update_market_conditions(market_data)
            
            # Update asset universe if needed
            await self._update_asset_universe(market_data)
            
            return AgentResponse(
                response_id=str(uuid.uuid4()),
                original_message_id=message.message_id,
                agent_id=self.agent_id,
                success=True,
                data={"status": "market_data_processed"}
            )
            
        except Exception as e:
            self.logger.error(f"Error handling market data update: {str(e)}")
            raise
    
    async def _handle_portfolio_update(self, message: Message) -> AgentResponse:
        """Handle portfolio state updates."""
        try:
            portfolio_data = message.payload.get('portfolio')
            if portfolio_data:
                self.current_portfolio = Portfolio.from_dict(portfolio_data)
                self.logger.info(f"Updated portfolio state: {self.current_portfolio.total_value}")
            
            return AgentResponse(
                response_id=str(uuid.uuid4()),
                original_message_id=message.message_id,
                agent_id=self.agent_id,
                success=True,
                data={"status": "portfolio_updated"}
            )
            
        except Exception as e:
            self.logger.error(f"Error handling portfolio update: {str(e)}")
            raise
    
    async def _handle_optimization_request(self, message: Message) -> AgentResponse:
        """Handle portfolio optimization requests."""
        try:
            optimization_type = message.payload.get('optimization_type', 'mean_variance')
            constraints = message.payload.get('constraints', {})
            
            if not self.asset_universe:
                raise ValueError("Asset universe not initialized")
            
            if self.correlation_matrix is None:
                raise ValueError("Correlation matrix not available")
            
            # Perform optimization
            if optimization_type == 'mean_variance':
                allocation_plan = self.optimizer.mean_variance_optimization(
                    self.asset_universe,
                    self.correlation_matrix,
                    constraints=constraints
                )
            elif optimization_type == 'multi_objective':
                objectives = message.payload.get('objectives', {'return': 0.6, 'risk': 0.3, 'diversification': 0.1})
                allocation_plan = self.optimizer.multi_objective_optimization(
                    self.asset_universe,
                    self.correlation_matrix,
                    objectives,
                    constraints
                )
            elif optimization_type == 'risk_parity':
                allocation_plan = self.optimizer.calculate_risk_parity_allocation(
                    self.asset_universe,
                    self.correlation_matrix,
                    constraints
                )
            else:
                # Default to market-condition-based optimization
                allocation_plan = self.rebalancing_engine.generate_allocation_plan(
                    self.asset_universe,
                    self.correlation_matrix,
                    self.market_conditions,
                    constraints
                )
            
            # Update target allocations
            self.target_allocations = allocation_plan.allocations
            
            # Record optimization history
            self.optimization_history.append({
                'timestamp': datetime.utcnow(),
                'optimization_type': optimization_type,
                'allocation_plan': allocation_plan,
                'market_conditions': self.market_conditions.copy()
            })
            
            return AgentResponse(
                response_id=str(uuid.uuid4()),
                original_message_id=message.message_id,
                agent_id=self.agent_id,
                success=True,
                data={
                    "allocation_plan": allocation_plan.__dict__,
                    "optimization_type": optimization_type
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error handling optimization request: {str(e)}")
            raise
    
    async def _handle_rebalancing_request(self, message: Message) -> AgentResponse:
        """Handle portfolio rebalancing requests."""
        try:
            if not self.current_portfolio:
                raise ValueError("Current portfolio not available")
            
            if not self.target_allocations:
                raise ValueError("Target allocations not set")
            
            transaction_costs = message.payload.get('transaction_costs', {})
            market_data = message.payload.get('market_data', {})
            
            # Convert market data to MarketData objects
            market_data_objects = {}
            for symbol, data in market_data.items():
                if isinstance(data, dict):
                    market_data_objects[symbol] = MarketData.from_dict(data)
                else:
                    market_data_objects[symbol] = data
            
            # Calculate rebalancing plan
            rebalancing_plan = self.rebalancing_engine.calculate_rebalancing_plan(
                self.current_portfolio,
                self.target_allocations,
                transaction_costs,
                market_data_objects
            )
            
            # Determine if rebalancing should proceed
            should_rebalance, rationale = self.rebalancing_engine.should_rebalance(
                rebalancing_plan.current_allocations,
                rebalancing_plan.target_allocations,
                rebalancing_plan.rebalancing_cost,
                rebalancing_plan.expected_improvement
            )
            
            # Record rebalancing history
            self.rebalancing_history.append(rebalancing_plan)
            
            return AgentResponse(
                response_id=str(uuid.uuid4()),
                original_message_id=message.message_id,
                agent_id=self.agent_id,
                success=True,
                data={
                    "rebalancing_plan": rebalancing_plan.__dict__,
                    "should_rebalance": should_rebalance,
                    "rationale": rationale
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error handling rebalancing request: {str(e)}")
            raise
    
    async def _handle_trading_proposal(self, message: Message) -> AgentResponse:
        """Handle trading proposal evaluation."""
        try:
            proposal_data = message.payload.get('proposal')
            if not proposal_data:
                raise ValueError("No proposal data provided")
            
            proposal = TradingProposal.from_dict(proposal_data)
            vote = await self.vote_on_decision(proposal)
            
            return AgentResponse(
                response_id=str(uuid.uuid4()),
                original_message_id=message.message_id,
                agent_id=self.agent_id,
                success=True,
                data={"vote": vote.__dict__}
            )
            
        except Exception as e:
            self.logger.error(f"Error handling trading proposal: {str(e)}")
            raise
    
    async def _analyze_trading_proposal(self, proposal: TradingProposal) -> tuple[int, float, str]:
        """
        Analyze trading proposal from portfolio optimization perspective.
        
        Args:
            proposal: Trading proposal to analyze
            
        Returns:
            Tuple of (vote_score, confidence, rationale)
        """
        try:
            if not self.current_portfolio or not self.target_allocations:
                return 50, 0.3, "Insufficient portfolio data for analysis"
            
            symbol = proposal.symbol
            action = proposal.action
            quantity = proposal.quantity
            
            # Calculate current allocation
            current_allocation = 0.0
            if symbol in self.current_portfolio.positions:
                position = self.current_portfolio.positions[symbol]
                current_allocation = position.market_value / self.current_portfolio.total_value
            
            # Get target allocation
            target_allocation = self.target_allocations.get(symbol, 0.0)
            
            # Calculate what the allocation would be after the trade
            trade_value = quantity * proposal.price_target
            new_portfolio_value = self.current_portfolio.total_value + (
                trade_value if action == TradeAction.BUY else -trade_value
            )
            
            if new_portfolio_value <= 0:
                return 0, 0.9, "Trade would result in negative portfolio value"
            
            new_position_value = (
                self.current_portfolio.positions.get(symbol, Position(
                    symbol=symbol, quantity=0, entry_price=proposal.price_target, current_price=proposal.price_target,
                    market_value=0, unrealized_pnl=0, weight=0
                )).market_value + trade_value
            )
            
            new_allocation = new_position_value / new_portfolio_value
            
            # Analyze alignment with target allocation
            current_deviation = abs(current_allocation - target_allocation)
            new_deviation = abs(new_allocation - target_allocation)
            
            # Base score on improvement in allocation
            if new_deviation < current_deviation:
                # Trade improves allocation
                improvement = (current_deviation - new_deviation) / max(current_deviation, 0.01)
                base_score = min(80, 60 + improvement * 20)
                confidence = min(0.9, 0.6 + improvement * 0.3)
                rationale = f"Trade improves allocation alignment (deviation: {current_deviation:.1%} → {new_deviation:.1%})"
            elif new_deviation > current_deviation:
                # Trade worsens allocation
                deterioration = (new_deviation - current_deviation) / max(current_deviation, 0.01)
                base_score = max(20, 40 - deterioration * 20)
                confidence = min(0.8, 0.5 + deterioration * 0.2)
                rationale = f"Trade worsens allocation alignment (deviation: {current_deviation:.1%} → {new_deviation:.1%})"
            else:
                # Neutral impact
                base_score = 50
                confidence = 0.4
                rationale = "Trade has neutral impact on portfolio allocation"
            
            # Adjust for portfolio concentration risk
            if new_allocation > 0.3:  # High concentration
                base_score -= 10
                confidence += 0.1
                rationale += "; High concentration risk"
            elif new_allocation < 0.05 and target_allocation > 0.05:  # Under-allocation
                base_score += 5
                rationale += "; Addresses under-allocation"
            
            # Adjust for diversification impact
            if self.asset_universe:
                current_hhi = sum(
                    (pos.market_value / self.current_portfolio.total_value) ** 2
                    for pos in self.current_portfolio.positions.values()
                )
                
                # Estimate new HHI (simplified)
                new_hhi_estimate = current_hhi + (new_allocation ** 2 - current_allocation ** 2)
                
                if new_hhi_estimate < current_hhi:  # Improves diversification
                    base_score += 5
                    rationale += "; Improves diversification"
                elif new_hhi_estimate > current_hhi + 0.05:  # Significantly worsens diversification
                    base_score -= 10
                    rationale += "; Reduces diversification"
            
            # Ensure score is within bounds
            vote_score = max(0, min(100, int(base_score)))
            confidence = max(0.1, min(0.95, confidence))
            
            return vote_score, confidence, rationale
            
        except Exception as e:
            self.logger.error(f"Error analyzing trading proposal: {str(e)}")
            return 50, 0.2, f"Analysis error: {str(e)}"
    
    def _update_market_conditions(self, market_data: Dict[str, Any]) -> None:
        """Update market conditions based on market data."""
        try:
            # Calculate market volatility
            if 'volatility' in market_data:
                self.market_conditions['volatility'] = market_data['volatility']
            
            # Determine market regime
            if 'regime' in market_data:
                self.market_conditions['regime'] = market_data['regime']
            else:
                # Simple regime detection based on volatility
                volatility = self.market_conditions.get('volatility', 0.15)
                if volatility > 0.25:
                    self.market_conditions['regime'] = 'high_volatility'
                elif volatility < 0.10:
                    self.market_conditions['regime'] = 'low_volatility'
                else:
                    self.market_conditions['regime'] = 'normal'
            
            # Update timestamp
            self.market_conditions['last_update'] = datetime.utcnow()
            
        except Exception as e:
            self.logger.error(f"Error updating market conditions: {str(e)}")
    
    async def _update_asset_universe(self, market_data: Dict[str, Any]) -> None:
        """Update asset universe based on market data."""
        try:
            # This is a simplified implementation
            # In practice, this would integrate with market data feeds
            
            assets = []
            symbols = market_data.get('symbols', [])
            
            for symbol in symbols:
                symbol_data = market_data.get(symbol, {})
                if symbol_data:
                    asset = Asset(
                        symbol=symbol,
                        expected_return=symbol_data.get('expected_return', 0.08),
                        volatility=symbol_data.get('volatility', 0.20),
                        current_price=symbol_data.get('price', 100.0),
                        market_cap=symbol_data.get('market_cap', 1000000000),
                        sector=symbol_data.get('sector', 'Unknown')
                    )
                    assets.append(asset)
            
            if assets:
                self.asset_universe = assets
                
                # Update correlation matrix (simplified - would use historical data in practice)
                n_assets = len(assets)
                if n_assets > 1:
                    # Create a simple correlation matrix
                    correlation_matrix = np.eye(n_assets)
                    for i in range(n_assets):
                        for j in range(i + 1, n_assets):
                            # Same sector assets have higher correlation
                            if assets[i].sector == assets[j].sector:
                                correlation = 0.6
                            else:
                                correlation = 0.3
                            correlation_matrix[i, j] = correlation
                            correlation_matrix[j, i] = correlation
                    
                    self.correlation_matrix = correlation_matrix
                
                self.logger.info(f"Updated asset universe with {len(assets)} assets")
            
        except Exception as e:
            self.logger.error(f"Error updating asset universe: {str(e)}")
    
    def get_portfolio_state(self) -> Dict[str, Any]:
        """
        Get current portfolio optimization state.
        
        Returns:
            Dictionary containing portfolio state information
        """
        return {
            'agent_id': self.agent_id,
            'current_portfolio': self.current_portfolio.__dict__ if self.current_portfolio else None,
            'target_allocations': self.target_allocations,
            'asset_universe_size': len(self.asset_universe),
            'market_conditions': self.market_conditions,
            'optimization_history_count': len(self.optimization_history),
            'rebalancing_history_count': len(self.rebalancing_history),
            'last_optimization': self.optimization_history[-1] if self.optimization_history else None
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get portfolio optimization performance metrics.
        
        Returns:
            Dictionary containing performance metrics
        """
        if not self.optimization_history:
            return {'status': 'no_optimization_history'}
        
        recent_optimizations = self.optimization_history[-10:]  # Last 10 optimizations
        
        # Calculate average Sharpe ratio
        sharpe_ratios = [opt['allocation_plan'].sharpe_ratio for opt in recent_optimizations]
        avg_sharpe_ratio = np.mean(sharpe_ratios) if sharpe_ratios else 0
        
        # Calculate rebalancing frequency
        rebalancing_frequency = len(self.rebalancing_history) / max(len(self.optimization_history), 1)
        
        return {
            'average_sharpe_ratio': avg_sharpe_ratio,
            'optimization_count': len(self.optimization_history),
            'rebalancing_count': len(self.rebalancing_history),
            'rebalancing_frequency': rebalancing_frequency,
            'last_optimization_time': self.optimization_history[-1]['timestamp'].isoformat(),
            'current_market_regime': self.market_conditions.get('regime', 'unknown')
        }
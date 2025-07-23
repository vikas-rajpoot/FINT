"""
Risk Manager Agent implementation.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from ..models.message_models import Message, AgentResponse, MessageType
from ..models.trading_models import (
    TradingProposal, Vote, Portfolio, RiskParameters, 
    PositionSize, ExposureReport, RiskMetrics
)
from ..services.risk_engine import RiskCalculationEngine
from .base_agent import BaseAgent


class RiskManagerAgent(BaseAgent):
    """
    Risk Manager Agent specializing in risk assessment and management.
    
    Responsibilities:
    - Position sizing calculations
    - Stop-loss and take-profit level determination
    - Portfolio exposure monitoring
    - Correlation analysis and risk metrics calculation
    - Risk-based voting on trading proposals
    - Risk alert generation
    """
    
    def __init__(self, agent_id: str, config, portfolio: Optional[Portfolio] = None):
        """
        Initialize the Risk Manager Agent.
        
        Args:
            agent_id: Unique identifier for this agent
            config: Agent configuration parameters
            portfolio: Current portfolio state (optional)
        """
        super().__init__(agent_id, config)
        
        # Initialize risk calculation engine
        self.risk_engine = RiskCalculationEngine(risk_free_rate=0.02)
        
        # Portfolio and market data
        self.current_portfolio = portfolio
        self.historical_prices: Dict[str, List[float]] = {}
        self.historical_returns: Dict[str, List[float]] = {}
        self.market_volatility = 0.20  # Default market volatility
        self.vix_level: Optional[float] = None
        
        # Risk parameters and limits
        self.base_risk_params = RiskParameters(
            stop_loss_pct=0.02,
            take_profit_pct=0.05,
            max_position_size_pct=0.1,
            max_portfolio_risk_pct=0.02,
            volatility_adjustment=1.0
        )
        
        # Sector and currency mappings for exposure analysis
        self.sector_mapping: Dict[str, str] = {}
        self.currency_mapping: Dict[str, str] = {}
        
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{agent_id}")
    
    async def process_message(self, message: Message) -> AgentResponse:
        """Process incoming messages for risk analysis."""
        try:
            if message.message_type == MessageType.MARKET_DATA:
                return await self._handle_market_data_update(message)
            elif message.message_type == MessageType.SYSTEM_ALERT and message.payload.get("request_type") == "risk_analysis":
                return await self._handle_risk_analysis_request(message)
            elif message.message_type == MessageType.TRADING_PROPOSAL:
                return await self._handle_trading_proposal(message)
            elif message.message_type == MessageType.VOTE_REQUEST:
                return await self._handle_vote_request(message)
            else:
                return AgentResponse(
                    response_id=f"resp_{datetime.utcnow().timestamp()}",
                    original_message_id=message.message_id,
                    agent_id=self.agent_id,
                    success=True,
                    result={"status": "message_type_not_handled", "message_type": message.message_type.value}
                )
                
        except Exception as e:
            self.logger.error(f"Error processing message {message.message_id}: {str(e)}")
            return AgentResponse(
                response_id=f"resp_{datetime.utcnow().timestamp()}",
                original_message_id=message.message_id,
                agent_id=self.agent_id,
                success=False,
                error_message=str(e)
            )
    
    async def vote_on_decision(self, proposal: TradingProposal) -> Vote:
        """Cast vote based on comprehensive risk analysis."""
        try:
            if not self.current_portfolio:
                return Vote(
                    agent_id=self.agent_id,
                    proposal_id=proposal.proposal_id,
                    score=0,
                    confidence=0.1,
                    rationale="No portfolio data available for risk assessment",
                    timestamp=datetime.utcnow()
                )
            
            # Calculate dynamic risk parameters
            risk_params = self.risk_engine.calculate_stop_loss_take_profit(
                proposal, self.current_portfolio, self.historical_prices
            )
            
            # Adjust for market volatility
            adjusted_params = self.risk_engine.adjust_risk_for_market_volatility(
                risk_params, self.market_volatility, self.vix_level
            )
            
            # Enforce risk limits
            is_allowed, violations = self.risk_engine.enforce_risk_limits(
                proposal, self.current_portfolio, adjusted_params
            )
            
            # Calculate position size
            position_size = self.risk_engine.calculate_position_size(
                proposal, self.current_portfolio, adjusted_params, self.historical_prices
            )
            
            # Calculate portfolio risk metrics
            portfolio_risk_metrics = self.risk_engine.calculate_risk_metrics(
                self.current_portfolio, self.historical_returns
            )
            
            # Use proposal risk metrics for voting (these are more relevant for the specific trade)
            # but also consider portfolio risk metrics for context
            combined_risk_metrics = RiskMetrics(
                var_95=max(proposal.risk_metrics.var_95, portfolio_risk_metrics.var_95),
                cvar_95=max(proposal.risk_metrics.cvar_95, portfolio_risk_metrics.cvar_95),
                sharpe_ratio=proposal.risk_metrics.sharpe_ratio,  # Use proposal's Sharpe ratio
                max_drawdown=max(proposal.risk_metrics.max_drawdown, portfolio_risk_metrics.max_drawdown),
                volatility=max(proposal.risk_metrics.volatility, portfolio_risk_metrics.volatility)
            )
            
            # Generate risk-based vote
            score, confidence, rationale = self._calculate_risk_vote(
                proposal, position_size, combined_risk_metrics, is_allowed, violations
            )
            
            # Generate alerts for extreme scenarios
            await self._generate_risk_alerts(proposal, violations, combined_risk_metrics)
            
            return Vote(
                agent_id=self.agent_id,
                proposal_id=proposal.proposal_id,
                score=score,
                confidence=confidence,
                rationale=rationale,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            self.logger.error(f"Error voting on proposal {proposal.proposal_id}: {str(e)}")
            return Vote(
                agent_id=self.agent_id,
                proposal_id=proposal.proposal_id,
                score=0,
                confidence=0.1,
                rationale=f"Error in risk analysis: {str(e)}",
                timestamp=datetime.utcnow()
            )
    
    def update_portfolio(self, portfolio: Portfolio) -> None:
        """Update current portfolio state."""
        self.current_portfolio = portfolio
        self.logger.info(f"Portfolio updated: {len(portfolio.positions)} positions, total value: ${portfolio.total_value:,.2f}")
    
    def update_market_data(self, symbol: str, prices: List[float], returns: List[float]) -> None:
        """Update historical market data for a symbol."""
        self.historical_prices[symbol] = prices[-252:]  # Keep last year of data
        self.historical_returns[symbol] = returns[-252:]  # Keep last year of data
        
        # Update market volatility (simple average of all symbols)
        if self.historical_returns:
            all_returns = []
            for symbol_returns in self.historical_returns.values():
                all_returns.extend(symbol_returns)
            
            if all_returns:
                import numpy as np
                self.market_volatility = np.std(all_returns) * np.sqrt(252)  # Annualized
    
    def update_sector_mapping(self, mapping: Dict[str, str]) -> None:
        """Update sector mapping for exposure analysis."""
        self.sector_mapping.update(mapping)
    
    def update_currency_mapping(self, mapping: Dict[str, str]) -> None:
        """Update currency mapping for exposure analysis."""
        self.currency_mapping.update(mapping)
    
    async def _handle_market_data_update(self, message: Message) -> AgentResponse:
        """Handle market data update messages."""
        try:
            data = message.payload
            symbol = data.get("symbol")
            prices = data.get("prices", [])
            returns = data.get("returns", [])
            
            if symbol and prices:
                self.update_market_data(symbol, prices, returns)
            
            return AgentResponse(
                response_id=f"resp_{datetime.utcnow().timestamp()}",
                original_message_id=message.message_id,
                agent_id=self.agent_id,
                success=True,
                result={"status": "market_data_updated", "symbol": symbol}
            )
            
        except Exception as e:
            raise Exception(f"Error handling market data update: {str(e)}")
    
    async def _handle_portfolio_update(self, message: Message) -> AgentResponse:
        """Handle portfolio update messages."""
        try:
            portfolio_data = message.payload
            # Convert portfolio data to Portfolio object
            # This would need proper deserialization in a real implementation
            
            return AgentResponse(
                response_id=f"resp_{datetime.utcnow().timestamp()}",
                original_message_id=message.message_id,
                agent_id=self.agent_id,
                success=True,
                result={"status": "portfolio_updated"}
            )
            
        except Exception as e:
            raise Exception(f"Error handling portfolio update: {str(e)}")
    
    async def _handle_risk_analysis_request(self, message: Message) -> AgentResponse:
        """Handle risk analysis requests."""
        try:
            if not self.current_portfolio:
                return AgentResponse(
                    response_id=f"resp_{datetime.utcnow().timestamp()}",
                    original_message_id=message.message_id,
                    agent_id=self.agent_id,
                    success=False,
                    error_message="No portfolio data available"
                )
            
            # Calculate comprehensive risk metrics
            risk_metrics = self.risk_engine.calculate_risk_metrics(
                self.current_portfolio, self.historical_returns
            )
            
            # Calculate exposure report
            exposure_report = self.risk_engine.monitor_portfolio_exposure(
                self.current_portfolio, self.sector_mapping, 
                self.currency_mapping, self.historical_returns
            )
            
            return AgentResponse(
                response_id=f"resp_{datetime.utcnow().timestamp()}",
                original_message_id=message.message_id,
                agent_id=self.agent_id,
                success=True,
                result={
                    "risk_metrics": risk_metrics.to_dict(),
                    "exposure_report": {
                        "total_exposure": exposure_report.total_exposure,
                        "sector_exposure": exposure_report.sector_exposure,
                        "currency_exposure": exposure_report.currency_exposure,
                        "concentration_risk": exposure_report.concentration_risk,
                        "diversification_ratio": exposure_report.diversification_ratio
                    }
                }
            )
            
        except Exception as e:
            raise Exception(f"Error handling risk analysis request: {str(e)}")
    
    async def _handle_trading_proposal(self, message: Message) -> AgentResponse:
        """Handle trading proposal messages."""
        try:
            proposal_data = message.payload
            # Convert to TradingProposal object
            # This would need proper deserialization in a real implementation
            
            return AgentResponse(
                response_id=f"resp_{datetime.utcnow().timestamp()}",
                original_message_id=message.message_id,
                agent_id=self.agent_id,
                success=True,
                result={"status": "proposal_analyzed"}
            )
            
        except Exception as e:
            raise Exception(f"Error handling trading proposal: {str(e)}")
    
    async def _handle_vote_request(self, message: Message) -> AgentResponse:
        """Handle vote request messages from consensus engine."""
        try:
            proposal_data = message.payload.get("proposal")
            if not proposal_data:
                return AgentResponse(
                    response_id=f"resp_{datetime.utcnow().timestamp()}",
                    original_message_id=message.message_id,
                    agent_id=self.agent_id,
                    success=False,
                    error_message="No proposal data in vote request"
                )
            
            # Deserialize trading proposal
            proposal = TradingProposal.from_dict(proposal_data)
            
            # Cast vote based on risk analysis
            vote = await self.vote_on_decision(proposal)
            
            # Publish vote response
            vote_response_message = Message(
                message_id=f"vote_resp_{datetime.utcnow().timestamp()}",
                message_type=MessageType.VOTE_RESPONSE,
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                payload={"vote": vote.to_dict()},
                timestamp=datetime.utcnow(),
                correlation_id=message.correlation_id
            )
            
            # In a real implementation, this would be sent via message bus
            # await self.message_bus.publish("consensus_votes", vote_response_message)
            
            return AgentResponse(
                response_id=f"resp_{datetime.utcnow().timestamp()}",
                original_message_id=message.message_id,
                agent_id=self.agent_id,
                success=True,
                result={
                    "status": "vote_cast",
                    "vote": vote.to_dict()
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error handling vote request: {str(e)}")
            return AgentResponse(
                response_id=f"resp_{datetime.utcnow().timestamp()}",
                original_message_id=message.message_id,
                agent_id=self.agent_id,
                success=False,
                error_message=f"Error processing vote request: {str(e)}"
            )
    
    def _calculate_risk_vote(
        self, 
        proposal: TradingProposal, 
        position_size: PositionSize,
        risk_metrics: RiskMetrics,
        is_allowed: bool,
        violations: List[str]
    ) -> tuple[int, float, str]:
        """Calculate risk-based vote score and confidence with enhanced weighting."""
        
        # Base score starts at 50 (neutral)
        score = 50
        confidence = 0.5
        rationale_parts = []
        risk_weight_factors = []
        
        # Risk limit violations - heavily penalize with high confidence
        if not is_allowed:
            score -= 45  # Increased penalty
            confidence = 0.95  # Very high confidence in rejection
            rationale_parts.append(f"CRITICAL: Risk limit violations: {', '.join(violations)}")
            risk_weight_factors.append("critical_violations")
        
        # Position size analysis with graduated penalties
        if position_size.recommended_quantity == 0:
            score -= 35
            confidence = max(confidence, 0.9)
            rationale_parts.append("CRITICAL: Recommended position size is zero")
            risk_weight_factors.append("zero_position")
        elif position_size.risk_pct > 0.08:  # More than 8% portfolio risk
            score -= 30
            confidence = max(confidence, 0.85)
            rationale_parts.append(f"EXTREME: Portfolio risk {position_size.risk_pct:.2%} exceeds 8%")
            risk_weight_factors.append("extreme_risk")
        elif position_size.risk_pct > 0.05:  # More than 5% portfolio risk
            score -= 20
            confidence = max(confidence, 0.8)
            rationale_parts.append(f"HIGH: Portfolio risk {position_size.risk_pct:.2%} exceeds 5%")
            risk_weight_factors.append("high_risk")
        elif position_size.risk_pct > 0.03:  # More than 3% portfolio risk
            score -= 10
            rationale_parts.append(f"MODERATE: Portfolio risk {position_size.risk_pct:.2%} exceeds 3%")
            risk_weight_factors.append("moderate_risk")
        elif position_size.risk_pct < 0.01:  # Less than 1% portfolio risk
            score += 15
            rationale_parts.append(f"CONSERVATIVE: Low risk level {position_size.risk_pct:.2%}")
            risk_weight_factors.append("conservative")
        
        # Portfolio risk metrics analysis with enhanced thresholds
        if risk_metrics.var_95 > 0.15:  # VaR > 15%
            score -= 25
            confidence = max(confidence, 0.9)
            rationale_parts.append(f"EXTREME: Portfolio VaR {risk_metrics.var_95:.2%} > 15%")
            risk_weight_factors.append("extreme_var")
        elif risk_metrics.var_95 > 0.10:  # VaR > 10%
            score -= 15
            confidence = max(confidence, 0.8)
            rationale_parts.append(f"HIGH: Portfolio VaR {risk_metrics.var_95:.2%} > 10%")
            risk_weight_factors.append("high_var")
        
        if risk_metrics.max_drawdown > 0.25:  # Max drawdown > 25%
            score -= 25
            confidence = max(confidence, 0.9)
            rationale_parts.append(f"EXTREME: Max drawdown {risk_metrics.max_drawdown:.2%} > 25%")
            risk_weight_factors.append("extreme_drawdown")
        elif risk_metrics.max_drawdown > 0.15:  # Max drawdown > 15%
            score -= 15
            confidence = max(confidence, 0.8)
            rationale_parts.append(f"HIGH: Max drawdown {risk_metrics.max_drawdown:.2%} > 15%")
            risk_weight_factors.append("high_drawdown")
        
        # Sharpe ratio analysis with enhanced scoring
        if risk_metrics.sharpe_ratio < 0:  # Negative Sharpe ratio
            score -= 20
            confidence = max(confidence, 0.85)
            rationale_parts.append(f"POOR: Negative Sharpe ratio {risk_metrics.sharpe_ratio:.2f}")
            risk_weight_factors.append("negative_sharpe")
        elif risk_metrics.sharpe_ratio < 0.5:  # Poor risk-adjusted returns
            score -= 10
            rationale_parts.append(f"LOW: Sharpe ratio {risk_metrics.sharpe_ratio:.2f} < 0.5")
            risk_weight_factors.append("low_sharpe")
        elif risk_metrics.sharpe_ratio > 2.0:  # Excellent risk-adjusted returns
            score += 15
            rationale_parts.append(f"EXCELLENT: Sharpe ratio {risk_metrics.sharpe_ratio:.2f} > 2.0")
            risk_weight_factors.append("excellent_sharpe")
        elif risk_metrics.sharpe_ratio > 1.5:  # Good risk-adjusted returns
            score += 10
            rationale_parts.append(f"GOOD: Sharpe ratio {risk_metrics.sharpe_ratio:.2f} > 1.5")
            risk_weight_factors.append("good_sharpe")
        
        # Market volatility adjustment with enhanced thresholds
        if self.market_volatility > 0.40:  # Extreme market volatility
            score -= 20
            confidence = max(confidence, 0.85)
            rationale_parts.append(f"EXTREME: Market volatility {self.market_volatility:.2%} > 40%")
            risk_weight_factors.append("extreme_volatility")
        elif self.market_volatility > 0.30:  # High market volatility
            score -= 12
            confidence = max(confidence, 0.75)
            rationale_parts.append(f"HIGH: Market volatility {self.market_volatility:.2%} > 30%")
            risk_weight_factors.append("high_volatility")
        elif self.market_volatility < 0.10:  # Very low market volatility
            score += 8
            rationale_parts.append(f"STABLE: Low market volatility {self.market_volatility:.2%}")
            risk_weight_factors.append("low_volatility")
        elif self.market_volatility < 0.15:  # Low market volatility
            score += 5
            rationale_parts.append(f"CALM: Market volatility {self.market_volatility:.2%} < 15%")
            risk_weight_factors.append("calm_market")
        
        # VIX level adjustment with enhanced thresholds
        if self.vix_level:
            if self.vix_level > 40:  # Extreme fear
                score -= 25
                confidence = max(confidence, 0.9)
                rationale_parts.append(f"PANIC: VIX level {self.vix_level} indicates extreme fear")
                risk_weight_factors.append("panic_vix")
            elif self.vix_level > 30:  # High fear
                score -= 15
                confidence = max(confidence, 0.8)
                rationale_parts.append(f"FEAR: High VIX level {self.vix_level}")
                risk_weight_factors.append("fear_vix")
            elif self.vix_level < 12:  # Extreme complacency
                score -= 8  # Complacency can be risky too
                rationale_parts.append(f"COMPLACENT: Very low VIX {self.vix_level} may indicate complacency")
                risk_weight_factors.append("complacent_vix")
            elif self.vix_level < 15:  # Low fear
                score += 8
                rationale_parts.append(f"STABLE: Low VIX level {self.vix_level}")
                risk_weight_factors.append("stable_vix")
        
        # Proposal confidence factor
        if proposal.confidence < 0.3:  # Low confidence proposal
            score -= 15
            rationale_parts.append(f"LOW CONFIDENCE: Proposal confidence {proposal.confidence:.2f}")
            risk_weight_factors.append("low_proposal_confidence")
        elif proposal.confidence > 0.8:  # High confidence proposal
            score += 8
            rationale_parts.append(f"HIGH CONFIDENCE: Proposal confidence {proposal.confidence:.2f}")
            risk_weight_factors.append("high_proposal_confidence")
        
        # Ensure score is within bounds
        score = max(0, min(100, score))
        
        # Adjust confidence based on data quality and risk factors
        if not self.historical_prices or not self.historical_returns:
            confidence *= 0.6  # Reduced confidence with limited data
            rationale_parts.append("LIMITED DATA: Reduced confidence due to insufficient historical data")
            risk_weight_factors.append("limited_data")
        
        # Calculate risk-weighted confidence
        extreme_risk_factors = sum(1 for factor in risk_weight_factors 
                                 if factor.startswith(('critical', 'extreme', 'panic')))
        if extreme_risk_factors > 0:
            confidence = min(0.98, confidence + (extreme_risk_factors * 0.1))
        
        # Higher confidence for extreme scores with strong risk signals
        if score < 15 and len(risk_weight_factors) >= 2:
            confidence = min(0.98, confidence + 0.15)
        elif score > 85 and 'conservative' in risk_weight_factors:
            confidence = min(0.95, confidence + 0.1)
        
        # Generate comprehensive rationale
        if rationale_parts:
            rationale = f"Risk-weighted assessment: {'; '.join(rationale_parts)}"
        else:
            rationale = "Standard risk analysis with no significant concerns"
        
        # Add risk weighting summary
        if risk_weight_factors:
            rationale += f" [Risk factors: {', '.join(set(risk_weight_factors))}]"
        
        return score, confidence, rationale
    
    async def _generate_risk_alerts(
        self, 
        proposal: TradingProposal, 
        violations: List[str],
        risk_metrics: RiskMetrics
    ) -> None:
        """Generate comprehensive risk alerts for extreme scenarios."""
        alerts = []
        
        # Critical risk limit violations
        if violations:
            severity = "CRITICAL" if len(violations) > 2 else "HIGH"
            alerts.append({
                "type": "RISK_LIMIT_VIOLATION",
                "severity": severity,
                "message": f"Risk limits violated for {proposal.symbol}: {', '.join(violations)}",
                "proposal_id": proposal.proposal_id,
                "symbol": proposal.symbol,
                "action": proposal.action.value,
                "timestamp": datetime.utcnow().isoformat(),
                "violations": violations
            })
        
        # Extreme portfolio risk metrics
        if risk_metrics.var_95 > 0.20:  # VaR > 20%
            alerts.append({
                "type": "EXTREME_VAR",
                "severity": "CRITICAL",
                "message": f"Portfolio VaR critically high: {risk_metrics.var_95:.2%} (>20%)",
                "proposal_id": proposal.proposal_id,
                "symbol": proposal.symbol,
                "var_95": risk_metrics.var_95,
                "timestamp": datetime.utcnow().isoformat()
            })
        elif risk_metrics.var_95 > 0.15:  # VaR > 15%
            alerts.append({
                "type": "HIGH_VAR",
                "severity": "HIGH",
                "message": f"Portfolio VaR extremely high: {risk_metrics.var_95:.2%} (>15%)",
                "proposal_id": proposal.proposal_id,
                "symbol": proposal.symbol,
                "var_95": risk_metrics.var_95,
                "timestamp": datetime.utcnow().isoformat()
            })
        
        if risk_metrics.max_drawdown > 0.30:  # Max drawdown > 30%
            alerts.append({
                "type": "EXTREME_DRAWDOWN",
                "severity": "CRITICAL",
                "message": f"Portfolio max drawdown critically high: {risk_metrics.max_drawdown:.2%} (>30%)",
                "proposal_id": proposal.proposal_id,
                "symbol": proposal.symbol,
                "max_drawdown": risk_metrics.max_drawdown,
                "timestamp": datetime.utcnow().isoformat()
            })
        elif risk_metrics.max_drawdown > 0.20:  # Max drawdown > 20%
            alerts.append({
                "type": "HIGH_DRAWDOWN",
                "severity": "HIGH",
                "message": f"Portfolio max drawdown extremely high: {risk_metrics.max_drawdown:.2%} (>20%)",
                "proposal_id": proposal.proposal_id,
                "symbol": proposal.symbol,
                "max_drawdown": risk_metrics.max_drawdown,
                "timestamp": datetime.utcnow().isoformat()
            })
        
        # Negative Sharpe ratio alert
        if risk_metrics.sharpe_ratio < -0.5:
            alerts.append({
                "type": "NEGATIVE_SHARPE",
                "severity": "HIGH",
                "message": f"Portfolio Sharpe ratio severely negative: {risk_metrics.sharpe_ratio:.2f}",
                "proposal_id": proposal.proposal_id,
                "symbol": proposal.symbol,
                "sharpe_ratio": risk_metrics.sharpe_ratio,
                "timestamp": datetime.utcnow().isoformat()
            })
        
        # Market volatility alerts
        if self.market_volatility > 0.50:  # Extreme market volatility
            alerts.append({
                "type": "EXTREME_MARKET_VOLATILITY",
                "severity": "CRITICAL",
                "message": f"Market volatility critically high: {self.market_volatility:.2%} (>50%)",
                "proposal_id": proposal.proposal_id,
                "symbol": proposal.symbol,
                "market_volatility": self.market_volatility,
                "timestamp": datetime.utcnow().isoformat()
            })
        elif self.market_volatility > 0.40:  # Very high market volatility
            alerts.append({
                "type": "HIGH_MARKET_VOLATILITY",
                "severity": "HIGH",
                "message": f"Market volatility extremely high: {self.market_volatility:.2%} (>40%)",
                "proposal_id": proposal.proposal_id,
                "symbol": proposal.symbol,
                "market_volatility": self.market_volatility,
                "timestamp": datetime.utcnow().isoformat()
            })
        
        # VIX level alerts
        if self.vix_level:
            if self.vix_level > 50:  # Extreme panic
                alerts.append({
                    "type": "EXTREME_VIX",
                    "severity": "CRITICAL",
                    "message": f"VIX indicates extreme market panic: {self.vix_level} (>50)",
                    "proposal_id": proposal.proposal_id,
                    "symbol": proposal.symbol,
                    "vix_level": self.vix_level,
                    "timestamp": datetime.utcnow().isoformat()
                })
            elif self.vix_level > 40:  # High fear
                alerts.append({
                    "type": "HIGH_VIX",
                    "severity": "HIGH",
                    "message": f"VIX indicates high market fear: {self.vix_level} (>40)",
                    "proposal_id": proposal.proposal_id,
                    "symbol": proposal.symbol,
                    "vix_level": self.vix_level,
                    "timestamp": datetime.utcnow().isoformat()
                })
            elif self.vix_level < 10:  # Extreme complacency
                alerts.append({
                    "type": "EXTREME_COMPLACENCY",
                    "severity": "MEDIUM",
                    "message": f"VIX indicates extreme market complacency: {self.vix_level} (<10)",
                    "proposal_id": proposal.proposal_id,
                    "symbol": proposal.symbol,
                    "vix_level": self.vix_level,
                    "timestamp": datetime.utcnow().isoformat()
                })
        
        # Proposal-specific alerts
        if proposal.confidence < 0.2:
            alerts.append({
                "type": "LOW_PROPOSAL_CONFIDENCE",
                "severity": "MEDIUM",
                "message": f"Trading proposal has very low confidence: {proposal.confidence:.2f}",
                "proposal_id": proposal.proposal_id,
                "symbol": proposal.symbol,
                "proposal_confidence": proposal.confidence,
                "timestamp": datetime.utcnow().isoformat()
            })
        
        # Log and process alerts
        for alert in alerts:
            # Log with appropriate level based on severity
            if alert["severity"] == "CRITICAL":
                self.logger.critical(f"CRITICAL Risk Alert: {alert['message']}")
            elif alert["severity"] == "HIGH":
                self.logger.error(f"HIGH Risk Alert: {alert['message']}")
            else:
                self.logger.warning(f"Risk Alert: {alert['message']}")
            
            # In a real implementation, these would be sent to monitoring systems
            # await self.message_bus.publish("risk_alerts", alert)
            
            # For critical alerts, also send to emergency channels
            if alert["severity"] == "CRITICAL":
                # await self.message_bus.publish("emergency_alerts", alert)
                pass
        
        # Store alerts for audit trail
        if alerts:
            self.logger.info(f"Generated {len(alerts)} risk alerts for proposal {proposal.proposal_id}")
            # In a real implementation, store in database for compliance
            # await self._store_risk_alerts(alerts)
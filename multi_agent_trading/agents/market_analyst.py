"""
Market Analyst Agent implementation.
"""

import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

from ..models.message_models import Message, AgentResponse, MessageType
from ..models.trading_models import TradingProposal, Vote, MarketData, TradeAction
from ..analysis.technical_analysis import TechnicalAnalyzer, TechnicalAnalysis
from ..analysis.sentiment_analysis import SentimentAnalyzer, SentimentAnalysis, NewsData
from .base_agent import BaseAgent


class MarketAnalystAgent(BaseAgent):
    """
    Market Analyst Agent specializing in technical and sentiment analysis.
    
    Responsibilities:
    - Technical analysis using multiple indicators
    - Sentiment analysis from news and social media
    - Pattern recognition and trend identification
    - Market regime detection
    - Generate trading proposals based on analysis
    - Vote on trading proposals from other agents
    """
    
    def __init__(self, agent_id: str, config):
        """Initialize the Market Analyst Agent."""
        super().__init__(agent_id, config)
        
        # Initialize analysis engines
        self.technical_analyzer = TechnicalAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Cache for recent analysis results
        self.technical_analysis_cache: Dict[str, TechnicalAnalysis] = {}
        self.sentiment_analysis_cache: Dict[str, SentimentAnalysis] = {}
        
        # Historical data storage for trend analysis
        self.price_history: Dict[str, List[float]] = {}
        self.sentiment_history: Dict[str, List[float]] = {}
        
        self.logger.info("Market Analyst Agent initialized with technical and sentiment analysis capabilities")
    
    async def process_message(self, message: Message) -> AgentResponse:
        """Process incoming messages for market analysis."""
        try:
            if message.message_type == MessageType.MARKET_DATA:
                return await self._handle_market_data(message)
            elif message.message_type == MessageType.VOTE_REQUEST:
                return await self._handle_vote_request(message)
            elif message.message_type == MessageType.HEALTH_CHECK:
                return await self._handle_health_check(message)
            else:
                return AgentResponse(
                    response_id=str(uuid.uuid4()),
                    original_message_id=message.message_id,
                    agent_id=self.agent_id,
                    success=False,
                    error_message=f"Unsupported message type: {message.message_type.value}"
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
    
    async def _handle_market_data(self, message: Message) -> AgentResponse:
        """Handle incoming market data and perform analysis."""
        try:
            # Extract market data from message payload
            market_data = MarketData.from_dict(message.payload)
            
            # Perform technical analysis
            technical_analysis = await self._analyze_technical_data(market_data)
            
            # Perform sentiment analysis if news data is available
            sentiment_analysis = None
            if "news_data" in message.payload:
                sentiment_analysis = await self._analyze_sentiment_data(
                    market_data.symbol, 
                    message.payload["news_data"]
                )
            
            # Generate trading proposal if conditions are met
            proposal = await self._generate_trading_proposal(
                market_data, technical_analysis, sentiment_analysis
            )
            
            # Prepare response
            result = {
                "symbol": market_data.symbol,
                "technical_analysis": technical_analysis.to_dict() if technical_analysis else None,
                "sentiment_analysis": sentiment_analysis.to_dict() if sentiment_analysis else None,
                "trading_proposal": proposal.to_dict() if proposal else None
            }
            
            # Publish trading proposal if generated
            if proposal:
                await self._publish_trading_proposal(proposal)
            
            return AgentResponse(
                response_id=str(uuid.uuid4()),
                original_message_id=message.message_id,
                agent_id=self.agent_id,
                success=True,
                result=result
            )
        
        except Exception as e:
            self.logger.error(f"Error handling market data: {str(e)}")
            return AgentResponse(
                response_id=str(uuid.uuid4()),
                original_message_id=message.message_id,
                agent_id=self.agent_id,
                success=False,
                error_message=str(e)
            )
    
    async def _handle_vote_request(self, message: Message) -> AgentResponse:
        """Handle vote request for trading proposal."""
        try:
            # Extract trading proposal from message payload
            proposal = TradingProposal.from_dict(message.payload["proposal"])
            
            # Cast vote based on analysis
            vote = await self.vote_on_decision(proposal)
            
            # Send vote response
            await self._send_vote_response(vote, message.correlation_id)
            
            return AgentResponse(
                response_id=str(uuid.uuid4()),
                original_message_id=message.message_id,
                agent_id=self.agent_id,
                success=True,
                result={"vote": vote.to_dict()}
            )
        
        except Exception as e:
            self.logger.error(f"Error handling vote request: {str(e)}")
            return AgentResponse(
                response_id=str(uuid.uuid4()),
                original_message_id=message.message_id,
                agent_id=self.agent_id,
                success=False,
                error_message=str(e)
            )
    
    async def _handle_health_check(self, message: Message) -> AgentResponse:
        """Handle health check request."""
        health_status = self.get_health_status()
        health_status["analysis_cache_size"] = {
            "technical": len(self.technical_analysis_cache),
            "sentiment": len(self.sentiment_analysis_cache)
        }
        
        return AgentResponse(
            response_id=str(uuid.uuid4()),
            original_message_id=message.message_id,
            agent_id=self.agent_id,
            success=True,
            result=health_status
        )
    
    async def _analyze_technical_data(self, market_data: MarketData) -> TechnicalAnalysis:
        """Perform technical analysis on market data."""
        try:
            symbol = market_data.symbol
            
            # Update price history
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            
            self.price_history[symbol].append(market_data.price)
            
            # Keep only recent history (last 200 data points)
            if len(self.price_history[symbol]) > 200:
                self.price_history[symbol] = self.price_history[symbol][-200:]
            
            # Create price arrays for analysis (simplified - in real implementation would have OHLC data)
            prices = self.price_history[symbol]
            
            # Ensure we have enough data points for analysis
            if len(prices) < 2:
                # Add some dummy historical data for initial analysis
                base_price = market_data.price
                prices = [base_price * (1 + (i - 25) * 0.001) for i in range(50)] + prices
                self.price_history[symbol] = prices
            
            highs = [p * 1.01 for p in prices]  # Approximate highs
            lows = [p * 0.99 for p in prices]   # Approximate lows
            volumes = [1000] * len(prices)      # Placeholder volumes
            
            # Perform technical analysis
            technical_analysis = self.technical_analyzer.analyze(
                highs=highs,
                lows=lows,
                closes=prices,
                volumes=volumes,
                symbol=symbol
            )
            
            # Cache the result
            self.technical_analysis_cache[symbol] = technical_analysis
            
            self.logger.debug(f"Technical analysis completed for {symbol}: {technical_analysis.overall_signal}")
            
            return technical_analysis
        
        except Exception as e:
            self.logger.error(f"Error in technical analysis for {symbol}: {str(e)}")
            # Return a neutral technical analysis on error
            from ..analysis.technical_analysis import (
                TechnicalAnalysis, TechnicalIndicators, PatternAnalysis, TrendDirection, PatternType
            )
            
            neutral_indicators = TechnicalIndicators(
                rsi=50.0, macd=0.0, macd_signal=0.0, macd_histogram=0.0,
                bollinger_upper=market_data.price * 1.02, 
                bollinger_middle=market_data.price, 
                bollinger_lower=market_data.price * 0.98,
                sma_20=market_data.price, sma_50=market_data.price, 
                ema_12=market_data.price, ema_26=market_data.price,
                stochastic_k=50.0, stochastic_d=50.0, williams_r=-50.0,
                atr=market_data.price * 0.02, adx=25.0
            )
            
            neutral_pattern = PatternAnalysis(
                pattern_type=PatternType.NONE,
                confidence=0.0,
                support_level=market_data.price * 0.95,
                resistance_level=market_data.price * 1.05,
                target_price=None,
                stop_loss=None
            )
            
            return TechnicalAnalysis(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                indicators=neutral_indicators,
                trend_direction=TrendDirection.SIDEWAYS,
                trend_strength=0.5,
                pattern_analysis=neutral_pattern,
                overall_signal="HOLD",
                confidence=0.5
            )
    
    async def _analyze_sentiment_data(self, symbol: str, news_data: List[Dict[str, Any]]) -> SentimentAnalysis:
        """Perform sentiment analysis on news and social media data."""
        try:
            # Convert news data to NewsData objects
            news_items = []
            social_posts = []
            analyst_reports = []
            
            for item in news_data:
                if item.get("type") == "news":
                    news_items.append(NewsData(
                        title=item.get("title", ""),
                        content=item.get("content", ""),
                        source=item.get("source", "FINANCIAL_NEWS"),
                        timestamp=datetime.fromisoformat(item.get("timestamp", datetime.utcnow().isoformat())),
                        symbol=symbol
                    ))
                elif item.get("type") == "social":
                    social_posts.append(item.get("content", ""))
                elif item.get("type") == "analyst":
                    analyst_reports.append(item.get("content", ""))
            
            # Get historical sentiment for trend analysis
            historical_sentiment = self.sentiment_history.get(symbol, [])
            
            # Perform sentiment analysis
            sentiment_analysis = self.sentiment_analyzer.analyze(
                symbol=symbol,
                news_data=news_items,
                social_data=social_posts,
                analyst_reports=analyst_reports,
                historical_sentiment=historical_sentiment
            )
            
            # Update sentiment history
            if symbol not in self.sentiment_history:
                self.sentiment_history[symbol] = []
            
            self.sentiment_history[symbol].append(sentiment_analysis.overall_score)
            
            # Keep only recent history (last 50 data points)
            if len(self.sentiment_history[symbol]) > 50:
                self.sentiment_history[symbol] = self.sentiment_history[symbol][-50:]
            
            # Cache the result
            self.sentiment_analysis_cache[symbol] = sentiment_analysis
            
            self.logger.debug(f"Sentiment analysis completed for {symbol}: {sentiment_analysis.overall_sentiment.value}")
            
            return sentiment_analysis
        
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {str(e)}")
            # Return neutral sentiment on error
            from ..analysis.sentiment_analysis import SentimentPolarity, SentimentScore
            return SentimentAnalysis(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                overall_sentiment=SentimentPolarity.NEUTRAL,
                overall_score=0.0,
                overall_confidence=0.0,
                news_sentiment=SentimentScore(SentimentPolarity.NEUTRAL, 0.0, 0.0, []),
                social_sentiment=SentimentScore(SentimentPolarity.NEUTRAL, 0.0, 0.0, []),
                analyst_sentiment=SentimentScore(SentimentPolarity.NEUTRAL, 0.0, 0.0, []),
                sentiment_trend="STABLE",
                key_themes=[],
                risk_factors=[]
            )
    
    async def _generate_trading_proposal(self, market_data: MarketData, 
                                       technical_analysis: TechnicalAnalysis,
                                       sentiment_analysis: Optional[SentimentAnalysis]) -> Optional[TradingProposal]:
        """Generate trading proposal based on technical and sentiment analysis."""
        try:
            # Only generate proposals if we have strong signals
            technical_confidence = technical_analysis.confidence
            technical_signal = technical_analysis.overall_signal
            
            sentiment_confidence = sentiment_analysis.overall_confidence if sentiment_analysis else 0.0
            sentiment_score = sentiment_analysis.overall_score if sentiment_analysis else 0.0
            
            # Combined confidence threshold
            combined_confidence = (technical_confidence * 0.6 + sentiment_confidence * 0.4)
            
            # Only generate proposal if combined confidence is above threshold
            if combined_confidence < 0.7:
                return None
            
            # Determine action based on signals
            action = TradeAction.HOLD
            rationale_parts = []
            
            if technical_signal == "BUY":
                if sentiment_score >= 0 or sentiment_analysis is None:
                    action = TradeAction.BUY
                    rationale_parts.append(f"Technical analysis suggests BUY with {technical_confidence:.2f} confidence")
                    if sentiment_analysis:
                        rationale_parts.append(f"Sentiment analysis shows {sentiment_analysis.overall_sentiment.value} sentiment")
            elif technical_signal == "SELL":
                if sentiment_score <= 0 or sentiment_analysis is None:
                    action = TradeAction.SELL
                    rationale_parts.append(f"Technical analysis suggests SELL with {technical_confidence:.2f} confidence")
                    if sentiment_analysis:
                        rationale_parts.append(f"Sentiment analysis shows {sentiment_analysis.overall_sentiment.value} sentiment")
            
            # Don't generate proposal for HOLD actions
            if action == TradeAction.HOLD:
                return None
            
            # Calculate position size (simplified)
            base_quantity = 100
            confidence_multiplier = min(2.0, combined_confidence * 2)
            quantity = int(base_quantity * confidence_multiplier)
            
            # Set price target based on technical analysis
            current_price = market_data.price
            if action == TradeAction.BUY:
                price_target = current_price * 1.02  # 2% above current price
            else:
                price_target = current_price * 0.98  # 2% below current price
            
            # Create risk metrics (simplified)
            from ..models.trading_models import RiskMetrics
            risk_metrics = RiskMetrics(
                var_95=current_price * 0.05,  # 5% VaR
                cvar_95=current_price * 0.08,  # 8% CVaR
                sharpe_ratio=1.2,  # Placeholder
                max_drawdown=0.15,  # 15% max drawdown
                volatility=technical_analysis.indicators.atr / current_price if technical_analysis.indicators.atr > 0 else 0.02
            )
            
            # Create trading proposal
            proposal = TradingProposal(
                proposal_id=str(uuid.uuid4()),
                symbol=market_data.symbol,
                action=action,
                quantity=quantity,
                price_target=price_target,
                rationale="; ".join(rationale_parts),
                confidence=combined_confidence,
                risk_metrics=risk_metrics,
                timestamp=datetime.utcnow()
            )
            
            self.logger.info(f"Generated trading proposal for {market_data.symbol}: {action.value} {quantity} shares at {price_target:.2f}")
            
            return proposal
        
        except Exception as e:
            self.logger.error(f"Error generating trading proposal: {str(e)}")
            return None
    
    async def _publish_trading_proposal(self, proposal: TradingProposal) -> None:
        """Publish trading proposal to message bus."""
        try:
            message = Message(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.TRADING_PROPOSAL,
                sender_id=self.agent_id,
                recipient_id=None,  # Broadcast to all agents
                payload=proposal.to_dict(),
                timestamp=datetime.utcnow(),
                correlation_id=proposal.proposal_id
            )
            
            await self.message_bus.publish("trading_proposals", message)
            self.logger.info(f"Published trading proposal {proposal.proposal_id} for {proposal.symbol}")
        
        except Exception as e:
            self.logger.error(f"Error publishing trading proposal: {str(e)}")
    
    async def _send_vote_response(self, vote: Vote, correlation_id: Optional[str]) -> None:
        """Send vote response to consensus engine."""
        try:
            message = Message(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.VOTE_RESPONSE,
                sender_id=self.agent_id,
                recipient_id="consensus_engine",
                payload=vote.to_dict(),
                timestamp=datetime.utcnow(),
                correlation_id=correlation_id
            )
            
            await self.message_bus.publish("vote_responses", message)
            self.logger.debug(f"Sent vote response for proposal {vote.proposal_id}")
        
        except Exception as e:
            self.logger.error(f"Error sending vote response: {str(e)}")
    
    async def vote_on_decision(self, proposal: TradingProposal) -> Vote:
        """Cast vote based on technical and sentiment analysis."""
        try:
            symbol = proposal.symbol
            
            # Get recent analysis results
            technical_analysis = self.technical_analysis_cache.get(symbol)
            sentiment_analysis = self.sentiment_analysis_cache.get(symbol)
            
            # If no recent analysis, perform quick analysis
            if not technical_analysis and symbol in self.price_history:
                prices = self.price_history[symbol]
                if len(prices) >= 10:
                    highs = [p * 1.01 for p in prices[-50:]]
                    lows = [p * 0.99 for p in prices[-50:]]
                    volumes = [1000] * len(prices[-50:])
                    
                    technical_analysis = self.technical_analyzer.analyze(
                        highs=highs,
                        lows=lows,
                        closes=prices[-50:],
                        volumes=volumes,
                        symbol=symbol
                    )
            
            # Calculate vote score based on analysis
            base_score = 50  # Neutral
            confidence = 0.5
            rationale_parts = []
            
            if technical_analysis:
                # Technical analysis contribution
                if proposal.action == TradeAction.BUY and technical_analysis.overall_signal == "BUY":
                    base_score += int(technical_analysis.confidence * 30)
                    rationale_parts.append(f"Technical analysis supports BUY ({technical_analysis.confidence:.2f} confidence)")
                elif proposal.action == TradeAction.SELL and technical_analysis.overall_signal == "SELL":
                    base_score += int(technical_analysis.confidence * 30)
                    rationale_parts.append(f"Technical analysis supports SELL ({technical_analysis.confidence:.2f} confidence)")
                elif technical_analysis.overall_signal == "HOLD":
                    base_score -= 10
                    rationale_parts.append("Technical analysis suggests HOLD")
                else:
                    base_score -= int(technical_analysis.confidence * 20)
                    rationale_parts.append(f"Technical analysis contradicts proposal ({technical_analysis.overall_signal})")
                
                confidence = max(confidence, technical_analysis.confidence)
            
            if sentiment_analysis:
                # Sentiment analysis contribution
                sentiment_score = sentiment_analysis.overall_score
                sentiment_confidence = sentiment_analysis.overall_confidence
                
                if proposal.action == TradeAction.BUY and sentiment_score > 0.1:
                    base_score += int(sentiment_confidence * abs(sentiment_score) * 20)
                    rationale_parts.append(f"Positive sentiment supports BUY ({sentiment_analysis.overall_sentiment.value})")
                elif proposal.action == TradeAction.SELL and sentiment_score < -0.1:
                    base_score += int(sentiment_confidence * abs(sentiment_score) * 20)
                    rationale_parts.append(f"Negative sentiment supports SELL ({sentiment_analysis.overall_sentiment.value})")
                elif abs(sentiment_score) <= 0.1:
                    rationale_parts.append("Neutral sentiment")
                else:
                    base_score -= int(sentiment_confidence * abs(sentiment_score) * 15)
                    rationale_parts.append(f"Sentiment contradicts proposal ({sentiment_analysis.overall_sentiment.value})")
                
                confidence = max(confidence, sentiment_confidence)
            
            # Ensure score is within bounds
            final_score = max(0, min(100, base_score))
            
            # Adjust confidence based on agreement
            if not rationale_parts:
                rationale_parts.append("Limited analysis data available")
                confidence = 0.3
            
            rationale = "; ".join(rationale_parts)
            
            vote = Vote(
                agent_id=self.agent_id,
                proposal_id=proposal.proposal_id,
                score=final_score,
                confidence=confidence,
                rationale=rationale,
                timestamp=datetime.utcnow()
            )
            
            self.logger.debug(f"Cast vote for proposal {proposal.proposal_id}: score={final_score}, confidence={confidence:.2f}")
            
            return vote
        
        except Exception as e:
            self.logger.error(f"Error casting vote: {str(e)}")
            # Return neutral vote on error
            return Vote(
                agent_id=self.agent_id,
                proposal_id=proposal.proposal_id,
                score=50,
                confidence=0.1,
                rationale=f"Error in analysis: {str(e)}",
                timestamp=datetime.utcnow()
            )
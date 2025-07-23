"""
Integration tests for Market Analyst Agent.
"""

import pytest
import asyncio
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from multi_agent_trading.agents.market_analyst import MarketAnalystAgent
from multi_agent_trading.models.config_models import AgentConfig, MessageQueueConfig
from multi_agent_trading.models.message_models import Message, MessageType
from multi_agent_trading.models.trading_models import (
    MarketData, TradingProposal, TradeAction, RiskMetrics
)


class TestMarketAnalystAgent:
    """Test cases for MarketAnalystAgent integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create mock configuration
        self.config = MagicMock(spec=AgentConfig)
        self.config.agent_type = "market_analyst"
        self.config.model_version = "1.0.0"
        self.config.model_path = "/tmp/models"
        self.config.health_check_interval = 30
        self.config.timeout_seconds = 10
        self.config.message_queue_config = MagicMock(spec=MessageQueueConfig)
        self.config.message_queue_config.to_dict.return_value = {
            "host": "localhost",
            "port": 5672,
            "username": "guest",
            "password": "guest"
        }
        
        # Create agent instance
        self.agent = MarketAnalystAgent("market_analyst_1", self.config)
        
        # Mock the message bus and other services
        self.agent.message_bus = AsyncMock()
        self.agent.model_manager = AsyncMock()
        self.agent.metrics_collector = AsyncMock()
        
        # Sample market data
        self.sample_market_data = MarketData(
            symbol="AAPL",
            timestamp=datetime.utcnow(),
            price=150.0,
            volume=1000000,
            bid=149.95,
            ask=150.05,
            technical_indicators={"rsi": 65.0, "macd": 1.2}
        )
        
        # Sample news data
        self.sample_news_data = [
            {
                "type": "news",
                "title": "Apple Reports Strong Quarterly Earnings",
                "content": "Apple exceeded analyst expectations with record revenue growth and bullish guidance for next quarter.",
                "source": "FINANCIAL_NEWS",
                "timestamp": datetime.utcnow().isoformat()
            },
            {
                "type": "social",
                "content": "AAPL looking strong! Great earnings beat, buying more shares"
            },
            {
                "type": "analyst",
                "content": "Upgrade to strong buy with price target increase to $180"
            }
        ]
    
    @pytest.mark.asyncio
    async def test_handle_market_data_message(self):
        """Test handling of market data messages."""
        # Create market data message
        message = Message(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.MARKET_DATA,
            sender_id="data_feed",
            recipient_id=self.agent.agent_id,
            payload=self.sample_market_data.to_dict(),
            timestamp=datetime.utcnow()
        )
        
        # Process the message
        response = await self.agent.process_message(message)
        
        # Verify response
        assert response.success is True
        assert response.agent_id == self.agent.agent_id
        assert response.original_message_id == message.message_id
        assert "symbol" in response.result
        assert response.result["symbol"] == "AAPL"
        assert "technical_analysis" in response.result
    
    @pytest.mark.asyncio
    async def test_handle_market_data_with_news(self):
        """Test handling of market data with news information."""
        # Create market data message with news
        payload = self.sample_market_data.to_dict()
        payload["news_data"] = self.sample_news_data
        
        message = Message(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.MARKET_DATA,
            sender_id="data_feed",
            recipient_id=self.agent.agent_id,
            payload=payload,
            timestamp=datetime.utcnow()
        )
        
        # Process the message
        response = await self.agent.process_message(message)
        
        # Verify response includes sentiment analysis
        assert response.success is True
        assert "technical_analysis" in response.result
        assert "sentiment_analysis" in response.result
        assert response.result["sentiment_analysis"] is not None
    
    @pytest.mark.asyncio
    async def test_handle_vote_request_message(self):
        """Test handling of vote request messages."""
        # Create sample trading proposal
        proposal = TradingProposal(
            proposal_id=str(uuid.uuid4()),
            symbol="AAPL",
            action=TradeAction.BUY,
            quantity=100,
            price_target=155.0,
            rationale="Strong technical signals",
            confidence=0.8,
            risk_metrics=RiskMetrics(
                var_95=7.5, cvar_95=12.0, sharpe_ratio=1.2,
                max_drawdown=0.15, volatility=0.25
            ),
            timestamp=datetime.utcnow()
        )
        
        # Create vote request message
        message = Message(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.VOTE_REQUEST,
            sender_id="consensus_engine",
            recipient_id=self.agent.agent_id,
            payload={"proposal": proposal.to_dict()},
            timestamp=datetime.utcnow(),
            correlation_id=proposal.proposal_id
        )
        
        # Process the message
        response = await self.agent.process_message(message)
        
        # Verify response
        assert response.success is True
        assert "vote" in response.result
        vote_data = response.result["vote"]
        assert vote_data["agent_id"] == self.agent.agent_id
        assert vote_data["proposal_id"] == proposal.proposal_id
        assert 0 <= vote_data["score"] <= 100
        assert 0 <= vote_data["confidence"] <= 1
        assert isinstance(vote_data["rationale"], str)
        
        # Verify vote response was sent
        self.agent.message_bus.publish.assert_called()
    
    @pytest.mark.asyncio
    async def test_handle_health_check_message(self):
        """Test handling of health check messages."""
        message = Message(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.HEALTH_CHECK,
            sender_id="system_monitor",
            recipient_id=self.agent.agent_id,
            payload={},
            timestamp=datetime.utcnow()
        )
        
        # Process the message
        response = await self.agent.process_message(message)
        
        # Verify response
        assert response.success is True
        assert "agent_id" in response.result
        assert "status" in response.result
        assert "analysis_cache_size" in response.result
        assert "technical" in response.result["analysis_cache_size"]
        assert "sentiment" in response.result["analysis_cache_size"]
    
    @pytest.mark.asyncio
    async def test_unsupported_message_type(self):
        """Test handling of unsupported message types."""
        message = Message(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.EXECUTION_RESULT,  # Unsupported for market analyst
            sender_id="execution_agent",
            recipient_id=self.agent.agent_id,
            payload={},
            timestamp=datetime.utcnow()
        )
        
        # Process the message
        response = await self.agent.process_message(message)
        
        # Verify error response
        assert response.success is False
        assert "Unsupported message type" in response.error_message
    
    @pytest.mark.asyncio
    async def test_technical_analysis_caching(self):
        """Test that technical analysis results are cached."""
        # Process market data multiple times
        message = Message(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.MARKET_DATA,
            sender_id="data_feed",
            recipient_id=self.agent.agent_id,
            payload=self.sample_market_data.to_dict(),
            timestamp=datetime.utcnow()
        )
        
        # First processing
        await self.agent.process_message(message)
        assert "AAPL" in self.agent.technical_analysis_cache
        
        # Second processing with updated price
        updated_market_data = MarketData(
            symbol="AAPL",
            timestamp=datetime.utcnow(),
            price=152.0,  # Updated price
            volume=1100000,
            bid=151.95,
            ask=152.05,
            technical_indicators={"rsi": 68.0, "macd": 1.5}
        )
        
        message.payload = updated_market_data.to_dict()
        await self.agent.process_message(message)
        
        # Verify cache was updated
        cached_analysis = self.agent.technical_analysis_cache["AAPL"]
        assert cached_analysis is not None
        assert cached_analysis.symbol == "AAPL"
    
    @pytest.mark.asyncio
    async def test_sentiment_analysis_caching(self):
        """Test that sentiment analysis results are cached."""
        # Create market data message with news
        payload = self.sample_market_data.to_dict()
        payload["news_data"] = self.sample_news_data
        
        message = Message(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.MARKET_DATA,
            sender_id="data_feed",
            recipient_id=self.agent.agent_id,
            payload=payload,
            timestamp=datetime.utcnow()
        )
        
        # Process the message
        await self.agent.process_message(message)
        
        # Verify sentiment analysis was cached
        assert "AAPL" in self.agent.sentiment_analysis_cache
        cached_sentiment = self.agent.sentiment_analysis_cache["AAPL"]
        assert cached_sentiment is not None
        assert cached_sentiment.symbol == "AAPL"
    
    @pytest.mark.asyncio
    async def test_trading_proposal_generation(self):
        """Test trading proposal generation with strong signals."""
        # Add some price history to enable analysis
        self.agent.price_history["AAPL"] = [140.0, 142.0, 145.0, 148.0, 150.0] * 10
        
        # Create market data with strong buy signals
        strong_buy_data = MarketData(
            symbol="AAPL",
            timestamp=datetime.utcnow(),
            price=155.0,
            volume=2000000,
            bid=154.95,
            ask=155.05,
            technical_indicators={"rsi": 25.0, "macd": 2.5}  # Oversold RSI, strong MACD
        )
        
        # Create message with positive news
        positive_news = [
            {
                "type": "news",
                "title": "Apple Announces Revolutionary Product",
                "content": "Breakthrough innovation expected to drive massive revenue growth and market expansion with bullish analyst upgrades.",
                "source": "FINANCIAL_NEWS",
                "timestamp": datetime.utcnow().isoformat()
            }
        ]
        
        payload = strong_buy_data.to_dict()
        payload["news_data"] = positive_news
        
        message = Message(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.MARKET_DATA,
            sender_id="data_feed",
            recipient_id=self.agent.agent_id,
            payload=payload,
            timestamp=datetime.utcnow()
        )
        
        # Process the message
        response = await self.agent.process_message(message)
        
        # Verify trading proposal was generated
        assert response.success is True
        if response.result.get("trading_proposal"):
            proposal_data = response.result["trading_proposal"]
            assert proposal_data["symbol"] == "AAPL"
            assert proposal_data["action"] in ["BUY", "SELL"]
            assert proposal_data["quantity"] > 0
            assert proposal_data["confidence"] > 0
            
            # Verify proposal was published
            self.agent.message_bus.publish.assert_called()
    
    @pytest.mark.asyncio
    async def test_vote_with_cached_analysis(self):
        """Test voting with cached analysis data."""
        # Add cached technical analysis
        from multi_agent_trading.analysis.technical_analysis import (
            TechnicalAnalysis, TechnicalIndicators, PatternAnalysis, TrendDirection, PatternType
        )
        
        indicators = TechnicalIndicators(
            rsi=30.0, macd=1.5, macd_signal=1.0, macd_histogram=0.5,
            bollinger_upper=160.0, bollinger_middle=150.0, bollinger_lower=140.0,
            sma_20=148.0, sma_50=145.0, ema_12=149.0, ema_26=147.0,
            stochastic_k=25.0, stochastic_d=28.0, williams_r=-75.0,
            atr=3.0, adx=35.0
        )
        
        pattern = PatternAnalysis(
            pattern_type=PatternType.DOUBLE_BOTTOM,
            confidence=0.7,
            support_level=140.0,
            resistance_level=160.0,
            target_price=165.0,
            stop_loss=135.0
        )
        
        technical_analysis = TechnicalAnalysis(
            symbol="AAPL",
            timestamp=datetime.utcnow(),
            indicators=indicators,
            trend_direction=TrendDirection.BULLISH,
            trend_strength=0.8,
            pattern_analysis=pattern,
            overall_signal="BUY",
            confidence=0.85
        )
        
        self.agent.technical_analysis_cache["AAPL"] = technical_analysis
        
        # Create BUY proposal
        proposal = TradingProposal(
            proposal_id=str(uuid.uuid4()),
            symbol="AAPL",
            action=TradeAction.BUY,
            quantity=100,
            price_target=155.0,
            rationale="Strong technical signals",
            confidence=0.8,
            risk_metrics=RiskMetrics(
                var_95=7.5, cvar_95=12.0, sharpe_ratio=1.2,
                max_drawdown=0.15, volatility=0.25
            ),
            timestamp=datetime.utcnow()
        )
        
        # Cast vote
        vote = await self.agent.vote_on_decision(proposal)
        
        # Verify vote supports the BUY proposal
        assert vote.agent_id == self.agent.agent_id
        assert vote.proposal_id == proposal.proposal_id
        assert vote.score > 50  # Should support BUY
        assert vote.confidence > 0.5
        assert "Technical analysis supports BUY" in vote.rationale
    
    @pytest.mark.asyncio
    async def test_vote_against_proposal(self):
        """Test voting against a proposal based on analysis."""
        # Add cached technical analysis suggesting SELL
        from multi_agent_trading.analysis.technical_analysis import (
            TechnicalAnalysis, TechnicalIndicators, PatternAnalysis, TrendDirection, PatternType
        )
        
        indicators = TechnicalIndicators(
            rsi=80.0, macd=-1.5, macd_signal=-1.0, macd_histogram=-0.5,
            bollinger_upper=160.0, bollinger_middle=150.0, bollinger_lower=140.0,
            sma_20=148.0, sma_50=152.0, ema_12=147.0, ema_26=151.0,
            stochastic_k=85.0, stochastic_d=88.0, williams_r=-15.0,
            atr=3.0, adx=35.0
        )
        
        pattern = PatternAnalysis(
            pattern_type=PatternType.DOUBLE_TOP,
            confidence=0.7,
            support_level=140.0,
            resistance_level=160.0,
            target_price=135.0,
            stop_loss=165.0
        )
        
        technical_analysis = TechnicalAnalysis(
            symbol="AAPL",
            timestamp=datetime.utcnow(),
            indicators=indicators,
            trend_direction=TrendDirection.BEARISH,
            trend_strength=0.8,
            pattern_analysis=pattern,
            overall_signal="SELL",
            confidence=0.85
        )
        
        self.agent.technical_analysis_cache["AAPL"] = technical_analysis
        
        # Create BUY proposal (contradicts analysis)
        proposal = TradingProposal(
            proposal_id=str(uuid.uuid4()),
            symbol="AAPL",
            action=TradeAction.BUY,
            quantity=100,
            price_target=155.0,
            rationale="Bullish outlook",
            confidence=0.8,
            risk_metrics=RiskMetrics(
                var_95=7.5, cvar_95=12.0, sharpe_ratio=1.2,
                max_drawdown=0.15, volatility=0.25
            ),
            timestamp=datetime.utcnow()
        )
        
        # Cast vote
        vote = await self.agent.vote_on_decision(proposal)
        
        # Verify vote opposes the BUY proposal
        assert vote.score < 50  # Should oppose BUY
        assert "contradicts proposal" in vote.rationale.lower()
    
    @pytest.mark.asyncio
    async def test_error_handling_in_message_processing(self):
        """Test error handling during message processing."""
        # Create message with invalid payload
        message = Message(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.MARKET_DATA,
            sender_id="data_feed",
            recipient_id=self.agent.agent_id,
            payload={"invalid": "data"},  # Invalid market data
            timestamp=datetime.utcnow()
        )
        
        # Process the message
        response = await self.agent.process_message(message)
        
        # Verify error response
        assert response.success is False
        assert response.error_message is not None
        assert response.agent_id == self.agent.agent_id
    
    @pytest.mark.asyncio
    async def test_price_history_management(self):
        """Test that price history is properly managed."""
        # Add many price points to test history limit
        symbol = "AAPL"
        for i in range(250):  # More than the 200 limit
            market_data = MarketData(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                price=100.0 + i,
                volume=1000000,
                bid=99.95 + i,
                ask=100.05 + i,
                technical_indicators={}
            )
            
            message = Message(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.MARKET_DATA,
                sender_id="data_feed",
                recipient_id=self.agent.agent_id,
                payload=market_data.to_dict(),
                timestamp=datetime.utcnow()
            )
            
            await self.agent.process_message(message)
        
        # Verify history is limited to 200 entries
        assert len(self.agent.price_history[symbol]) == 200
        
        # Verify most recent prices are kept
        assert self.agent.price_history[symbol][-1] == 349.0  # 100 + 249
    
    def test_agent_initialization(self):
        """Test proper agent initialization."""
        assert self.agent.agent_id == "market_analyst_1"
        assert self.agent.technical_analyzer is not None
        assert self.agent.sentiment_analyzer is not None
        assert isinstance(self.agent.technical_analysis_cache, dict)
        assert isinstance(self.agent.sentiment_analysis_cache, dict)
        assert isinstance(self.agent.price_history, dict)
        assert isinstance(self.agent.sentiment_history, dict)


if __name__ == "__main__":
    pytest.main([__file__])
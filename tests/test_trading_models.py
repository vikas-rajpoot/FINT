"""
Unit tests for trading data models.
"""

import pytest
import json
from datetime import datetime
from multi_agent_trading.models.trading_models import (
    MarketData, RiskMetrics, TradingProposal, Vote, Experience,
    TradeAction, ValidationError
)


class TestMarketData:
    """Test cases for MarketData model."""
    
    def test_valid_market_data_creation(self):
        """Test creating valid market data."""
        timestamp = datetime.now()
        technical_indicators = {"rsi": 65.5, "macd": 0.12}
        
        market_data = MarketData(
            symbol="AAPL",
            timestamp=timestamp,
            price=150.25,
            volume=1000000,
            bid=150.20,
            ask=150.30,
            technical_indicators=technical_indicators
        )
        
        assert market_data.symbol == "AAPL"
        assert market_data.timestamp == timestamp
        assert market_data.price == 150.25
        assert market_data.volume == 1000000
        assert market_data.bid == 150.20
        assert market_data.ask == 150.30
        assert market_data.technical_indicators == technical_indicators
    
    def test_market_data_validation_errors(self):
        """Test validation errors for invalid market data."""
        timestamp = datetime.now()
        
        # Test empty symbol
        with pytest.raises(ValidationError, match="Symbol must be a non-empty string"):
            MarketData("", timestamp, 150.25, 1000000, 150.20, 150.30, {})
        
        # Test negative price
        with pytest.raises(ValidationError, match="Price must be a positive number"):
            MarketData("AAPL", timestamp, -150.25, 1000000, 150.20, 150.30, {})
        
        # Test negative volume
        with pytest.raises(ValidationError, match="Volume must be a non-negative integer"):
            MarketData("AAPL", timestamp, 150.25, -1000000, 150.20, 150.30, {})
        
        # Test bid >= ask
        with pytest.raises(ValidationError, match="Bid must be less than ask"):
            MarketData("AAPL", timestamp, 150.25, 1000000, 150.30, 150.20, {})
    
    def test_market_data_serialization(self):
        """Test JSON serialization and deserialization."""
        timestamp = datetime.now()
        technical_indicators = {"rsi": 65.5, "macd": 0.12}
        
        original = MarketData(
            symbol="AAPL",
            timestamp=timestamp,
            price=150.25,
            volume=1000000,
            bid=150.20,
            ask=150.30,
            technical_indicators=technical_indicators
        )
        
        # Test to_dict
        data_dict = original.to_dict()
        assert data_dict["symbol"] == "AAPL"
        assert data_dict["timestamp"] == timestamp.isoformat()
        assert data_dict["price"] == 150.25
        
        # Test from_dict
        reconstructed = MarketData.from_dict(data_dict)
        assert reconstructed.symbol == original.symbol
        assert reconstructed.timestamp == original.timestamp
        assert reconstructed.price == original.price
        
        # Test JSON serialization
        json_str = original.to_json()
        from_json = MarketData.from_json(json_str)
        assert from_json.symbol == original.symbol
        assert from_json.price == original.price


class TestRiskMetrics:
    """Test cases for RiskMetrics model."""
    
    def test_valid_risk_metrics_creation(self):
        """Test creating valid risk metrics."""
        risk_metrics = RiskMetrics(
            var_95=-0.05,
            cvar_95=-0.08,
            sharpe_ratio=1.2,
            max_drawdown=0.15,
            volatility=0.25
        )
        
        assert risk_metrics.var_95 == -0.05
        assert risk_metrics.cvar_95 == -0.08
        assert risk_metrics.sharpe_ratio == 1.2
        assert risk_metrics.max_drawdown == 0.15
        assert risk_metrics.volatility == 0.25
    
    def test_risk_metrics_validation_errors(self):
        """Test validation errors for invalid risk metrics."""
        # Test negative max_drawdown
        with pytest.raises(ValidationError, match="Max drawdown must be a non-negative number"):
            RiskMetrics(-0.05, -0.08, 1.2, -0.15, 0.25)
        
        # Test negative volatility
        with pytest.raises(ValidationError, match="Volatility must be a non-negative number"):
            RiskMetrics(-0.05, -0.08, 1.2, 0.15, -0.25)
    
    def test_risk_metrics_serialization(self):
        """Test JSON serialization and deserialization."""
        original = RiskMetrics(
            var_95=-0.05,
            cvar_95=-0.08,
            sharpe_ratio=1.2,
            max_drawdown=0.15,
            volatility=0.25
        )
        
        # Test to_dict
        data_dict = original.to_dict()
        assert data_dict["var_95"] == -0.05
        assert data_dict["sharpe_ratio"] == 1.2
        
        # Test from_dict
        reconstructed = RiskMetrics.from_dict(data_dict)
        assert reconstructed.var_95 == original.var_95
        assert reconstructed.sharpe_ratio == original.sharpe_ratio
        
        # Test JSON serialization
        json_str = original.to_json()
        from_json = RiskMetrics.from_json(json_str)
        assert from_json.var_95 == original.var_95


class TestTradingProposal:
    """Test cases for TradingProposal model."""
    
    def test_valid_trading_proposal_creation(self):
        """Test creating valid trading proposal."""
        timestamp = datetime.now()
        risk_metrics = RiskMetrics(-0.05, -0.08, 1.2, 0.15, 0.25)
        
        proposal = TradingProposal(
            proposal_id="prop_123",
            symbol="AAPL",
            action=TradeAction.BUY,
            quantity=100,
            price_target=155.0,
            rationale="Strong technical indicators",
            confidence=0.85,
            risk_metrics=risk_metrics,
            timestamp=timestamp
        )
        
        assert proposal.proposal_id == "prop_123"
        assert proposal.symbol == "AAPL"
        assert proposal.action == TradeAction.BUY
        assert proposal.quantity == 100
        assert proposal.confidence == 0.85
    
    def test_trading_proposal_validation_errors(self):
        """Test validation errors for invalid trading proposal."""
        timestamp = datetime.now()
        risk_metrics = RiskMetrics(-0.05, -0.08, 1.2, 0.15, 0.25)
        
        # Test invalid confidence
        with pytest.raises(ValidationError, match="Confidence must be a number between 0 and 1"):
            TradingProposal("prop_123", "AAPL", TradeAction.BUY, 100, 155.0, 
                          "rationale", 1.5, risk_metrics, timestamp)
        
        # Test negative quantity
        with pytest.raises(ValidationError, match="Quantity must be a positive integer"):
            TradingProposal("prop_123", "AAPL", TradeAction.BUY, -100, 155.0, 
                          "rationale", 0.85, risk_metrics, timestamp)
    
    def test_trading_proposal_serialization(self):
        """Test JSON serialization and deserialization."""
        timestamp = datetime.now()
        risk_metrics = RiskMetrics(-0.05, -0.08, 1.2, 0.15, 0.25)
        
        original = TradingProposal(
            proposal_id="prop_123",
            symbol="AAPL",
            action=TradeAction.BUY,
            quantity=100,
            price_target=155.0,
            rationale="Strong technical indicators",
            confidence=0.85,
            risk_metrics=risk_metrics,
            timestamp=timestamp
        )
        
        # Test to_dict
        data_dict = original.to_dict()
        assert data_dict["proposal_id"] == "prop_123"
        assert data_dict["action"] == "BUY"
        
        # Test from_dict
        reconstructed = TradingProposal.from_dict(data_dict)
        assert reconstructed.proposal_id == original.proposal_id
        assert reconstructed.action == original.action
        
        # Test JSON serialization
        json_str = original.to_json()
        from_json = TradingProposal.from_json(json_str)
        assert from_json.proposal_id == original.proposal_id


class TestVote:
    """Test cases for Vote model."""
    
    def test_valid_vote_creation(self):
        """Test creating valid vote."""
        timestamp = datetime.now()
        
        vote = Vote(
            agent_id="market_analyst",
            proposal_id="prop_123",
            score=85,
            confidence=0.9,
            rationale="Strong buy signal from technical analysis",
            timestamp=timestamp
        )
        
        assert vote.agent_id == "market_analyst"
        assert vote.proposal_id == "prop_123"
        assert vote.score == 85
        assert vote.confidence == 0.9
    
    def test_vote_validation_errors(self):
        """Test validation errors for invalid vote."""
        timestamp = datetime.now()
        
        # Test invalid score
        with pytest.raises(ValidationError, match="Score must be an integer between 0 and 100"):
            Vote("agent", "prop", 150, 0.9, "rationale", timestamp)
        
        # Test invalid confidence
        with pytest.raises(ValidationError, match="Confidence must be a number between 0 and 1"):
            Vote("agent", "prop", 85, 1.5, "rationale", timestamp)
    
    def test_vote_serialization(self):
        """Test JSON serialization and deserialization."""
        timestamp = datetime.now()
        
        original = Vote(
            agent_id="market_analyst",
            proposal_id="prop_123",
            score=85,
            confidence=0.9,
            rationale="Strong buy signal",
            timestamp=timestamp
        )
        
        # Test to_dict
        data_dict = original.to_dict()
        assert data_dict["agent_id"] == "market_analyst"
        assert data_dict["score"] == 85
        
        # Test from_dict
        reconstructed = Vote.from_dict(data_dict)
        assert reconstructed.agent_id == original.agent_id
        assert reconstructed.score == original.score
        
        # Test JSON serialization
        json_str = original.to_json()
        from_json = Vote.from_json(json_str)
        assert from_json.agent_id == original.agent_id


class TestExperience:
    """Test cases for Experience model."""
    
    def test_valid_experience_creation(self):
        """Test creating valid experience."""
        timestamp = datetime.now()
        
        experience = Experience(
            experience_id="exp_123",
            state={"price": 150.0, "volume": 1000000},
            action={"action": "BUY", "quantity": 100},
            reward=0.05,
            next_state={"price": 152.0, "volume": 1100000},
            agent_contributions={"market_analyst": 0.3, "risk_manager": 0.2},
            timestamp=timestamp
        )
        
        assert experience.experience_id == "exp_123"
        assert experience.reward == 0.05
        assert experience.agent_contributions["market_analyst"] == 0.3
    
    def test_experience_validation_errors(self):
        """Test validation errors for invalid experience."""
        timestamp = datetime.now()
        
        # Test invalid agent contributions
        with pytest.raises(ValidationError, match="Agent contribution must be a number"):
            Experience("exp_123", {}, {}, 0.05, {}, {"agent": "invalid"}, timestamp)
    
    def test_experience_serialization(self):
        """Test JSON serialization and deserialization."""
        timestamp = datetime.now()
        
        original = Experience(
            experience_id="exp_123",
            state={"price": 150.0},
            action={"action": "BUY"},
            reward=0.05,
            next_state={"price": 152.0},
            agent_contributions={"agent1": 0.3},
            timestamp=timestamp
        )
        
        # Test to_dict
        data_dict = original.to_dict()
        assert data_dict["experience_id"] == "exp_123"
        assert data_dict["reward"] == 0.05
        
        # Test from_dict
        reconstructed = Experience.from_dict(data_dict)
        assert reconstructed.experience_id == original.experience_id
        assert reconstructed.reward == original.reward
        
        # Test JSON serialization
        json_str = original.to_json()
        from_json = Experience.from_json(json_str)
        assert from_json.experience_id == original.experience_id
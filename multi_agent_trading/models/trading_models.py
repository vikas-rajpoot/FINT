"""
Core trading data models for the multi-agent trading system.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, Union
import json
import uuid
from decimal import Decimal


class ValidationError(Exception):
    """Custom exception for data validation errors."""
    pass


class TradeAction(Enum):
    """Trading actions that can be taken."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class MarketData:
    """Market data structure for real-time trading information."""
    symbol: str
    timestamp: datetime
    price: float
    volume: int
    bid: float
    ask: float
    technical_indicators: Dict[str, float]
    
    def __post_init__(self):
        """Validate data after initialization."""
        self.validate()
    
    def validate(self):
        """Validate market data integrity."""
        if not self.symbol or not isinstance(self.symbol, str):
            raise ValidationError("Symbol must be a non-empty string")
        
        if not isinstance(self.timestamp, datetime):
            raise ValidationError("Timestamp must be a datetime object")
        
        if not isinstance(self.price, (int, float)) or self.price <= 0:
            raise ValidationError("Price must be a positive number")
        
        if not isinstance(self.volume, int) or self.volume < 0:
            raise ValidationError("Volume must be a non-negative integer")
        
        if not isinstance(self.bid, (int, float)) or self.bid <= 0:
            raise ValidationError("Bid must be a positive number")
        
        if not isinstance(self.ask, (int, float)) or self.ask <= 0:
            raise ValidationError("Ask must be a positive number")
        
        if self.bid >= self.ask:
            raise ValidationError("Bid must be less than ask")
        
        if not isinstance(self.technical_indicators, dict):
            raise ValidationError("Technical indicators must be a dictionary")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "price": self.price,
            "volume": self.volume,
            "bid": self.bid,
            "ask": self.ask,
            "technical_indicators": self.technical_indicators
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarketData':
        """Create instance from dictionary."""
        return cls(
            symbol=data["symbol"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            price=data["price"],
            volume=data["volume"],
            bid=data["bid"],
            ask=data["ask"],
            technical_indicators=data["technical_indicators"]
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'MarketData':
        """Create instance from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


@dataclass
class RiskMetrics:
    """Risk assessment metrics for trading decisions."""
    var_95: float  # Value at Risk at 95% confidence
    cvar_95: float  # Conditional Value at Risk
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    
    def __post_init__(self):
        """Validate data after initialization."""
        self.validate()
    
    def validate(self):
        """Validate risk metrics data."""
        if not isinstance(self.var_95, (int, float)):
            raise ValidationError("VaR 95% must be a number")
        
        if not isinstance(self.cvar_95, (int, float)):
            raise ValidationError("CVaR 95% must be a number")
        
        if not isinstance(self.sharpe_ratio, (int, float)):
            raise ValidationError("Sharpe ratio must be a number")
        
        if not isinstance(self.max_drawdown, (int, float)) or self.max_drawdown < 0:
            raise ValidationError("Max drawdown must be a non-negative number")
        
        if not isinstance(self.volatility, (int, float)) or self.volatility < 0:
            raise ValidationError("Volatility must be a non-negative number")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "var_95": self.var_95,
            "cvar_95": self.cvar_95,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "volatility": self.volatility
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RiskMetrics':
        """Create instance from dictionary."""
        return cls(
            var_95=data["var_95"],
            cvar_95=data["cvar_95"],
            sharpe_ratio=data["sharpe_ratio"],
            max_drawdown=data["max_drawdown"],
            volatility=data["volatility"]
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'RiskMetrics':
        """Create instance from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


@dataclass
class TradingProposal:
    """Trading proposal generated by agents."""
    proposal_id: str
    symbol: str
    action: TradeAction
    quantity: int
    price_target: float
    rationale: str
    confidence: float
    risk_metrics: RiskMetrics
    timestamp: datetime
    
    def __post_init__(self):
        """Validate data after initialization."""
        self.validate()
    
    def validate(self):
        """Validate trading proposal data."""
        if not self.proposal_id or not isinstance(self.proposal_id, str):
            raise ValidationError("Proposal ID must be a non-empty string")
        
        if not self.symbol or not isinstance(self.symbol, str):
            raise ValidationError("Symbol must be a non-empty string")
        
        if not isinstance(self.action, TradeAction):
            raise ValidationError("Action must be a TradeAction enum")
        
        if not isinstance(self.quantity, int) or self.quantity <= 0:
            raise ValidationError("Quantity must be a positive integer")
        
        if not isinstance(self.price_target, (int, float)) or self.price_target <= 0:
            raise ValidationError("Price target must be a positive number")
        
        if not self.rationale or not isinstance(self.rationale, str):
            raise ValidationError("Rationale must be a non-empty string")
        
        if not isinstance(self.confidence, (int, float)) or not (0 <= self.confidence <= 1):
            raise ValidationError("Confidence must be a number between 0 and 1")
        
        if not isinstance(self.risk_metrics, RiskMetrics):
            raise ValidationError("Risk metrics must be a RiskMetrics instance")
        
        if not isinstance(self.timestamp, datetime):
            raise ValidationError("Timestamp must be a datetime object")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "proposal_id": self.proposal_id,
            "symbol": self.symbol,
            "action": self.action.value,
            "quantity": self.quantity,
            "price_target": self.price_target,
            "rationale": self.rationale,
            "confidence": self.confidence,
            "risk_metrics": self.risk_metrics.to_dict(),
            "timestamp": self.timestamp.isoformat()
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradingProposal':
        """Create instance from dictionary."""
        return cls(
            proposal_id=data["proposal_id"],
            symbol=data["symbol"],
            action=TradeAction(data["action"]),
            quantity=data["quantity"],
            price_target=data["price_target"],
            rationale=data["rationale"],
            confidence=data["confidence"],
            risk_metrics=RiskMetrics.from_dict(data["risk_metrics"]),
            timestamp=datetime.fromisoformat(data["timestamp"])
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'TradingProposal':
        """Create instance from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


@dataclass
class Vote:
    """Agent vote on a trading proposal."""
    agent_id: str
    proposal_id: str
    score: int  # 0-100
    confidence: float  # 0-1
    rationale: str
    timestamp: datetime
    
    def __post_init__(self):
        """Validate data after initialization."""
        self.validate()
    
    def validate(self):
        """Validate vote data."""
        if not self.agent_id or not isinstance(self.agent_id, str):
            raise ValidationError("Agent ID must be a non-empty string")
        
        if not self.proposal_id or not isinstance(self.proposal_id, str):
            raise ValidationError("Proposal ID must be a non-empty string")
        
        if not isinstance(self.score, int) or not (0 <= self.score <= 100):
            raise ValidationError("Score must be an integer between 0 and 100")
        
        if not isinstance(self.confidence, (int, float)) or not (0 <= self.confidence <= 1):
            raise ValidationError("Confidence must be a number between 0 and 1")
        
        if not self.rationale or not isinstance(self.rationale, str):
            raise ValidationError("Rationale must be a non-empty string")
        
        if not isinstance(self.timestamp, datetime):
            raise ValidationError("Timestamp must be a datetime object")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent_id": self.agent_id,
            "proposal_id": self.proposal_id,
            "score": self.score,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "timestamp": self.timestamp.isoformat()
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Vote':
        """Create instance from dictionary."""
        return cls(
            agent_id=data["agent_id"],
            proposal_id=data["proposal_id"],
            score=data["score"],
            confidence=data["confidence"],
            rationale=data["rationale"],
            timestamp=datetime.fromisoformat(data["timestamp"])
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Vote':
        """Create instance from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


@dataclass
class Position:
    """Portfolio position data."""
    symbol: str
    quantity: int
    entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    weight: float  # Portfolio weight
    
    def __post_init__(self):
        """Validate data after initialization."""
        self.validate()
    
    def validate(self):
        """Validate position data."""
        if not self.symbol or not isinstance(self.symbol, str):
            raise ValidationError("Symbol must be a non-empty string")
        
        if not isinstance(self.quantity, int):
            raise ValidationError("Quantity must be an integer")
        
        if not isinstance(self.entry_price, (int, float)) or self.entry_price <= 0:
            raise ValidationError("Entry price must be a positive number")
        
        if not isinstance(self.current_price, (int, float)) or self.current_price <= 0:
            raise ValidationError("Current price must be a positive number")
        
        if not isinstance(self.market_value, (int, float)):
            raise ValidationError("Market value must be a number")
        
        if not isinstance(self.unrealized_pnl, (int, float)):
            raise ValidationError("Unrealized PnL must be a number")
        
        if not isinstance(self.weight, (int, float)) or not (0 <= self.weight <= 1):
            raise ValidationError("Weight must be a number between 0 and 1")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "market_value": self.market_value,
            "unrealized_pnl": self.unrealized_pnl,
            "weight": self.weight
        }


@dataclass
class Portfolio:
    """Portfolio data structure."""
    portfolio_id: str
    total_value: float
    cash: float
    positions: Dict[str, Position]
    timestamp: datetime
    
    def __post_init__(self):
        """Validate data after initialization."""
        self.validate()
    
    def validate(self):
        """Validate portfolio data."""
        if not self.portfolio_id or not isinstance(self.portfolio_id, str):
            raise ValidationError("Portfolio ID must be a non-empty string")
        
        if not isinstance(self.total_value, (int, float)) or self.total_value < 0:
            raise ValidationError("Total value must be a non-negative number")
        
        if not isinstance(self.cash, (int, float)) or self.cash < 0:
            raise ValidationError("Cash must be a non-negative number")
        
        if not isinstance(self.positions, dict):
            raise ValidationError("Positions must be a dictionary")
        
        for symbol, position in self.positions.items():
            if not isinstance(symbol, str):
                raise ValidationError("Position symbol must be a string")
            if not isinstance(position, Position):
                raise ValidationError("Position must be a Position instance")
        
        if not isinstance(self.timestamp, datetime):
            raise ValidationError("Timestamp must be a datetime object")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "portfolio_id": self.portfolio_id,
            "total_value": self.total_value,
            "cash": self.cash,
            "positions": {symbol: position.to_dict() for symbol, position in self.positions.items()},
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Portfolio':
        """Create instance from dictionary."""
        positions = {
            symbol: Position(
                symbol=pos_data["symbol"],
                quantity=pos_data["quantity"],
                entry_price=pos_data["entry_price"],
                current_price=pos_data["current_price"],
                market_value=pos_data["market_value"],
                unrealized_pnl=pos_data["unrealized_pnl"],
                weight=pos_data["weight"]
            )
            for symbol, pos_data in data["positions"].items()
        }
        
        return cls(
            portfolio_id=data["portfolio_id"],
            total_value=data["total_value"],
            cash=data["cash"],
            positions=positions,
            timestamp=datetime.fromisoformat(data["timestamp"])
        )


@dataclass
class RiskParameters:
    """Risk parameters for trading decisions."""
    stop_loss_pct: float
    take_profit_pct: float
    max_position_size_pct: float
    max_portfolio_risk_pct: float
    volatility_adjustment: float
    
    def __post_init__(self):
        """Validate data after initialization."""
        self.validate()
    
    def validate(self):
        """Validate risk parameters."""
        if not isinstance(self.stop_loss_pct, (int, float)) or not (0 < self.stop_loss_pct <= 1):
            raise ValidationError("Stop loss percentage must be between 0 and 1")
        
        if not isinstance(self.take_profit_pct, (int, float)) or self.take_profit_pct <= 0:
            raise ValidationError("Take profit percentage must be positive")
        
        if not isinstance(self.max_position_size_pct, (int, float)) or not (0 < self.max_position_size_pct <= 1):
            raise ValidationError("Max position size percentage must be between 0 and 1")
        
        if not isinstance(self.max_portfolio_risk_pct, (int, float)) or not (0 < self.max_portfolio_risk_pct <= 1):
            raise ValidationError("Max portfolio risk percentage must be between 0 and 1")
        
        if not isinstance(self.volatility_adjustment, (int, float)) or self.volatility_adjustment <= 0:
            raise ValidationError("Volatility adjustment must be positive")


@dataclass
class PositionSize:
    """Position sizing calculation result."""
    symbol: str
    recommended_quantity: int
    max_quantity: int
    risk_amount: float
    position_value: float
    risk_pct: float
    rationale: str
    
    def __post_init__(self):
        """Validate data after initialization."""
        self.validate()
    
    def validate(self):
        """Validate position size data."""
        if not self.symbol or not isinstance(self.symbol, str):
            raise ValidationError("Symbol must be a non-empty string")
        
        if not isinstance(self.recommended_quantity, int) or self.recommended_quantity < 0:
            raise ValidationError("Recommended quantity must be a non-negative integer")
        
        if not isinstance(self.max_quantity, int) or self.max_quantity < 0:
            raise ValidationError("Max quantity must be a non-negative integer")
        
        if not isinstance(self.risk_amount, (int, float)) or self.risk_amount < 0:
            raise ValidationError("Risk amount must be non-negative")
        
        if not isinstance(self.position_value, (int, float)) or self.position_value < 0:
            raise ValidationError("Position value must be non-negative")
        
        if not isinstance(self.risk_pct, (int, float)) or not (0 <= self.risk_pct <= 1):
            raise ValidationError("Risk percentage must be between 0 and 1")
        
        if not self.rationale or not isinstance(self.rationale, str):
            raise ValidationError("Rationale must be a non-empty string")


@dataclass
class ExposureReport:
    """Portfolio exposure analysis report."""
    total_exposure: float
    sector_exposure: Dict[str, float]
    currency_exposure: Dict[str, float]
    correlation_matrix: Dict[str, Dict[str, float]]
    concentration_risk: float
    diversification_ratio: float
    timestamp: datetime
    
    def __post_init__(self):
        """Validate data after initialization."""
        self.validate()
    
    def validate(self):
        """Validate exposure report data."""
        if not isinstance(self.total_exposure, (int, float)) or self.total_exposure < 0:
            raise ValidationError("Total exposure must be non-negative")
        
        if not isinstance(self.sector_exposure, dict):
            raise ValidationError("Sector exposure must be a dictionary")
        
        if not isinstance(self.currency_exposure, dict):
            raise ValidationError("Currency exposure must be a dictionary")
        
        if not isinstance(self.correlation_matrix, dict):
            raise ValidationError("Correlation matrix must be a dictionary")
        
        if not isinstance(self.concentration_risk, (int, float)) or not (0 <= self.concentration_risk <= 1):
            raise ValidationError("Concentration risk must be between 0 and 1")
        
        if not isinstance(self.diversification_ratio, (int, float)) or self.diversification_ratio < 0:
            raise ValidationError("Diversification ratio must be non-negative")
        
        if not isinstance(self.timestamp, datetime):
            raise ValidationError("Timestamp must be a datetime object")


@dataclass
class Experience:
    """Experience data for reinforcement learning."""
    experience_id: str
    state: Dict[str, Any]
    action: Dict[str, Any]
    reward: float
    next_state: Dict[str, Any]
    agent_contributions: Dict[str, float]
    timestamp: datetime
    
    def __post_init__(self):
        """Validate data after initialization."""
        self.validate()
    
    def validate(self):
        """Validate experience data."""
        if not self.experience_id or not isinstance(self.experience_id, str):
            raise ValidationError("Experience ID must be a non-empty string")
        
        if not isinstance(self.state, dict):
            raise ValidationError("State must be a dictionary")
        
        if not isinstance(self.action, dict):
            raise ValidationError("Action must be a dictionary")
        
        if not isinstance(self.reward, (int, float)):
            raise ValidationError("Reward must be a number")
        
        if not isinstance(self.next_state, dict):
            raise ValidationError("Next state must be a dictionary")
        
        if not isinstance(self.agent_contributions, dict):
            raise ValidationError("Agent contributions must be a dictionary")
        
        # Validate agent contributions are all floats
        for agent_id, contribution in self.agent_contributions.items():
            if not isinstance(agent_id, str):
                raise ValidationError("Agent ID in contributions must be a string")
            if not isinstance(contribution, (int, float)):
                raise ValidationError("Agent contribution must be a number")
        
        if not isinstance(self.timestamp, datetime):
            raise ValidationError("Timestamp must be a datetime object")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "experience_id": self.experience_id,
            "state": self.state,
            "action": self.action,
            "reward": self.reward,
            "next_state": self.next_state,
            "agent_contributions": self.agent_contributions,
            "timestamp": self.timestamp.isoformat()
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Experience':
        """Create instance from dictionary."""
        return cls(
            experience_id=data["experience_id"],
            state=data["state"],
            action=data["action"],
            reward=data["reward"],
            next_state=data["next_state"],
            agent_contributions=data["agent_contributions"],
            timestamp=datetime.fromisoformat(data["timestamp"])
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Experience':
        """Create instance from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
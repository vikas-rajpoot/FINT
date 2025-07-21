"""
Simple test to verify the project structure and imports work correctly.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all core modules can be imported."""
    try:
        # Test model imports
        from multi_agent_trading.models.trading_models import MarketData, TradingProposal, Vote, Experience
        from multi_agent_trading.models.message_models import Message, AgentResponse
        from multi_agent_trading.models.config_models import AgentConfig, SystemConfig
        
        # Test agent imports
        from multi_agent_trading.agents.base_agent import BaseAgent
        from multi_agent_trading.agents.market_analyst import MarketAnalystAgent
        from multi_agent_trading.agents.risk_manager import RiskManagerAgent
        from multi_agent_trading.agents.portfolio_optimizer import PortfolioOptimizerAgent
        from multi_agent_trading.agents.execution_agent import ExecutionAgent
        
        # Test service imports
        from multi_agent_trading.services.message_bus import MessageBus
        from multi_agent_trading.services.model_manager import ModelManager
        from multi_agent_trading.services.metrics_collector import MetricsCollector
        from multi_agent_trading.services.consensus_engine import ConsensusEngine
        
        # Test infrastructure imports
        from multi_agent_trading.infrastructure.database import DatabaseManager
        
        print("✅ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of core classes."""
    try:
        from datetime import datetime
        from multi_agent_trading.models.trading_models import MarketData, TradeAction
        from multi_agent_trading.models.config_models import AgentConfig, AgentType
        
        # Test MarketData creation
        market_data = MarketData(
            symbol="AAPL",
            timestamp=datetime.utcnow(),
            price=150.0,
            volume=1000,
            bid=149.5,
            ask=150.5,
            technical_indicators={"rsi": 65.0, "macd": 1.2}
        )
        
        # Test serialization
        data_dict = market_data.to_dict()
        restored_data = MarketData.from_dict(data_dict)
        
        assert restored_data.symbol == market_data.symbol
        assert restored_data.price == market_data.price
        
        # Test AgentConfig creation
        config = AgentConfig(
            agent_id="test-agent",
            agent_type=AgentType.MARKET_ANALYST,
            model_version="v1.0.0",
            parameters={},
            resource_limits={},
            message_queue_config={}
        )
        
        assert config.agent_id == "test-agent"
        assert config.agent_type == AgentType.MARKET_ANALYST
        
        print("✅ Basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Functionality test error: {e}")
        return False

if __name__ == "__main__":
    print("Testing Multi-Agent Trading System structure...")
    
    import_success = test_imports()
    functionality_success = test_basic_functionality()
    
    if import_success and functionality_success:
        print("\n🎉 All tests passed! Project structure is correctly set up.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)
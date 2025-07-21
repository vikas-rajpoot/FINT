"""
Main entry point for the Multi-Agent Trading System.
"""

import asyncio
import logging
import os
import signal
import sys
from typing import Dict, Any

from .models.config_models import AgentConfig, AgentType, SystemConfig
from .agents import (
    MarketAnalystAgent,
    RiskManagerAgent, 
    PortfolioOptimizerAgent,
    ExecutionAgent
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TradingSystemOrchestrator:
    """Orchestrates the multi-agent trading system."""
    
    def __init__(self):
        self.agents = {}
        self.system_config = self._load_system_config()
        self._shutdown_event = asyncio.Event()
    
    def _load_system_config(self) -> SystemConfig:
        """Load system configuration from environment variables."""
        return SystemConfig(
            environment=os.getenv("ENVIRONMENT", "dev"),
            message_broker_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
            database_url=os.getenv("POSTGRES_URL", "postgresql://localhost:5432/trading_system"),
            time_series_db_url=os.getenv("INFLUXDB_URL", "http://localhost:8086"),
            logging_level=os.getenv("LOG_LEVEL", "INFO")
        )
    
    def _create_agent_config(self, agent_type: AgentType, agent_id: str) -> AgentConfig:
        """Create agent configuration."""
        return AgentConfig(
            agent_id=agent_id,
            agent_type=agent_type,
            model_version="v1.0.0",
            parameters={},
            resource_limits={"memory": "1Gi", "cpu": "500m"},
            message_queue_config={"broker_url": self.system_config.message_broker_url}
        )
    
    async def start_agent(self, agent_type: AgentType, agent_id: str):
        """Start a specific agent."""
        config = self._create_agent_config(agent_type, agent_id)
        
        # Create agent based on type
        if agent_type == AgentType.MARKET_ANALYST:
            agent = MarketAnalystAgent(agent_id, config)
        elif agent_type == AgentType.RISK_MANAGER:
            agent = RiskManagerAgent(agent_id, config)
        elif agent_type == AgentType.PORTFOLIO_OPTIMIZER:
            agent = PortfolioOptimizerAgent(agent_id, config)
        elif agent_type == AgentType.EXECUTION_AGENT:
            agent = ExecutionAgent(agent_id, config)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        # Start the agent
        await agent.start()
        self.agents[agent_id] = agent
        logger.info(f"Started {agent_type.value} agent: {agent_id}")
    
    async def stop_agent(self, agent_id: str):
        """Stop a specific agent."""
        if agent_id in self.agents:
            await self.agents[agent_id].stop()
            del self.agents[agent_id]
            logger.info(f"Stopped agent: {agent_id}")
    
    async def start_all_agents(self):
        """Start all agents in the system."""
        agent_configs = [
            (AgentType.MARKET_ANALYST, "market-analyst-1"),
            (AgentType.RISK_MANAGER, "risk-manager-1"),
            (AgentType.PORTFOLIO_OPTIMIZER, "portfolio-optimizer-1"),
            (AgentType.EXECUTION_AGENT, "execution-agent-1")
        ]
        
        for agent_type, agent_id in agent_configs:
            await self.start_agent(agent_type, agent_id)
    
    async def stop_all_agents(self):
        """Stop all agents in the system."""
        for agent_id in list(self.agents.keys()):
            await self.stop_agent(agent_id)
    
    async def run(self):
        """Run the trading system."""
        logger.info("Starting Multi-Agent Trading System")
        
        try:
            # Start all agents
            await self.start_all_agents()
            
            # Wait for shutdown signal
            await self._shutdown_event.wait()
            
        except Exception as e:
            logger.error(f"Error running trading system: {e}")
        finally:
            # Cleanup
            await self.stop_all_agents()
            logger.info("Multi-Agent Trading System stopped")
    
    def shutdown(self):
        """Signal shutdown."""
        self._shutdown_event.set()


# Global orchestrator instance
orchestrator = TradingSystemOrchestrator()


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, shutting down...")
    orchestrator.shutdown()


async def main():
    """Main entry point."""
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Check if running as single agent
    agent_type_str = os.getenv("AGENT_TYPE")
    agent_id = os.getenv("AGENT_ID")
    
    if agent_type_str and agent_id:
        # Run single agent
        try:
            agent_type = AgentType(agent_type_str)
            await orchestrator.start_agent(agent_type, agent_id)
            await orchestrator._shutdown_event.wait()
        except ValueError:
            logger.error(f"Invalid agent type: {agent_type_str}")
            sys.exit(1)
    else:
        # Run full system
        await orchestrator.run()


if __name__ == "__main__":
    asyncio.run(main())
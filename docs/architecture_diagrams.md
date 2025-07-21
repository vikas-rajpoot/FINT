# Multi-Agent Trading System Architecture Diagrams

## 1. Class Diagram - Core Models and Data Structures

```mermaid
classDiagram
    class TradeAction {
        <<enumeration>>
        BUY
        SELL
        HOLD
    }

    class MessageType {
        <<enumeration>>
        MARKET_DATA
        TRADING_PROPOSAL
        VOTE_REQUEST
        VOTE_RESPONSE
        CONSENSUS_RESULT
        EXECUTION_ORDER
        EXECUTION_RESULT
        HEALTH_CHECK
        SYSTEM_ALERT
    }

    class AgentType {
        <<enumeration>>
        MARKET_ANALYST
        RISK_MANAGER
        PORTFOLIO_OPTIMIZER
        EXECUTION_AGENT
    }

    class MarketData {
        -string symbol
        -datetime timestamp
        -float price
        -int volume
        -float bid
        -float ask
        -Dict~string,float~ technical_indicators
        +to_dict() Dict~string,Any~
        +from_dict(data) MarketData
    }

    class RiskMetrics {
        -float var_95
        -float cvar_95
        -float sharpe_ratio
        -float max_drawdown
        -float volatility
        +to_dict() Dict~string,Any~
    }

    class TradingProposal {
        -string proposal_id
        -string symbol
        -TradeAction action
        -int quantity
        -float price_target
        -string rationale
        -float confidence
        -RiskMetrics risk_metrics
        -datetime timestamp
        +to_dict() Dict~string,Any~
    }

    class Vote {
        -string agent_id
        -string proposal_id
        -int score
        -float confidence
        -string rationale
        -datetime timestamp
        +to_dict() Dict~string,Any~
    }

    class Experience {
        -string experience_id
        -Dict~string,Any~ state
        -Dict~string,Any~ action
        -float reward
        -Dict~string,Any~ next_state
        -Dict~string,float~ agent_contributions
        -datetime timestamp
        +to_dict() Dict~string,Any~
    }

    class Message {
        -string message_id
        -MessageType message_type
        -string sender_id
        -string recipient_id
        -Dict~string,Any~ payload
        -datetime timestamp
        -string correlation_id
        +to_dict() Dict~string,Any~
        +from_dict(data) Message
    }

    class AgentResponse {
        -string response_id
        -string original_message_id
        -string agent_id
        -bool success
        -Dict~string,Any~ result
        -string error_message
        -float processing_time_ms
        -datetime timestamp
        +to_dict() Dict~string,Any~
    }

    class AgentConfig {
        -string agent_id
        -AgentType agent_type
        -string model_version
        -Dict~string,Any~ parameters
        -Dict~string,Any~ resource_limits
        -Dict~string,Any~ message_queue_config
        -int health_check_interval
        -int max_retry_attempts
        -int timeout_seconds
        +to_dict() Dict~string,Any~
    }

    class SystemConfig {
        -string environment
        -float consensus_threshold
        -float max_disagreement_variance
        -string message_broker_url
        -string database_url
        -string time_series_db_url
        -string model_registry_url
        -bool monitoring_enabled
        -string logging_level
        +to_dict() Dict~string,Any~
    }

    TradingProposal --> TradeAction
    TradingProposal --> RiskMetrics
    Vote --> TradingProposal : votes_on
    Message --> MessageType
    AgentConfig --> AgentType
```

## 2. Class Diagram - Agent Architecture

```mermaid
classDiagram
    class BaseAgent {
        <<abstract>>
        -string agent_id
        -AgentConfig config
        -Logger logger
        -MessageBus message_bus
        -ModelManager model_manager
        -MetricsCollector metrics_collector
        -bool _is_running
        -string _health_status
        -datetime _last_heartbeat
        +start() void
        +stop() void
        +process_message(message)* AgentResponse
        +vote_on_decision(proposal)* Vote
        +update_experience(experience) void
        +get_health_status() Dict~string,Any~
        -_handle_message(message) AgentResponse
        -_health_check_loop() void
        -_perform_health_check() void
    }

    class MarketAnalystAgent {
        +process_message(message) AgentResponse
        +vote_on_decision(proposal) Vote
        -analyze_technical_indicators() Dict
        -analyze_sentiment() float
        -detect_patterns() List
    }

    class RiskManagerAgent {
        +process_message(message) AgentResponse
        +vote_on_decision(proposal) Vote
        -calculate_position_size() int
        -assess_portfolio_risk() RiskMetrics
        -check_correlation() float
    }

    class PortfolioOptimizerAgent {
        +process_message(message) AgentResponse
        +vote_on_decision(proposal) Vote
        -optimize_allocation() Dict
        -rebalance_portfolio() List
        -calculate_diversification() float
    }

    class ExecutionAgent {
        +process_message(message) AgentResponse
        +vote_on_decision(proposal) Vote
        -execute_trade() Dict
        -optimize_timing() datetime
        -minimize_slippage() float
    }

    BaseAgent <|-- MarketAnalystAgent
    BaseAgent <|-- RiskManagerAgent
    BaseAgent <|-- PortfolioOptimizerAgent
    BaseAgent <|-- ExecutionAgent

    BaseAgent --> AgentConfig : uses
    BaseAgent --> MessageBus : uses
    BaseAgent --> ModelManager : uses
    BaseAgent --> MetricsCollector : uses
```

## 3. Class Diagram - Service Layer

```mermaid
classDiagram
    class MessageBus {
        -Dict~string,Any~ config
        -Logger logger
        -Dict~string,Callable~ _subscribers
        -bool _is_connected
        +connect() void
        +disconnect() void
        +publish(channel, message) bool
        +subscribe(channel, handler) void
        +unsubscribe(channel) void
        +is_connected() bool
    }

    class ModelManager {
        -string agent_id
        -string model_version
        -Logger logger
        -Any _current_model
        +load_model(model_path) bool
        +update_model(new_version) bool
        +get_model_info() Dict~string,Any~
        +cleanup() void
    }

    class MetricsCollector {
        -string agent_id
        -Logger logger
        -List _metrics_buffer
        +record_message_processed(type, time, success) void
        +record_health_check(status) void
        +record_vote_cast(proposal_id, score, confidence) void
        +flush() void
        +get_metrics_summary() Dict~string,Any~
    }

    class ConsensusResult {
        -string decision
        -float confidence
        -string reason
        -datetime timestamp
    }

    class ConsensusEngine {
        -float threshold
        -float max_variance
        -Logger logger
        +calculate_consensus(votes) ConsensusResult
        +log_decision(proposal, votes, result) Dict~string,Any~
    }

    class DatabaseManager {
        -Dict~string,Any~ config
        -Logger logger
        -Any _connection_pool
        +connect() bool
        +disconnect() void
        +execute_query(query, params) Any
    }

    ConsensusEngine --> ConsensusResult : creates
    ConsensusEngine --> Vote : processes
    ConsensusEngine --> TradingProposal : evaluates
```

## 4. Component Diagram - System Architecture

```mermaid
graph TB
    subgraph "External Systems"
        MarketData[Market Data Feeds]
        Brokers[Trading Brokers]
        NewsAPI[News & Social Media APIs]
    end

    subgraph "Infrastructure Layer"
        Redis[(Redis<br/>Message Broker)]
        PostgreSQL[(PostgreSQL<br/>Relational Data)]
        InfluxDB[(InfluxDB<br/>Time Series Data)]
        MLRegistry[ML Model Registry]
    end

    subgraph "Agent Layer"
        MA[Market Analyst<br/>Agent]
        RM[Risk Manager<br/>Agent]
        PO[Portfolio Optimizer<br/>Agent]
        EA[Execution Agent]
    end

    subgraph "Service Layer"
        MessageBus[Message Bus]
        ConsensusEngine[Consensus Engine]
        ModelManager[Model Manager]
        MetricsCollector[Metrics Collector]
    end

    subgraph "Orchestration Layer"
        Orchestrator[Trading System<br/>Orchestrator]
        HealthMonitor[Health Monitor]
        ConfigManager[Config Manager]
    end

    %% External connections
    MarketData --> MA
    NewsAPI --> MA
    EA --> Brokers

    %% Infrastructure connections
    MessageBus --> Redis
    ModelManager --> MLRegistry
    MetricsCollector --> InfluxDB
    DatabaseManager --> PostgreSQL

    %% Agent to Service connections
    MA --> MessageBus
    RM --> MessageBus
    PO --> MessageBus
    EA --> MessageBus

    MA --> ModelManager
    RM --> ModelManager
    PO --> ModelManager
    EA --> ModelManager

    MA --> MetricsCollector
    RM --> MetricsCollector
    PO --> MetricsCollector
    EA --> MetricsCollector

    %% Service interactions
    MessageBus --> ConsensusEngine
    ConsensusEngine --> DatabaseManager

    %% Orchestration
    Orchestrator --> MA
    Orchestrator --> RM
    Orchestrator --> PO
    Orchestrator --> EA
    
    HealthMonitor --> MA
    HealthMonitor --> RM
    HealthMonitor --> PO
    HealthMonitor --> EA

    style MA fill:#e1f5fe
    style RM fill:#fff3e0
    style PO fill:#f3e5f5
    style EA fill:#e8f5e8
```

## 5. Sequence Diagram - Trading Decision Flow

```mermaid
sequenceDiagram
    participant MD as Market Data
    participant MA as Market Analyst
    participant MB as Message Bus
    participant RM as Risk Manager
    participant PO as Portfolio Optimizer
    participant CE as Consensus Engine
    participant EA as Execution Agent
    participant DB as Database

    MD->>MA: New market data
    MA->>MA: Analyze technical indicators
    MA->>MA: Analyze sentiment
    MA->>MB: Publish trading proposal
    
    MB->>RM: Forward proposal
    MB->>PO: Forward proposal
    MB->>EA: Forward proposal
    
    par Parallel Voting
        RM->>RM: Assess risk metrics
        RM->>MB: Submit vote
    and
        PO->>PO: Optimize portfolio impact
        PO->>MB: Submit vote
    and
        EA->>EA: Evaluate execution feasibility
        EA->>MB: Submit vote
    end
    
    MB->>CE: Collect all votes
    CE->>CE: Calculate consensus
    
    alt Consensus Reached
        CE->>MB: Execute decision
        MB->>EA: Execute trade order
        EA->>EA: Execute trade
        EA->>DB: Log execution result
        EA->>MB: Confirm execution
    else Manual Review Required
        CE->>DB: Log for manual review
    else Rejected
        CE->>DB: Log rejection reason
    end
    
    MB->>MA: Update experience
    MB->>RM: Update experience
    MB->>PO: Update experience
```

## 6. Deployment Diagram - Docker Architecture

```mermaid
graph TB
    subgraph "Docker Host"
        subgraph "Application Containers"
            MA_C[market-analyst<br/>Container]
            RM_C[risk-manager<br/>Container]
            PO_C[portfolio-optimizer<br/>Container]
            EA_C[execution-agent<br/>Container]
        end
        
        subgraph "Infrastructure Containers"
            Redis_C[Redis<br/>Container]
            PG_C[PostgreSQL<br/>Container]
            Influx_C[InfluxDB<br/>Container]
        end
        
        subgraph "Monitoring"
            Prometheus[Prometheus<br/>Container]
            Grafana[Grafana<br/>Container]
        end
    end

    subgraph "External Services"
        MarketAPI[Market Data APIs]
        BrokerAPI[Broker APIs]
        MLFlow[MLFlow Registry]
    end

    %% Internal connections
    MA_C --> Redis_C
    RM_C --> Redis_C
    PO_C --> Redis_C
    EA_C --> Redis_C

    MA_C --> PG_C
    RM_C --> PG_C
    PO_C --> PG_C
    EA_C --> PG_C

    MA_C --> Influx_C
    RM_C --> Influx_C
    PO_C --> Influx_C
    EA_C --> Influx_C

    %% External connections
    MA_C --> MarketAPI
    EA_C --> BrokerAPI
    MA_C --> MLFlow
    RM_C --> MLFlow
    PO_C --> MLFlow
    EA_C --> MLFlow

    %% Monitoring
    Prometheus --> MA_C
    Prometheus --> RM_C
    Prometheus --> PO_C
    Prometheus --> EA_C
    Grafana --> Prometheus

    style MA_C fill:#e1f5fe
    style RM_C fill:#fff3e0
    style PO_C fill:#f3e5f5
    style EA_C fill:#e8f5e8
```

## 7. Data Flow Diagram - Information Architecture

```mermaid
flowchart TD
    subgraph "Data Sources"
        A[Market Data Feeds]
        B[News & Social Media]
        C[Economic Indicators]
    end

    subgraph "Data Processing"
        D[Market Analyst Agent]
        E[Technical Analysis]
        F[Sentiment Analysis]
    end

    subgraph "Decision Making"
        G[Trading Proposal]
        H[Risk Assessment]
        I[Portfolio Impact]
        J[Execution Feasibility]
        K[Consensus Engine]
    end

    subgraph "Execution"
        L[Trade Execution]
        M[Order Management]
        N[Broker Integration]
    end

    subgraph "Learning & Feedback"
        O[Experience Storage]
        P[Model Updates]
        Q[Performance Metrics]
    end

    A --> D
    B --> D
    C --> D
    
    D --> E
    D --> F
    E --> G
    F --> G
    
    G --> H
    G --> I
    G --> J
    H --> K
    I --> K
    J --> K
    
    K --> L
    L --> M
    M --> N
    
    L --> O
    O --> P
    P --> D
    N --> Q
    Q --> O

    style G fill:#ffeb3b
    style K fill:#ff9800
    style L fill:#4caf50
```

## Architecture Summary

### Key Design Patterns Used:

1. **Observer Pattern**: Message Bus for agent communication
2. **Strategy Pattern**: Different agent types with specialized algorithms
3. **Template Method**: BaseAgent with abstract methods for specialization
4. **Factory Pattern**: Agent creation in the orchestrator
5. **Singleton Pattern**: System-wide configuration management

### Component Relationships:

1. **Inheritance**: All agent types inherit from BaseAgent
2. **Composition**: Agents compose services (MessageBus, ModelManager, MetricsCollector)
3. **Aggregation**: ConsensusEngine aggregates votes from multiple agents
4. **Dependency**: Agents depend on infrastructure services
5. **Association**: Messages associate agents through sender/recipient relationships

### Data Flow Characteristics:

- **Asynchronous**: All agent communication is async via message bus
- **Event-driven**: Agents react to market data and proposal events
- **Consensus-based**: Decisions require agreement from multiple agents
- **Auditable**: All decisions and votes are logged for compliance
- **Scalable**: Horizontal scaling through container orchestration
# UniV4 Backtesting with uAgents

This project implements a decentralized backtesting system for Uniswap V4 strategies using fetch.ai's uAgents framework and The Graph for historical data.

## Architecture

The system consists of two main agents:

1. **DataAgent** (`data_agent.py`): Fetches historical pool events from The Graph
2. **BacktestAgent** (`backtest_agent.py`): Runs Solidity backtests using the fetched data

## Key Features

- **The Graph Integration**: Replaces expensive archive node access with efficient subgraph queries
- **Deterministic Backtesting**: Uses proven Solidity framework for reproducible results
- **Agent Communication**: Decoupled architecture with message-based communication
- **Async Operations**: Non-blocking data fetching and processing

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set environment variables:
```bash
export UNI_RPC_URL="your_rpc_url_here"
```

3. Ensure you have Foundry installed for Solidity backtesting:
```bash
curl -L https://foundry.paradigm.xyz | bash
foundryup
```

## Usage

### Run Both Agents

```bash
python main.py
```

This starts both agents:
- DataAgent on port 8001
- BacktestAgent on port 8002

### Run Individual Agents

```bash
# Run DataAgent only
python data_agent.py

# Run BacktestAgent only
python backtest_agent.py
```

### Send Backtest Request

You can send requests to the BacktestAgent programmatically:

```python
from uagents import Agent, Context
from backtestAgentExample.backtest_agent import BacktestRequest

# Create your own agent to send requests
client = Agent(name="client", port=8003)


@client.on_startup()
async def send_backtest_request(ctx: Context):
    request = BacktestRequest(
        pool="0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640",  # ETH/USDC 0.05%
        start=1640995200,  # Jan 1, 2022
        end=1672531200,  # Jan 1, 2023
        strategy_params={}
    )
    await ctx.send("agent_address_here", request)


client.run()
```

## Message Flow

1. **BacktestRequest** → BacktestAgent
2. **FetchEventsRequest** → DataAgent
3. **EventsResponse** → BacktestAgent
4. **BacktestResults** → Original requester

## Data Format

The system uses the following event format (compatible with UniV4Backtester):

```json
{
  "amount": 34399999543676,
  "amount0": 24146777,
  "amount1": 2874562901206670000,
  "eventType": 0,
  "tickLower": 253320,
  "tickUpper": 264600,
  "unixTimestamp": 1620158974
}
```

## Event Types

- `eventType: 0` - Mint/Burn operations
- `eventType: 1` - Swap operations

## Configuration

### DataAgent Configuration

- **Port**: 8001
- **Endpoint**: `http://127.0.0.1:8001/submit`
- **The Graph Endpoint**: `https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3`

### BacktestAgent Configuration

- **Port**: 8002
- **Endpoint**: `http://127.0.0.1:8002/submit`
- **Data Directory**: `src/data/`
- **Events File**: `src/data/pool-events.json`

## Error Handling

Both agents include comprehensive error handling:

- Network connectivity issues
- GraphQL query errors
- Solidity compilation/execution errors
- Invalid message formats
- Missing dependencies

## Extending the System

### Adding New Data Sources

Extend the `TheGraphClient` class in `data_agent.py` to support additional subgraphs or data sources.

### Custom Backtesting Logic

Modify the `SolidityBacktester` class in `backtest_agent.py` to integrate with different Solidity frameworks.

### Additional Metrics

Extend the `BacktestResults` model to include more performance metrics.

## Troubleshooting

### Common Issues

1. **Forge not found**: Install Foundry
2. **RPC URL not set**: Set `UNI_RPC_URL` environment variable
3. **Port already in use**: Change ports in agent configurations
4. **GraphQL timeout**: Reduce query batch size or add retry logic

### Debugging

Enable debug logging by setting:
```bash
export UAGENTS_LOG_LEVEL=DEBUG
```

## License

This project is part of the uAgents framework and follows the same licensing terms.
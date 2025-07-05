# Data Layer - uAgent Implementation

This directory contains the Data Agent implementation that fetches pool events from The Graph subgraph and provides structured data for backtesting.

## Architecture

```
┌─────────────────┐    GraphQL    ┌─────────────────┐
│   Data Agent    │ ──────────────▶│   The Graph     │
│   (uAgents)     │                │   Subgraph      │
└─────────────────┘                └─────────────────┘
         │
         │ FetchEventsRequest
         │ EventsResponse
         │
┌─────────────────┐
│ Enhanced Agent  │
│ / Client        │
└─────────────────┘
```

## Files

- `data_agent.py` - Main data agent implementation
- `data_client.py` - Example client for testing
- `README.md` - This documentation

## Features

### Data Agent (`data_agent.py`)

- **GraphQL Integration**: Fetches data from The Graph Protocol
- **Event Normalization**: Converts swaps, mints, and burns into structured format
- **Auto Pool Selection**: Automatically selects top pool by volume if not specified
- **Rate Limiting**: Built-in rate limiting to avoid API overload
- **Mailbox Support**: Enables Agentverse integration
- **Error Handling**: Comprehensive error handling and logging
- **💬 Chat Interface**: Natural language data requests via chat protocol

### Message Models

#### `FetchEventsRequest`
```python
{
    "pool_id": "0x123...",  # Optional - auto-selects if not provided
    "minutes_back": 10080,  # Minutes back to fetch (default: 7 days)
    "max_events": 1000      # Maximum events to fetch
}
```

#### `EventsResponse`
```python
{
    "events": [PoolEvent...],  # List of normalized events
    "pool_id": "0x123...",     # Pool ID used
    "pool_info": {...},        # Pool metadata
    "total_events": 150,       # Total events count
    "time_range": {...}        # Time range information
}
```

#### `PoolEvent`
```python
{
    "amount": 1000.0,         # USD amount
    "amount0": 0.5,           # Token0 amount
    "amount1": 1500.0,        # Token1 amount
    "eventType": 1,           # 0=liquidity, 1=swap
    "unixTimestamp": 1234567, # Unix timestamp
    "tickLower": 100,         # Lower tick (liquidity events)
    "tickUpper": 200          # Upper tick (liquidity events)
}
```

## Chat Interface

The data agent supports natural language chat for easy data requests:

### Chat Commands

#### Greetings & Help
- `"Hello"` - Introduction and capabilities
- `"Help"` - Show available commands and examples
- `"What can you do?"` - Feature overview

#### Data Requests
- `"Get pool events from last 24 hours"` - Fetch recent events
- `"Show me swap data from past week"` - Weekly swap data
- `"Fetch 200 events from last 3 days"` - Custom event count and timeframe
- `"Get recent liquidity events"` - Liquidity-focused request

#### Time Specifications
- **Minutes**: `"last 30 minutes"`, `"past 60 minutes"`
- **Hours**: `"last 2 hours"`, `"past 6 hours"` 
- **Days**: `"last 3 days"`, `"past 7 days"`
- **Weeks**: `"past week"`, `"last 2 weeks"`

#### Event Limits
- `"Get 500 events from last day"` - Specify event count
- `"Show me 50 swap events"` - Limit and filter events

### Chat Response Format

```
✅ Data Fetch Complete!

🏊 Pool: USDC/WETH
💰 Volume: $6,122,899,442.41
📊 Events Found: 150
⏰ Time Range: 2024-07-04 12:00 UTC to 2024-07-05 12:00 UTC

📈 Event Breakdown:
• Swaps: 75
• Liquidity Events: 75

📝 Sample Events (first 3):
1. 🔄 Swap - $1500.50 - 2024-07-05 11:45 UTC
2. 💧 Liquidity - $2300.25 - 2024-07-05 11:40 UTC
3. 🔄 Swap - $850.75 - 2024-07-05 11:35 UTC

💡 Next Steps: This data can be used for backtesting analysis!
```

## Usage

### 1. Start Data Agent

```bash
cd datalayer
python data_agent.py
```

The agent will:
- Connect to The Graph subgraph
- Test the connection
- Display its address for client connections
- Listen for fetch requests

### 2. Test with Chat Client

```bash
# In another terminal
python chat_client.py
```

**Important**: Update `DATA_AGENT_ADDRESS` in `chat_client.py` with the actual address from step 1.

The chat client will automatically test various natural language requests.

### 3. Use Data Client (Structured Messages)

```bash
# In another terminal  
python data_client.py
```

**Important**: Update `DATA_AGENT_ADDRESS` in `data_client.py` with the actual address from step 1.

### 3. Integration with Enhanced Agent

```python
from data_agent import FetchEventsRequest, EventsResponse

# In your enhanced agent
@protocol.on_message(BacktestRequest)
async def handle_backtest(ctx: Context, sender: str, msg: BacktestRequest):
    # Request data from data agent
    data_request = FetchEventsRequest(
        pool_id=msg.pool,
        minutes_back=calculate_minutes_back(msg.start, msg.end),
        max_events=1000
    )
    
    await ctx.send(DATA_AGENT_ADDRESS, data_request)
```

## Configuration

### The Graph Configuration

The data agent is configured to use:
- **Subgraph ID**: `FqsRcH1XqSjqVx9GRTvEJe959aCbKrcyGgDWBrUkG24g`
- **API Key**: `bfb3f65c79e8d0f6e7dd728b91d46600`
- **Endpoint**: `https://gateway.thegraph.com/api/{API_KEY}/subgraphs/id/{SUBGRAPH_ID}`

### Rate Limiting

- **Window**: 60 minutes
- **Max Requests**: 100 per hour
- **Automatic retry**: Built-in exponential backoff

## Event Types

The data agent normalizes different event types:

| Event Type | Description | Fields |
|------------|-------------|---------|
| **Swaps** (1) | Token swaps | `amount`, `amount0`, `amount1`, `timestamp` |
| **Mints** (0) | Liquidity additions | `amount`, `amount0`, `amount1`, `tickLower`, `tickUpper`, `timestamp` |
| **Burns** (0) | Liquidity removals | `amount`, `amount0`, `amount1`, `tickLower`, `tickUpper`, `timestamp` |

## Error Handling

The data agent provides detailed error responses:

```python
{
    "error_type": "fetch_error",
    "message": "Failed to fetch pool events: ...",
    "details": {
        "pool_id": "0x123...",
        "minutes_back": 1440,
        "max_events": 1000
    }
}
```

## Integration Examples

### With Enhanced Backtest Agent

```python
# In enhanced_agent.py
DATA_AGENT_ADDRESS = "agent1q..."

@protocol.on_message(BacktestRequest)
async def handle_backtest_request(ctx: Context, sender: str, msg: BacktestRequest):
    # Request data
    data_request = FetchEventsRequest(
        pool_id=msg.pool,
        minutes_back=calculate_time_range(msg.start, msg.end)
    )
    
    # Store original request context
    ctx.storage.set(f"backtest_{sender}", {
        "original_request": msg.dict(),
        "sender": sender
    })
    
    await ctx.send(DATA_AGENT_ADDRESS, data_request)

@protocol.on_message(EventsResponse)
async def handle_events_response(ctx: Context, sender: str, msg: EventsResponse):
    # Process events and continue with backtest
    # ... backtest logic using msg.events
```

### Direct REST API

```python
# For direct integration without agents
import requests

response = requests.post(
    "http://localhost:8000/submit",
    json={
        "pool_id": "0x123...",
        "minutes_back": 1440,
        "max_events": 500
    }
)
```

## Monitoring

The data agent provides comprehensive logging:

```
🚀 Data Agent started: agent1q...
📊 Connected to The Graph subgraph: FqsRcH1X...
✅ Successfully connected to The Graph
📈 Top pool: USDC/ETH (ID: 0x123...)
📥 Received fetch events request from agent1q...
📊 Fetching events for pool 0x123...
✅ Successfully fetched 150 events
📈 Event breakdown: 75 swaps, 75 liquidity events
📤 Sent 150 events to agent1q...
```

## Performance

- **Concurrent Requests**: Supports multiple simultaneous requests
- **Caching**: Built-in result caching to reduce API calls
- **Rate Limiting**: Prevents API overload
- **Error Recovery**: Automatic retry with exponential backoff

## Next Steps

1. **Caching Layer**: Add persistent caching for frequently requested data
2. **Multiple Subgraphs**: Support for different blockchain networks
3. **Real-time Updates**: WebSocket support for live data
4. **Advanced Filtering**: More granular event filtering options
5. **Batch Processing**: Support for multiple pool requests in one call
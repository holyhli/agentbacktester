# 🤖 Backtest Chat Interface

This document explains how to use the natural language chat interface for the UniV4 Backtesting system.

## Overview

The chat interface allows you to interact with the backtesting system using natural language commands instead of complex API calls. You can request backtests, check status, and view results using simple conversational commands.

## Architecture

The chat system consists of three main components:

1. **ChatAgent** (`chat_agent.py`) - The main chat interface that processes natural language
2. **ChatClient** (`chat_client.py`) - A simple client for testing the chat interface
3. **Integration** - The chat agent communicates with DataAgent and BacktestAgent

## Quick Start

### 1. Start the System

```bash
# Start all agents (including chat)
python3 main.py
```

This will start:
- DataAgent on port 8001
- BacktestAgent on port 8002  
- ChatAgent on port 8003

### 2. Get the Chat Agent Address

When you start the system, look for the ChatAgent address in the logs:
```
Chat Agent Address: test-agent://agent1q...
```

### 3. Update Client Configuration

Edit `chat_client.py` and update the `CHAT_AGENT_ADDRESS` with the actual address from step 2.

### 4. Start Chatting

```bash
# Interactive chat
python3 chat_client.py

# Or run a demo
python3 chat_client.py demo
```

## Chat Commands

### Help Commands
- `help` - Show available commands
- `what can you do` - Show capabilities
- `commands` - List all commands

### Status Commands
- `status` - Check system status
- `running` - Check if agents are active

### Backtest Commands
- `backtest USDC-ETH for 1 week` - Run backtest on USDC-ETH pool
- `test WBTC-ETH for 1 month` - Run backtest on WBTC-ETH pool
- `simulate 0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640 for 3 months` - Use specific pool address
- `run backtest on USDC-WETH for last 5 days` - Custom time period

### Results Commands
- `results` - Show latest backtest results
- `show results` - Display recent results
- `latest` - Get most recent backtest
- `last backtest` - Show last completed backtest

## Supported Pools

The system recognizes these pool names:
- `USDC-ETH` → 0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640
- `WBTC-ETH` → 0xcbcdf9626bc03e24f779434178a73a0b4bad62ed
- `USDC-WETH` → 0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640
- `WBTC-WETH` → 0xcbcdf9626bc03e24f779434178a73a0b4bad62ed

You can also use any pool address directly (0x...).

## Time Periods

### Predefined Periods
- `1 day`, `1 week`, `1 month`
- `3 months`, `6 months`, `1 year`

### Custom Periods
- `last 5 days`
- `last 2 weeks`
- `last 3 months`

If no time period is specified, defaults to the last 30 days.

## Example Conversations

### Basic Backtest
```
You: help
Bot: 🤖 Backtest Chat Agent Help
     [Shows available commands]

You: backtest USDC-ETH for 1 week
Bot: 🚀 Backtest Started!
     Pool: USDC-ETH (0x88e6a0c2...)
     Period: 2024-01-01 to 2024-01-08
     [Processing message]

Bot: 🎉 Backtest Complete!
     📊 Results Summary:
     • PnL: 0.0234 (+2.34%)
     • Sharpe Ratio: 1.45
     [Full results]
```

### Status Check
```
You: status
Bot: 📊 System Status
     ✅ Chat Agent: Online
     ✅ Data Agent: Connected
     ✅ Backtest Agent: Connected
     ✅ Storage System: Active
```

### View Results
```
You: results
Bot: 📈 Latest Backtest Results
     Pool: 0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640
     Period: 2024-01-01 to 2024-01-08
     [Detailed results]
```

## Advanced Usage

### Using Mailbox (Agentverse)

The ChatAgent is configured with `mailbox=True`, which means it can connect to the Agentverse for enhanced messaging capabilities.

To use with Agentverse:
1. The agent will automatically register with Agentverse
2. You can interact with it through the Agentverse interface
3. Or use the local client as shown above

### Custom Integration

You can integrate the chat interface into your own applications by:

1. **Direct Agent Communication**:
```python
from uagents import Agent, Context
from chat_agent import ChatMessage, ChatResponse

# Send message to chat agent
chat_msg = ChatMessage(message="backtest USDC-ETH for 1 week", user_id="your_app")
await ctx.send(CHAT_AGENT_ADDRESS, chat_msg)
```

2. **REST API** (if enabled):
The agents expose REST endpoints for integration with web applications.

## Troubleshooting

### Common Issues

1. **Agent Address Not Found**
   - Make sure all agents are running
   - Check the console output for the correct addresses
   - Update the addresses in client code

2. **No Response from Chat Agent**
   - Verify the chat agent is running on port 8003
   - Check that the agent address is correct
   - Look at the logs for error messages

3. **Backtest Fails**
   - Ensure the pool address is valid
   - Check that the time period is reasonable
   - Verify the Foundry installation (for Solidity backtests)

4. **Connection Issues**
   - Make sure all required ports are available (8001, 8002, 8003, 8004)
   - Check firewall settings
   - Verify network connectivity

### Debug Mode

To enable debug logging:
```python
# In chat_agent.py, change log_level
chat_agent = Agent(
    name="BacktestChatAgent",
    # ... other params
    log_level="DEBUG"
)
```

### Logs Location

Agent activity is logged to:
- `backtest_data/ChatAgent_activity.log`
- `backtest_data/DataAgent_activity.log`
- `backtest_data/BacktestAgent_activity.log`

## API Reference

### Message Models

```python
class ChatMessage(Model):
    message: str
    user_id: str = "user"

class ChatResponse(Model):
    message: str
    agent_name: str = "BacktestChatAgent"
```

### Command Parser

The `ChatCommandParser` class handles natural language processing:
- Extracts pool information
- Parses time periods
- Identifies command types
- Provides error handling

## Future Enhancements

Planned improvements:
- [ ] Support for more complex trading strategies
- [ ] Integration with AI models for better NLP
- [ ] Web-based chat interface
- [ ] Voice commands
- [ ] Multi-language support
- [ ] Advanced analytics commands
- [ ] Portfolio management features

## Contributing

To extend the chat interface:

1. Add new command patterns in `ChatCommandParser`
2. Implement handlers in `chat_agent.py`
3. Update the help text
4. Add tests for new functionality

## Support

For issues or questions:
1. Check the logs for error messages
2. Review this documentation
3. Check the main project README
4. Open an issue in the project repository

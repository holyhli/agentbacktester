from uagents import Agent, Context, Protocol
from data_agent import FetchEventsRequest, EventsResponse, DataAgentError
import asyncio

# Client Agent
client_agent = Agent(
    name="data_client",
    seed="data_client_seed_2024",
    port=8003,  # Use different port
    endpoint=["http://localhost:8003/submit"],
    mailbox=False
)

# Data Agent Address (will be set after data_agent starts)
DATA_AGENT_ADDRESS = "agent1qd8p7zqn8x0gm4l6k5j3f2a9c7b4n0t8r6s9m2x5y1z3w4v"  # Replace with actual address

# Protocol for communication
client_protocol = Protocol(name="data_client_protocol", version="1.0")

@client_agent.on_event("startup")
async def startup(ctx: Context):
    ctx.logger.info(f"🚀 Data Client started: {client_agent.address}")
    ctx.logger.info(f"🎯 Will request data from: {DATA_AGENT_ADDRESS}")
    
    # Wait a moment then request data
    await asyncio.sleep(2)
    
    # Request pool events
    request = FetchEventsRequest(
        pool_id=None,  # Auto-select top pool
        minutes_back=1440,  # Last 24 hours
        max_events=100
    )
    
    ctx.logger.info(f"📤 Requesting pool events...")
    await ctx.send(DATA_AGENT_ADDRESS, request)

@client_protocol.on_message(model=EventsResponse)
async def handle_events_response(ctx: Context, sender: str, msg: EventsResponse):
    ctx.logger.info(f"📨 Received {msg.total_events} events from data agent")
    ctx.logger.info(f"📊 Pool: {msg.pool_id[:10]}...")
    ctx.logger.info(f"⏰ Time range: {msg.time_range['formatted_range']}")
    
    # Process events
    swap_events = [e for e in msg.events if e.eventType == 1]
    liquidity_events = [e for e in msg.events if e.eventType == 0]
    
    ctx.logger.info(f"📈 Event breakdown:")
    ctx.logger.info(f"   - Swaps: {len(swap_events)}")
    ctx.logger.info(f"   - Liquidity: {len(liquidity_events)}")
    
    # Show first few events
    if msg.events:
        ctx.logger.info(f"📝 First 3 events:")
        for i, event in enumerate(msg.events[:3]):
            ctx.logger.info(f"   {i+1}. Type: {event.eventType}, Amount: {event.amount}, Timestamp: {event.unixTimestamp}")

@client_protocol.on_message(model=DataAgentError)
async def handle_error(ctx: Context, sender: str, msg: DataAgentError):
    ctx.logger.error(f"❌ Data agent error: {msg.error_type}")
    ctx.logger.error(f"   Message: {msg.message}")
    ctx.logger.error(f"   Details: {msg.details}")

client_agent.include(client_protocol, publish_manifest=True)

if __name__ == "__main__":
    print("""
🤖 Starting Data Client...

This client will:
1. Start up and connect to the data agent
2. Request pool events from the last 24 hours
3. Display the received events and statistics

📍 Make sure to:
1. Start the data_agent.py first
2. Copy the data agent's address from its logs
3. Update DATA_AGENT_ADDRESS in this file
4. Run this client

🛑 Stop with Ctrl+C
    """)
    client_agent.run()
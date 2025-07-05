import requests
import json
import re
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from enum import Enum
from uuid import uuid4

from uagents import Agent, Context, Model, Field, Protocol
from uagents.experimental.quota import QuotaProtocol, RateLimit
from uagents_core.models import ErrorMessage
from uagents_core.contrib.protocols.chat import (
    ChatMessage, ChatAcknowledgement, TextContent, EndSessionContent, chat_protocol_spec
)


# Data Agent Configuration
SUBGRAPH_ID = "FqsRcH1XqSjqVx9GRTvEJe959aCbKrcyGgDWBrUkG24g"
API_KEY = "bfb3f65c79e8d0f6e7dd728b91d46600"
GRAPH_URL = f"https://gateway.thegraph.com/api/{API_KEY}/subgraphs/id/{SUBGRAPH_ID}"

# Create Data Agent
data_agent = Agent(
    name="data_agent",
    seed="data_agent_unique_seed_phrase_2024",
    port=8001,  # Use different port than enhanced agent
    endpoint=["http://localhost:8001/submit"],
    mailbox=True  # Enable mailbox for Agentverse integration
)

# Event types
class EventType(int, Enum):
    LIQUIDITY = 0  # Mints and Burns
    SWAP = 1       # Swaps

# Message Models
class PoolEvent(Model):
    """Individual pool event model"""
    amount: float = Field(description="Event amount")
    amount0: float = Field(description="Token0 amount")
    amount1: float = Field(description="Token1 amount")
    eventType: EventType = Field(description="Event type (0=liquidity, 1=swap)")
    unixTimestamp: int = Field(description="Unix timestamp")
    tickLower: Optional[int] = Field(default=None, description="Lower tick (for liquidity events)")
    tickUpper: Optional[int] = Field(default=None, description="Upper tick (for liquidity events)")

class FetchEventsRequest(Model):
    """Request to fetch events from The Graph"""
    pool_id: Optional[str] = Field(default=None, description="Pool ID (optional - will use top pool if not provided)")
    minutes_back: int = Field(default=10080, description="Minutes back to fetch (default 7 days)")
    max_events: int = Field(default=1000, description="Maximum events to fetch")

class EventsResponse(Model):
    """Response containing fetched events"""
    events: List[PoolEvent] = Field(description="List of pool events")
    pool_id: str = Field(description="Pool ID used")
    pool_info: Dict[str, Any] = Field(description="Pool information")
    total_events: int = Field(description="Total number of events")
    time_range: Dict[str, Any] = Field(description="Time range information")

class DataAgentError(Model):
    """Error response from data agent"""
    error_type: str = Field(description="Type of error")
    message: str = Field(description="Error message")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional error details")

# Rate limiting protocol
data_protocol = QuotaProtocol(
    storage_reference=data_agent.storage,
    name="DataAgent-Protocol",
    version="1.0.0",
    default_rate_limit=RateLimit(window_size_minutes=60, max_requests=100),
)

# Chat protocol
chat_proto = Protocol(spec=chat_protocol_spec)

# Chat helper functions
def create_text_chat(text: str, end_session: bool = False) -> ChatMessage:
    """Create a chat message with text content"""
    content = [TextContent(type="text", text=text)]
    if end_session:
        content.append(EndSessionContent(type="end-session"))
    return ChatMessage(
        timestamp=datetime.now(timezone.utc),
        msg_id=uuid4(),
        content=content,
    )

def parse_data_request(text: str) -> Dict[str, Any]:
    """Parse natural language request for data parameters"""
    text_lower = text.lower()
    
    # Default parameters
    params = {
        "pool_id": None,
        "minutes_back": 1440,  # Default 24 hours
        "max_events": 100
    }
    
    # Extract time period
    if "week" in text_lower or "7 day" in text_lower:
        params["minutes_back"] = 10080  # 7 days
    elif "hour" in text_lower:
        # Look for specific hour numbers
        hour_match = re.search(r'(\d+)\s*hour', text_lower)
        if hour_match:
            hours = int(hour_match.group(1))
            params["minutes_back"] = hours * 60
        else:
            params["minutes_back"] = 60  # 1 hour default
    elif "day" in text_lower:
        day_match = re.search(r'(\d+)\s*day', text_lower)
        if day_match:
            days = int(day_match.group(1))
            params["minutes_back"] = days * 1440
        else:
            params["minutes_back"] = 1440  # 1 day default
    elif "minute" in text_lower:
        min_match = re.search(r'(\d+)\s*minute', text_lower)
        if min_match:
            minutes = int(min_match.group(1))
            params["minutes_back"] = minutes
        else:
            params["minutes_back"] = 60  # 1 hour default
    
    # Extract max events
    if "events" in text_lower:
        event_match = re.search(r'(\d+)\s*events?', text_lower)
        if event_match:
            params["max_events"] = int(event_match.group(1))
    
    # Extract pool information
    if "pool" in text_lower:
        # Look for pool address (0x...)
        pool_match = re.search(r'0x[a-fA-F0-9]{40}', text)
        if pool_match:
            params["pool_id"] = pool_match.group(0)
    
    return params

def is_data_request(text: str) -> bool:
    """Check if the message is a data request"""
    data_keywords = [
        "data", "events", "pool", "swap", "liquidity", "fetch", "get", 
        "show", "find", "search", "query", "history", "recent"
    ]
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in data_keywords)

# Utility functions
def format_time(ts: int) -> str:
    """Format timestamp for logging"""
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')

def format_window_label(minutes: int) -> str:
    """Format time window for logging"""
    if minutes < 60:
        return f"{minutes} Minutes"
    elif minutes < 1440:
        hours = minutes // 60
        return f"{hours} Hours"
    else:
        days = minutes // 1440
        return f"{days} Days"

async def get_top_pool() -> Dict[str, Any]:
    """Get the top pool by volume"""
    query_top_pool = """
    {
      pools(first: 1, orderBy: volumeUSD, orderDirection: desc) {
        id
        token0 { symbol }
        token1 { symbol }
        volumeUSD
      }
    }
    """
    
    try:
        response = requests.post(GRAPH_URL, json={'query': query_top_pool}, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        
        if not data:
            raise ValueError("Empty response from The Graph")
        
        if "errors" in data:
            raise ValueError(f"GraphQL errors: {data['errors']}")
        
        if "data" not in data:
            raise ValueError(f"No 'data' field in response: {data}")
        
        if "pools" not in data["data"]:
            raise ValueError(f"No 'pools' field in data: {data['data']}")
        
        pools = data["data"]["pools"]
        if not pools or len(pools) == 0:
            raise ValueError("No pools found in response")
        
        return pools[0]
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"HTTP request failed: {str(e)}")
    except Exception as e:
        raise Exception(f"Failed to fetch top pool: {str(e)}")

async def fetch_pool_events(pool_id: str, minutes_back: int = 10080, max_events: int = 1000) -> Dict[str, Any]:
    """Fetch events from The Graph for a specific pool"""
    
    # Calculate time cutoff
    now_ts = int(datetime.now(timezone.utc).timestamp())
    ts_cutoff = now_ts - minutes_back * 60
    
    # GraphQL query for events
    query_events = f"""
    {{
      swaps(first: {max_events}, orderBy: timestamp, orderDirection: desc, where: {{
        pool: "{pool_id}", timestamp_gt: {ts_cutoff}
      }}) {{
        id
        amount0
        amount1
        amountUSD
        sender
        recipient
        timestamp
      }}
      mints(first: {max_events}, orderBy: timestamp, orderDirection: desc, where: {{
        pool: "{pool_id}", timestamp_gt: {ts_cutoff}
      }}) {{
        id
        amount
        amount0
        amount1
        sender
        origin
        tickLower
        tickUpper
        timestamp
      }}
      burns(first: {max_events}, orderBy: timestamp, orderDirection: desc, where: {{
        pool: "{pool_id}", timestamp_gt: {ts_cutoff}
      }}) {{
        id
        amount
        amount0
        amount1
        origin
        tickLower
        tickUpper
        timestamp
      }}
    }}
    """
    
    try:
        response = requests.post(GRAPH_URL, json={'query': query_events}, timeout=30)
        response.raise_for_status()
        event_data = response.json()
        
        if "errors" in event_data:
            raise ValueError(f"GraphQL errors: {event_data['errors']}")
        
        # Process and normalize events
        all_events = []
        
        # Process swaps (eventType = 1)
        for swap in event_data["data"].get("swaps", []):
            all_events.append(PoolEvent(
                amount=float(swap.get("amountUSD", 0)),
                amount0=float(swap.get("amount0", 0)),
                amount1=float(swap.get("amount1", 0)),
                eventType=EventType.SWAP,
                unixTimestamp=int(swap["timestamp"])
            ))
        
        # Process mints (eventType = 0)
        for mint in event_data["data"].get("mints", []):
            all_events.append(PoolEvent(
                amount=float(mint.get("amount", 0)),
                amount0=float(mint.get("amount0", 0)),
                amount1=float(mint.get("amount1", 0)),
                eventType=EventType.LIQUIDITY,
                tickLower=int(mint["tickLower"]) if mint.get("tickLower") else None,
                tickUpper=int(mint["tickUpper"]) if mint.get("tickUpper") else None,
                unixTimestamp=int(mint["timestamp"])
            ))
        
        # Process burns (eventType = 0)
        for burn in event_data["data"].get("burns", []):
            all_events.append(PoolEvent(
                amount=float(burn.get("amount", 0)),
                amount0=float(burn.get("amount0", 0)),
                amount1=float(burn.get("amount1", 0)),
                eventType=EventType.LIQUIDITY,
                tickLower=int(burn["tickLower"]) if burn.get("tickLower") else None,
                tickUpper=int(burn["tickUpper"]) if burn.get("tickUpper") else None,
                unixTimestamp=int(burn["timestamp"])
            ))
        
        # Sort events by timestamp
        all_events.sort(key=lambda x: x.unixTimestamp)
        
        return {
            "events": all_events,
            "total_events": len(all_events),
            "time_range": {
                "start_timestamp": ts_cutoff,
                "end_timestamp": now_ts,
                "minutes_back": minutes_back,
                "formatted_range": f"{format_time(ts_cutoff)} to {format_time(now_ts)}"
            }
        }
        
    except Exception as e:
        raise Exception(f"Failed to fetch pool events: {str(e)}")

# Event handlers
@data_agent.on_event("startup")
async def startup(ctx: Context):
    ctx.logger.info(f"🚀 Data Agent started: {data_agent.address}")
    ctx.logger.info(f"📊 Connected to The Graph subgraph: {SUBGRAPH_ID}")
    ctx.logger.info(f"🔗 GraphQL endpoint: {GRAPH_URL}")
    
    # Test connection to The Graph
    try:
        pool = await get_top_pool()
        ctx.logger.info(f"✅ Successfully connected to The Graph")
        ctx.logger.info(f"📈 Top pool: {pool['token0']['symbol']}/{pool['token1']['symbol']} (ID: {pool['id'][:10]}...)")
    except Exception as e:
        ctx.logger.error(f"❌ Failed to connect to The Graph: {e}")

@data_agent.on_event("shutdown")
async def shutdown(ctx: Context):
    ctx.logger.info("🛑 Data Agent shutting down...")

# Main message handler
@data_protocol.on_message(FetchEventsRequest, replies={EventsResponse, DataAgentError})
async def handle_fetch_events(ctx: Context, sender: str, msg: FetchEventsRequest):
    ctx.logger.info(f"📥 Received fetch events request from {sender}")
    ctx.logger.info(f"   Pool ID: {msg.pool_id or 'auto-select'}")
    ctx.logger.info(f"   Time range: {format_window_label(msg.minutes_back)}")
    ctx.logger.info(f"   Max events: {msg.max_events}")
    
    try:
        # Get pool ID if not provided
        pool_id = msg.pool_id
        pool_info = {}
        
        if not pool_id:
            ctx.logger.info("🔍 Auto-selecting top pool by volume...")
            pool = await get_top_pool()
            pool_id = pool["id"]
            pool_info = {
                "token0_symbol": pool["token0"]["symbol"],
                "token1_symbol": pool["token1"]["symbol"],
                "volume_usd": pool.get("volumeUSD", "0")
            }
            ctx.logger.info(f"✅ Selected pool: {pool_info['token0_symbol']}/{pool_info['token1_symbol']}")
        
        # Fetch events
        ctx.logger.info(f"📊 Fetching events for pool {pool_id[:10]}...")
        result = await fetch_pool_events(pool_id, msg.minutes_back, msg.max_events)
        
        # Log results
        ctx.logger.info(f"✅ Successfully fetched {result['total_events']} events")
        ctx.logger.info(f"⏰ Time range: {result['time_range']['formatted_range']}")
        
        # Count event types
        swap_count = len([e for e in result["events"] if e.eventType == EventType.SWAP])
        liquidity_count = len([e for e in result["events"] if e.eventType == EventType.LIQUIDITY])
        ctx.logger.info(f"📈 Event breakdown: {swap_count} swaps, {liquidity_count} liquidity events")
        
        # Send response
        response = EventsResponse(
            events=result["events"],
            pool_id=pool_id,
            pool_info=pool_info,
            total_events=result["total_events"],
            time_range=result["time_range"]
        )
        
        await ctx.send(sender, response)
        ctx.logger.info(f"📤 Sent {result['total_events']} events to {sender}")
        
    except Exception as e:
        ctx.logger.error(f"❌ Error processing fetch events request: {e}")
        error_response = DataAgentError(
            error_type="fetch_error",
            message=str(e),
            details={
                "pool_id": msg.pool_id,
                "minutes_back": msg.minutes_back,
                "max_events": msg.max_events
            }
        )
        await ctx.send(sender, error_response)

# Chat message handlers
@chat_proto.on_message(ChatMessage)
async def handle_chat_message(ctx: Context, sender: str, msg: ChatMessage):
    """Handle incoming chat messages"""
    ctx.logger.info(f"💬 Received chat message from {sender}")
    
    # Send acknowledgment immediately
    await ctx.send(sender, ChatAcknowledgement(
        timestamp=datetime.now(timezone.utc),
        acknowledged_msg_id=msg.msg_id
    ))
    
    # Store session context
    ctx.storage.set(str(ctx.session), sender)
    
    for item in msg.content:
        if isinstance(item, TextContent):
            user_text = item.text
            ctx.logger.info(f"📝 Processing message: {user_text}")
            
            # Check if this is a data request
            if is_data_request(user_text):
                ctx.logger.info("🔍 Detected data request")
                await handle_chat_data_request(ctx, sender, user_text)
            else:
                # General help or greeting
                await handle_chat_general(ctx, sender, user_text)

async def handle_chat_data_request(ctx: Context, sender: str, user_text: str):
    """Handle data request from chat"""
    try:
        # Parse the request
        params = parse_data_request(user_text)
        ctx.logger.info(f"📊 Parsed parameters: {params}")
        
        # Create fetch request
        fetch_request = FetchEventsRequest(
            pool_id=params["pool_id"],
            minutes_back=params["minutes_back"],
            max_events=params["max_events"]
        )
        
        # Send status message
        status_msg = f"🔍 Fetching pool events...\n"
        status_msg += f"• Time range: {format_window_label(params['minutes_back'])}\n"
        status_msg += f"• Max events: {params['max_events']}\n"
        if params["pool_id"]:
            status_msg += f"• Pool: {params['pool_id'][:10]}...\n"
        else:
            status_msg += f"• Pool: Auto-selecting top pool by volume\n"
        status_msg += f"• Please wait..."
        
        await ctx.send(sender, create_text_chat(status_msg))
        
        # Fetch the data (reuse existing logic)
        pool_id = params["pool_id"]
        pool_info = {}
        
        if not pool_id:
            ctx.logger.info("🔍 Auto-selecting top pool by volume...")
            pool = await get_top_pool()
            pool_id = pool["id"]
            pool_info = {
                "token0_symbol": pool["token0"]["symbol"],
                "token1_symbol": pool["token1"]["symbol"],
                "volume_usd": pool.get("volumeUSD", "0")
            }
        
        # Fetch events
        result = await fetch_pool_events(pool_id, params["minutes_back"], params["max_events"])
        
        # Format response for chat
        response_text = f"✅ **Data Fetch Complete!**\n\n"
        
        if pool_info:
            response_text += f"🏊 **Pool**: {pool_info['token0_symbol']}/{pool_info['token1_symbol']}\n"
            response_text += f"💰 **Volume**: ${float(pool_info['volume_usd']):.2f}\n"
        
        response_text += f"📊 **Events Found**: {result['total_events']}\n"
        response_text += f"⏰ **Time Range**: {result['time_range']['formatted_range']}\n"
        
        # Event breakdown
        swap_count = len([e for e in result["events"] if e.eventType == EventType.SWAP])
        liquidity_count = len([e for e in result["events"] if e.eventType == EventType.LIQUIDITY])
        
        response_text += f"\n📈 **Event Breakdown**:\n"
        response_text += f"• Swaps: {swap_count}\n"
        response_text += f"• Liquidity Events: {liquidity_count}\n"
        
        # Show sample events
        if result["events"]:
            response_text += f"\n📝 **Sample Events** (first 3):\n"
            for i, event in enumerate(result["events"][:3]):
                event_type = "🔄 Swap" if event.eventType == EventType.SWAP else "💧 Liquidity"
                response_text += f"{i+1}. {event_type} - ${event.amount:.2f} - {format_time(event.unixTimestamp)}\n"
        
        response_text += f"\n💡 **Next Steps**: This data can be used for backtesting analysis!"
        
        await ctx.send(sender, create_text_chat(response_text))
        
    except Exception as e:
        ctx.logger.error(f"❌ Chat data request failed: {e}")
        error_msg = f"❌ **Error fetching data**: {str(e)}\n\n"
        error_msg += f"💡 **Try again with**: 'Get pool events from last 24 hours' or 'Show me recent swap data'"
        await ctx.send(sender, create_text_chat(error_msg))

async def handle_chat_general(ctx: Context, sender: str, user_text: str):
    """Handle general chat messages (help, greetings, etc.)"""
    user_text_lower = user_text.lower()
    
    if any(word in user_text_lower for word in ["hello", "hi", "hey", "greetings"]):
        response = """👋 **Hello! I'm the Data Agent**

I can help you fetch pool events from The Graph Protocol! 

🔍 **What I can do**:
• Fetch swap and liquidity events from Uniswap pools
• Auto-select top pools by volume
• Filter by time range (minutes, hours, days, weeks)
• Provide detailed event breakdowns

💬 **Example requests**:
• "Get pool events from last 24 hours"
• "Show me swap data from the past week"
• "Fetch 500 events from last 3 days"
• "Get recent liquidity events"

📊 Just ask me for data and I'll fetch it for you!"""
        
    elif any(word in user_text_lower for word in ["help", "commands", "what", "how"]):
        response = """🤖 **Data Agent Help**

📋 **Available Commands**:
• **Data Requests**: Ask for pool events with natural language
• **Time Ranges**: Specify minutes, hours, days, or weeks
• **Event Limits**: Specify how many events you want (default: 100)
• **Pool Selection**: Let me auto-select or specify a pool address

🗣️ **Natural Language Examples**:
• "Get events from last hour"
• "Show me 200 swap events from past 3 days"  
• "Fetch recent pool data"
• "Get week of liquidity events"

⚡ **Features**:
• Real-time data from The Graph
• Automatic pool selection (top by volume)
• Event normalization for backtesting
• Detailed breakdowns and summaries

🔄 **Integration**: Data can be used by other agents for backtesting!"""
        
    else:
        response = """🤔 **I didn't quite understand that.**

I specialize in fetching pool event data from The Graph Protocol.

💡 **Try asking me**:
• "Get pool events from last 24 hours"
• "Show me recent swap data"  
• "Help" - for more commands
• "Hello" - for an introduction

📊 I'm here to help with your data needs!"""
    
    await ctx.send(sender, create_text_chat(response))

@chat_proto.on_message(ChatAcknowledgement)
async def handle_chat_acknowledgement(ctx: Context, sender: str, msg: ChatAcknowledgement):
    """Handle chat acknowledgements"""
    ctx.logger.info(f"✅ Received acknowledgement from {sender} for message: {msg.acknowledged_msg_id}")

# Include protocols
data_agent.include(data_protocol, publish_manifest=True)
data_agent.include(chat_proto, publish_manifest=True)

if __name__ == "__main__":
    print("""
🤖 Starting Data Agent...

📊 Features:
  • Fetch pool events from The Graph subgraph
  • Auto-select top pool by volume if not specified
  • Normalize events for backtest compatibility
  • Rate limiting and error handling
  • Mailbox support for Agentverse integration
  • 💬 Natural language chat interface

📡 Message Types:
  • FetchEventsRequest - Request pool events
  • EventsResponse - Pool events response
  • DataAgentError - Error response
  • ChatMessage - Natural language data requests

💬 Chat Commands:
  • "Hello" - Introduction and help
  • "Get pool events from last 24 hours"
  • "Show me swap data from past week"
  • "Fetch 200 events from last 3 days"
  • "Help" - Show available commands

🔗 Integration:
  • Connected to The Graph Protocol
  • Compatible with enhanced backtest agent
  • Supports structured event data
  • Chat-enabled for easy interaction

🛑 Stop with Ctrl+C
    """)
    data_agent.run()
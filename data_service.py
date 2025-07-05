import requests
import re
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from enum import Enum

# Data Service Configuration
SUBGRAPH_ID = "5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"
API_KEY = "bfb3f65c79e8d0f6e7dd728b91d46600"
GRAPH_URL = f"https://gateway.thegraph.com/api/{API_KEY}/subgraphs/id/{SUBGRAPH_ID}"

# Event types
class EventType(int, Enum):
    LIQUIDITY = 0  # Mints and Burns
    SWAP = 1       # Swaps

class PoolEvent:
    """Individual pool event model"""
    def __init__(self, amount: float, amount0: float, amount1: float,
                 eventType: EventType, unixTimestamp: int,
                 tickLower: Optional[int] = None, tickUpper: Optional[int] = None):
        self.amount = amount
        self.amount0 = amount0
        self.amount1 = amount1
        self.eventType = eventType
        self.unixTimestamp = unixTimestamp
        self.tickLower = tickLower
        self.tickUpper = tickUpper

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

async def fetch_pool_events(pool_id: str, minutes_back: int = 1440, max_events: int = 100, start_ts: int = None, end_ts: int = None) -> Dict[str, Any]:
    """Fetch events from The Graph for a specific pool"""

    # Use specific timestamps if provided, otherwise calculate from minutes_back
    if start_ts is not None and end_ts is not None:
        ts_cutoff = start_ts
        now_ts = end_ts
    else:
        # Calculate time cutoff
        now_ts = int(datetime.now(timezone.utc).timestamp())
        ts_cutoff = now_ts - minutes_back * 60

    # GraphQL query for events
    query_events = f"""
    {{
      swaps(first: {max_events}, orderBy: timestamp, orderDirection: desc, where: {{
        pool: "{pool_id}", timestamp_gte: {ts_cutoff}, timestamp_lte: {now_ts}
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
        pool: "{pool_id}", timestamp_gte: {ts_cutoff}, timestamp_lte: {now_ts}
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
        pool: "{pool_id}", timestamp_gte: {ts_cutoff}, timestamp_lte: {now_ts}
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

async def save_onchain_data(pool_id: str, events_data: Dict[str, Any], request_text: str):
    """Save onchain data to the onchainData folder"""
    try:
        import os
        import json
        from datetime import datetime

        # Create onchainData directory
        onchain_dir = os.path.join(os.path.dirname(__file__), "onchainData")
        os.makedirs(onchain_dir, exist_ok=True)

        # Create timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pool_short = pool_id[:10] if pool_id else "unknown"

        # Prepare data to save
        onchain_data = {
            "request": {
                "text": request_text,
                "timestamp": timestamp,
                "pool_id": pool_id
            },
            "response": {
                "total_events": events_data["total_events"],
                "time_range": events_data["time_range"],
                "events": [
                    {
                        "amount": event.amount,
                        "amount0": event.amount0,
                        "amount1": event.amount1,
                        "eventType": event.eventType.value,
                        "unixTimestamp": event.unixTimestamp,
                        "tickLower": event.tickLower,
                        "tickUpper": event.tickUpper
                    } for event in events_data["events"]
                ]
            }
        }

        # Save to file
        filename = f"graph_data_{pool_short}_{timestamp}.json"
        filepath = os.path.join(onchain_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(onchain_data, f, indent=2)

        print(f"💾 Saved Graph data to onchainData/{filename}")

    except Exception as e:
        print(f"❌ Failed to save onchain data: {e}")

async def handle_data_request(text: str) -> str:
    """Handle data request and return formatted response"""
    try:
        # Parse the request
        params = parse_data_request(text)

        # Get pool ID if not provided
        pool_id = params["pool_id"]
        pool_info = {}

        if not pool_id:
            pool = await get_top_pool()
            pool_id = pool["id"]
            pool_info = {
                "token0_symbol": pool["token0"]["symbol"],
                "token1_symbol": pool["token1"]["symbol"],
                "volume_usd": pool.get("volumeUSD", "0")
            }

        # Fetch events
        result = await fetch_pool_events(pool_id, params["minutes_back"], params["max_events"])

        # Save onchain data
        await save_onchain_data(pool_id, result, text)

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

        return response_text

    except Exception as e:
        error_msg = f"❌ **Error fetching data**: {str(e)}\n\n"
        error_msg += f"💡 **Try again with**: 'Get pool events from last 24 hours' or 'Show me recent swap data'"
        return error_msg

# Test connection function
async def test_graph_connection() -> bool:
    """Test connection to The Graph"""
    try:
        pool = await get_top_pool()
        return True
    except Exception:
        return False
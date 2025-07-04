from uagents import Agent, Context, Model
import json
import os
import asyncio
import aiohttp
from typing import List, Dict, Any
from data_storage import BacktestDataManager


class FetchEventsRequest(Model):
    pool: str
    start: int
    end: int


class EventsResponse(Model):
    kind: str
    events: List[Dict[str, Any]]


class PoolEvent(Model):
    amount: int
    amount0: int
    amount1: int
    eventType: int  # 0 = Mint/Burn, 1 = Swap
    tickLower: int
    tickUpper: int
    unixTimestamp: int


# Create DataAgent
data_agent = Agent(
    name="DataAgent",
    port=8001,
    seed="data_agent_secret_phrase",
    endpoint=["http://127.0.0.1:8001/submit"],
)


class TheGraphClient:
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.query = """
        query PoolEvents($pool: String!, $start: Int!, $end: Int!, $skip: Int!) {
          mints(
            first: 1000, skip: $skip,
            where: { pool: $pool, timestamp_gte: $start, timestamp_lte: $end }
            orderBy: timestamp, orderDirection: asc
          ) {
            amount
            amount0
            amount1
            tickLower
            tickUpper
            timestamp
          }
          burns(
            first: 1000, skip: $skip,
            where: { pool: $pool, timestamp_gte: $start, timestamp_lte: $end }
            orderBy: timestamp, orderDirection: asc
          ) {
            amount
            amount0
            amount1
            tickLower
            tickUpper
            timestamp
          }
          swaps(
            first: 1000, skip: $skip,
            where: { pool: $pool, timestamp_gte: $start, timestamp_lte: $end }
            orderBy: timestamp, orderDirection: asc
          ) {
            amount0
            amount1
            tick
            timestamp
          }
        }
        """

    async def fetch_events(self, pool: str, start: int, end: int) -> List[Dict[str, Any]]:
        """Fetch events from The Graph and convert to PoolEvent format"""
        events = []
        skip = 0

        async with aiohttp.ClientSession() as session:
            while True:
                variables = {
                    "pool": pool.lower(),
                    "start": start,
                    "end": end,
                    "skip": skip
                }

                payload = {
                    "query": self.query,
                    "variables": variables
                }

                try:
                    async with session.post(self.endpoint, json=payload) as response:
                        if response.status != 200:
                            break

                        data = await response.json()

                        if "errors" in data:
                            data_agent.ctx.logger.error(f"GraphQL errors: {data['errors']}")
                            break

                        result = data.get("data", {})
                        mints = result.get("mints", [])
                        burns = result.get("burns", [])
                        swaps = result.get("swaps", [])

                        # If no more data, break
                        if not mints and not burns and not swaps:
                            break

                        # Convert mints to PoolEvent format
                        for mint in mints:
                            events.append({
                                "amount": int(mint.get("amount", "0")),
                                "amount0": int(mint.get("amount0", "0")),
                                "amount1": int(mint.get("amount1", "0")),
                                "eventType": 0,  # Mint/Burn
                                "tickLower": int(mint.get("tickLower", "0")),
                                "tickUpper": int(mint.get("tickUpper", "0")),
                                "unixTimestamp": int(mint.get("timestamp", "0"))
                            })

                        # Convert burns to PoolEvent format (negative amount)
                        for burn in burns:
                            events.append({
                                "amount": -int(burn.get("amount", "0")),  # Negative for burns
                                "amount0": int(burn.get("amount0", "0")),
                                "amount1": int(burn.get("amount1", "0")),
                                "eventType": 0,  # Mint/Burn
                                "tickLower": int(burn.get("tickLower", "0")),
                                "tickUpper": int(burn.get("tickUpper", "0")),
                                "unixTimestamp": int(burn.get("timestamp", "0"))
                            })

                        # Convert swaps to PoolEvent format
                        for swap in swaps:
                            events.append({
                                "amount": 0,  # Not used for swaps
                                "amount0": int(swap.get("amount0", "0")),
                                "amount1": int(swap.get("amount1", "0")),
                                "eventType": 1,  # Swap
                                "tickLower": 0,  # Not used for swaps
                                "tickUpper": 0,  # Not used for swaps
                                "unixTimestamp": int(swap.get("timestamp", "0"))
                            })

                        skip += 1000

                except Exception as e:
                    data_agent.ctx.logger.error(f"Error fetching from The Graph: {e}")
                    break

        # Sort events by timestamp
        events.sort(key=lambda x: x["unixTimestamp"])
        return events


# Initialize The Graph client and data manager
graph_client = TheGraphClient("https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3")
data_manager = BacktestDataManager()


@data_agent.on_message(model=FetchEventsRequest)
async def fetch_events_handler(ctx: Context, sender: str, msg: FetchEventsRequest):
    """Handle fetch events requests"""
    try:
        ctx.logger.info(f"Fetching events for pool {msg.pool} from {msg.start} to {msg.end}")

        # Log the request
        data_manager.log_agent_activity("DataAgent", f"Received request for pool {msg.pool} from {msg.start} to {msg.end}")

        # Check if we have cached data first
        cached_events = data_manager.get_cached_graph_data(msg.pool, msg.start, msg.end)
        if cached_events:
            ctx.logger.info(f"Using cached data: {len(cached_events)} events")
            data_manager.log_agent_activity("DataAgent", f"Found cached data with {len(cached_events)} events")

            response = EventsResponse(
                kind="Events",
                events=cached_events
            )
            await ctx.send(sender, response)
            return

        # Fetch events from The Graph
        events = await graph_client.fetch_events(msg.pool, msg.start, msg.end)

        ctx.logger.info(f"Fetched {len(events)} events from The Graph")

        # Cache the fetched data
        if events:
            data_manager.cache_graph_data(msg.pool, msg.start, msg.end, events)
            data_manager.log_agent_activity("DataAgent", f"Cached {len(events)} events for future use")

        # Send response back
        response = EventsResponse(
            kind="Events",
            events=events
        )

        await ctx.send(sender, response)

    except Exception as e:
        ctx.logger.error(f"Error in fetch_events_handler: {e}")
        data_manager.log_agent_activity("DataAgent", f"Error fetching events: {str(e)}")

        # Send error response
        error_response = EventsResponse(
            kind="Error",
            events=[]
        )
        await ctx.send(sender, error_response)


@data_agent.on_event("startup")
async def startup_handler(ctx: Context):
    """Handle agent startup"""
    ctx.logger.info(f"DataAgent started with address: {ctx.agent.address}")


if __name__ == "__main__":
    data_agent.run()

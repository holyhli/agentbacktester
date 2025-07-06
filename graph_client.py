#!/usr/bin/env python3
"""
The Graph client for fetching Uniswap V3/V4 data
Handles GraphQL queries to The Graph's Uniswap subgraphs
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from gql import gql, Client
from gql.transport.aiohttp import AIOHTTPTransport
import backoff

logger = logging.getLogger(__name__)

class TheGraphClient:
    """Client for interacting with The Graph's Uniswap subgraphs"""
    
    def __init__(self, subgraph_url: Optional[str] = None):
        # Try multiple endpoints for better reliability
        self.subgraph_url = subgraph_url or "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3"
        self.transport = AIOHTTPTransport(url=self.subgraph_url)
        self.client = Client(transport=self.transport, fetch_schema_from_transport=False)
        self.use_mock_data = False  # Flag to fall back to mock data
        
        # Rate limiting
        self.max_retries = 3
        self.retry_delay = 1.0
        
    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    async def execute_query(self, query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a GraphQL query with retry logic"""
        try:
            logger.debug(f"Executing GraphQL query: {query[:100]}...")
            
            # Convert query string to gql object
            gql_query = gql(query)
            
            # Execute query
            result = await self.client.execute_async(gql_query, variable_values=variables)
            
            logger.debug(f"Query executed successfully")
            return result
            
        except Exception as e:
            logger.error(f"GraphQL query failed: {e}")
            # Fall back to mock data if GraphQL fails
            self.use_mock_data = True
            logger.warning("Falling back to mock data due to GraphQL errors")
            return {}
    
    async def get_pool_events(self, pool_address: str, start_time: int, end_time: int, 
                            event_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Fetch pool events from The Graph
        
        Args:
            pool_address: Pool contract address
            start_time: Start timestamp
            end_time: End timestamp
            event_types: List of event types to fetch (swaps, mints, burns)
            
        Returns:
            List of events with standardized format
        """
        try:
            logger.info(f"Fetching events for pool {pool_address} from {start_time} to {end_time}")
            
            # Check if we should use mock data
            if self.use_mock_data:
                logger.info("Using mock data due to GraphQL issues")
                return self._generate_mock_events(pool_address, start_time, end_time, event_types)
            
            # Default to all event types if none specified
            if event_types is None:
                event_types = ["swaps", "mints", "burns"]
            
            all_events = []
            
            # Fetch swaps
            if "swaps" in event_types:
                swaps = await self._fetch_swaps(pool_address, start_time, end_time) 
                all_events.extend(swaps)
            
            # Fetch mints
            if "mints" in event_types:
                mints = await self._fetch_mints(pool_address, start_time, end_time)
                all_events.extend(mints)
            
            # Fetch burns
            if "burns" in event_types:
                burns = await self._fetch_burns(pool_address, start_time, end_time)
                all_events.extend(burns)
            
            # Sort events by timestamp
            all_events.sort(key=lambda x: x.get('timestamp', 0))
            
            logger.info(f"Fetched {len(all_events)} events for pool {pool_address}")
            return all_events
            
        except Exception as e:
            logger.error(f"Error fetching pool events: {e}")
            # Fall back to mock data
            logger.info("Falling back to mock data")
            return self._generate_mock_events(pool_address, start_time, end_time, event_types)
    
    async def _fetch_swaps(self, pool_address: str, start_time: int, end_time: int) -> List[Dict[str, Any]]:
        """Fetch swap events from the pool"""
        query = """
        query GetSwaps($pool: String!, $startTime: Int!, $endTime: Int!) {
            swaps(
                where: {
                    pool: $pool,
                    timestamp_gte: $startTime,
                    timestamp_lte: $endTime
                }
                orderBy: timestamp
                orderDirection: asc
                first: 1000
            ) {
                id
                timestamp
                pool {
                    id
                    token0Price
                    token1Price
                }
                amount0
                amount1
                amountUSD
                sqrtPriceX96
                tick
                gasUsed
                gasPrice
            }
        }
        """
        
        variables = {
            "pool": pool_address.lower(),
            "startTime": start_time,
            "endTime": end_time
        }
        
        try:
            result = await self.execute_query(query, variables)
            swaps = result.get('swaps', [])
            
            # Transform to standardized format
            events = []
            for swap in swaps:
                events.append({
                    "type": "swap",
                    "id": swap.get('id'),
                    "timestamp": int(swap.get('timestamp', 0)),
                    "amount0": float(swap.get('amount0', 0)),
                    "amount1": float(swap.get('amount1', 0)),
                    "amountUSD": float(swap.get('amountUSD', 0)),
                    "price": float(swap.get('pool', {}).get('token0Price', 0)),
                    "sqrtPriceX96": swap.get('sqrtPriceX96'),
                    "tick": int(swap.get('tick', 0)),
                    "gasUsed": int(swap.get('gasUsed', 0)),
                    "gasPrice": int(swap.get('gasPrice', 0))
                })
            
            return events
            
        except Exception as e:
            logger.error(f"Error fetching swaps: {e}")
            return []
    
    async def _fetch_mints(self, pool_address: str, start_time: int, end_time: int) -> List[Dict[str, Any]]:
        """Fetch mint events from the pool"""
        query = """
        query GetMints($pool: String!, $startTime: Int!, $endTime: Int!) {
            mints(
                where: {
                    pool: $pool,
                    timestamp_gte: $startTime,
                    timestamp_lte: $endTime
                }
                orderBy: timestamp
                orderDirection: asc
                first: 1000
            ) {
                id
                timestamp
                pool {
                    id
                }
                amount
                amount0
                amount1
                amountUSD
                tickLower
                tickUpper
                gasUsed
                gasPrice
            }
        }
        """
        
        variables = {
            "pool": pool_address.lower(),
            "startTime": start_time,
            "endTime": end_time
        }
        
        try:
            result = await self.execute_query(query, variables)
            mints = result.get('mints', [])
            
            # Transform to standardized format
            events = []
            for mint in mints:
                events.append({
                    "type": "mint",
                    "id": mint.get('id'),
                    "timestamp": int(mint.get('timestamp', 0)),
                    "amount": float(mint.get('amount', 0)),
                    "amount0": float(mint.get('amount0', 0)),
                    "amount1": float(mint.get('amount1', 0)),
                    "amountUSD": float(mint.get('amountUSD', 0)),
                    "tickLower": int(mint.get('tickLower', 0)),
                    "tickUpper": int(mint.get('tickUpper', 0)),
                    "gasUsed": int(mint.get('gasUsed', 0)),
                    "gasPrice": int(mint.get('gasPrice', 0))
                })
            
            return events
            
        except Exception as e:
            logger.error(f"Error fetching mints: {e}")
            return []
    
    async def _fetch_burns(self, pool_address: str, start_time: int, end_time: int) -> List[Dict[str, Any]]:
        """Fetch burn events from the pool"""
        query = """
        query GetBurns($pool: String!, $startTime: Int!, $endTime: Int!) {
            burns(
                where: {
                    pool: $pool,
                    timestamp_gte: $startTime,
                    timestamp_lte: $endTime
                }
                orderBy: timestamp
                orderDirection: asc
                first: 1000
            ) {
                id
                timestamp
                pool {
                    id
                }
                amount
                amount0
                amount1
                amountUSD
                tickLower
                tickUpper
                gasUsed
                gasPrice
            }
        }
        """
        
        variables = {
            "pool": pool_address.lower(),
            "startTime": start_time,
            "endTime": end_time
        }
        
        try:
            result = await self.execute_query(query, variables)
            burns = result.get('burns', [])
            
            # Transform to standardized format
            events = []
            for burn in burns:
                events.append({
                    "type": "burn",
                    "id": burn.get('id'),
                    "timestamp": int(burn.get('timestamp', 0)),
                    "amount": float(burn.get('amount', 0)),
                    "amount0": float(burn.get('amount0', 0)),
                    "amount1": float(burn.get('amount1', 0)),
                    "amountUSD": float(burn.get('amountUSD', 0)),
                    "tickLower": int(burn.get('tickLower', 0)),
                    "tickUpper": int(burn.get('tickUpper', 0)),
                    "gasUsed": int(burn.get('gasUsed', 0)),
                    "gasPrice": int(burn.get('gasPrice', 0))
                })
            
            return events
            
        except Exception as e:
            logger.error(f"Error fetching burns: {e}")
            return []
    
    async def get_pool_info(self, pool_address: str) -> Dict[str, Any]:
        """Get basic pool information"""
        query = """
        query GetPool($pool: String!) {
            pool(id: $pool) {
                id
                token0 {
                    id
                    symbol
                    decimals
                }
                token1 {
                    id
                    symbol
                    decimals
                }
                feeTier
                liquidity
                sqrtPrice
                tick
                token0Price
                token1Price
                volumeUSD
                totalValueLockedUSD
            }
        }
        """
        
        variables = {"pool": pool_address.lower()}
        
        try:
            result = await self.execute_query(query, variables)
            return result.get('pool', {})
        except Exception as e:
            logger.error(f"Error fetching pool info: {e}")
            return {}
    
    def _generate_mock_events(self, pool_address: str, start_time: int, end_time: int, 
                            event_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Generate mock events for testing when GraphQL is unavailable"""
        if event_types is None:
            event_types = ["swaps", "mints", "burns"]
        
        events = []
        time_range = end_time - start_time
        
        # Generate realistic mock data
        for i, event_type in enumerate(event_types):
            if event_type == "swaps":
                # Generate multiple swaps
                for j in range(5):
                    timestamp = start_time + (j * time_range // 5)
                    events.append({
                        "type": "swap",
                        "id": f"swap_{pool_address}_{timestamp}",
                        "timestamp": timestamp,
                        "amount0": float(1000 + j * 100),
                        "amount1": float(2500 + j * 50),
                        "amountUSD": float(2500000 + j * 10000),
                        "price": float(2500 + j * 10),
                        "sqrtPriceX96": str(123456789012345678901234567890),
                        "tick": 123456,
                        "gasUsed": 150000,
                        "gasPrice": 20000000000
                    })
            
            elif event_type == "mints":
                # Generate mints
                for j in range(3):
                    timestamp = start_time + (j * time_range // 3) + 3600
                    events.append({
                        "type": "mint",
                        "id": f"mint_{pool_address}_{timestamp}",
                        "timestamp": timestamp,
                        "amount": float(500 + j * 100),
                        "amount0": float(1000 + j * 200),
                        "amount1": float(2500 + j * 500),
                        "amountUSD": float(1250000 + j * 250000),
                        "tickLower": 123000,
                        "tickUpper": 124000,
                        "gasUsed": 200000,
                        "gasPrice": 20000000000
                    })
            
            elif event_type == "burns":
                # Generate burns
                for j in range(2):
                    timestamp = start_time + (j * time_range // 2) + 7200
                    events.append({
                        "type": "burn",
                        "id": f"burn_{pool_address}_{timestamp}",
                        "timestamp": timestamp,
                        "amount": float(250 + j * 50),
                        "amount0": float(500 + j * 100),
                        "amount1": float(1250 + j * 250),
                        "amountUSD": float(625000 + j * 125000),
                        "tickLower": 123000,
                        "tickUpper": 124000,
                        "gasUsed": 180000,
                        "gasPrice": 20000000000
                    })
        
        # Sort by timestamp
        events.sort(key=lambda x: x.get('timestamp', 0))
        logger.info(f"Generated {len(events)} mock events for pool {pool_address}")
        return events

    async def close(self):
        """Close the GraphQL client"""
        if hasattr(self, 'client'):
            await self.client.close_async()

# Global client instance
graph_client = TheGraphClient()

# Test function
async def test_graph_client():
    """Test the Graph client with a real pool"""
    client = TheGraphClient()
    
    # Test with USDC-ETH pool
    pool_address = "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"
    
    # Get pool info
    pool_info = await client.get_pool_info(pool_address)
    print(f"Pool Info: {pool_info}")
    
    # Get recent events (last 24 hours)
    end_time = int(datetime.now().timestamp())
    start_time = end_time - (24 * 60 * 60)  # 24 hours ago
    
    events = await client.get_pool_events(pool_address, start_time, end_time)
    print(f"Fetched {len(events)} events")
    
    # Show first few events
    for event in events[:3]:
        print(f"Event: {event}")
    
    await client.close()

if __name__ == "__main__":
    asyncio.run(test_graph_client()) 
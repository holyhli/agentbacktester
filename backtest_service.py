import json
import os
import tempfile
import asyncio
import logging
from subprocess import run, PIPE, CalledProcessError
from typing import Dict, Any
from datetime import datetime
from uagents import Model, Field

from data_storage import BacktestDataManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BacktestRequest(Model):
    pool: str = Field(
        description="Uniswap V4 pool address to backtest (e.g., 0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640 for USDC-ETH)",
    )
    start: int = Field(
        description="Start timestamp (Unix timestamp, e.g., 1735689600 for 2025-01-01)",
    )
    end: int = Field(
        description="End timestamp (Unix timestamp, e.g., 1735776000 for 2025-01-02)",
    )
    strategy_params: Dict[str, Any] = Field(
        default={},
        description="Optional strategy parameters for the backtest",
    )

class BacktestResponse(Model):
    kind: str = Field(
        description="Type of result (backtest_result)",
    )
    pnl: float = Field(
        description="Profit and Loss from the backtest",
    )
    sharpe: float = Field(
        description="Sharpe ratio of the strategy",
    )
    total_fees: float = Field(
        description="Total fees paid during the backtest",
    )
    impermanent_loss: float = Field(
        description="Impermanent loss incurred",
    )
    gas_costs: float = Field(
        description="Gas costs for transactions",
    )
    success: bool = Field(
        description="Whether the backtest completed successfully",
    )
    error_message: str = Field(
        default="",
        description="Error message if backtest failed",
    )

# Data manager for logging and storage
data_manager = BacktestDataManager()

# Common pool addresses for easy reference
KNOWN_POOLS = {
    "usdc-eth": "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640",
    "wbtc-eth": "0xcbcdf9626bc03e24f779434178a73a0b4bad62ed",
    "usdc-weth": "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640", 
    "wbtc-weth": "0xcbcdf9626bc03e24f779434178a73a0b4bad62ed",
}

async def run_backtest(pool: str, start: int, end: int, strategy_params: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Run a backtest for the specified pool and time period
    
    Args:
        pool: Pool address to backtest
        start: Start timestamp  
        end: End timestamp
        strategy_params: Optional strategy parameters
        
    Returns:
        Dictionary with backtest results
    """
    try:
        logger.info(f"Starting backtest for pool {pool} from {start} to {end}")
        
        # Validate inputs
        if not pool or not isinstance(pool, str):
            raise ValueError("Pool address must be a valid string")
            
        if not start or not end or start >= end:
            raise ValueError("Start and end timestamps must be valid and start < end")
        
        # Log the backtest request
        data_manager.log_agent_activity("BacktestService", f"Starting backtest for {pool}")
        
        # Check if pool is a known pool name and convert to address
        pool_address = KNOWN_POOLS.get(pool.lower(), pool)
        
        # Simulate data fetching (in real implementation, you'd call your data agent)
        events_data = await simulate_fetch_events(pool_address, start, end)
        
        if not events_data:
            raise ValueError("No data available for the specified pool and time period")
        
        # Run the actual backtest simulation
        backtest_results = await simulate_backtest(pool_address, events_data, strategy_params or {})
        
        # Store results
        result_data = {
            'pool_address': pool_address,
            'start_time': start,
            'end_time': end,
            'strategy_params': strategy_params or {},
            'results': backtest_results,
            'timestamp': datetime.now().isoformat()
        }
        
        data_manager.store_backtest_results(result_data)
        data_manager.log_agent_activity("BacktestService", f"Completed backtest for {pool_address}")
        
        return {
            "kind": "backtest_result",
            "success": True,
            **backtest_results
        }
        
    except Exception as e:
        error_msg = f"Backtest failed: {str(e)}"
        logger.error(error_msg)
        
        data_manager.log_agent_activity("BacktestService", f"Backtest failed: {error_msg}")
        
        return {
            "kind": "backtest_result", 
            "success": False,
            "error_message": error_msg,
            "pnl": 0.0,
            "sharpe": 0.0,
            "total_fees": 0.0,
            "impermanent_loss": 0.0,
            "gas_costs": 0.0
        }

async def simulate_fetch_events(pool: str, start: int, end: int) -> Dict[str, Any]:
    """
    Simulate fetching historical events for a pool
    In a real implementation, this would call your data fetching logic
    """
    try:
        logger.info(f"Fetching events for pool {pool}")
        
        # Simulate some processing time
        await asyncio.sleep(0.5)
        
        # Generate some mock events data
        # In real implementation, replace this with actual data fetching
        mock_events = {
            "pool": pool,
            "start": start,
            "end": end,
            "events": [
                {
                    "type": "swap",
                    "timestamp": start + 3600,
                    "amount0": "1000.0",
                    "amount1": "2500.0",
                    "price": "2500.0"
                },
                {
                    "type": "mint", 
                    "timestamp": start + 7200,
                    "amount": "500.0",
                    "liquidity": "1250.0"
                },
                {
                    "type": "burn",
                    "timestamp": start + 10800,
                    "amount": "250.0",
                    "liquidity": "625.0"
                }
            ]
        }
        
        logger.info(f"Fetched {len(mock_events['events'])} events for pool {pool}")
        return mock_events
        
    except Exception as e:
        logger.error(f"Error fetching events: {e}")
        return {}

async def simulate_backtest(pool: str, events_data: Dict[str, Any], strategy_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simulate running a backtest with the provided events data
    In a real implementation, this would use your backtesting engine
    """
    try:
        logger.info(f"Running backtest simulation for pool {pool}")
        
        # Simulate some processing time
        await asyncio.sleep(1.0)
        
        # Extract events
        events = events_data.get("events", [])
        
        if not events:
            raise ValueError("No events data available for backtesting")
        
        # Simulate backtest calculations
        # In real implementation, replace this with actual backtesting logic
        
        # Mock calculations based on events
        total_swaps = len([e for e in events if e.get("type") == "swap"])
        total_volume = sum(float(e.get("amount0", 0)) for e in events if e.get("type") == "swap")
        
        # Simulate strategy performance
        base_return = (total_volume / 10000) * 0.001  # 0.1% of volume as base return
        volatility_penalty = total_swaps * 0.0001  # Small penalty for volatility
        
        pnl = base_return - volatility_penalty
        sharpe = pnl / max(0.01, abs(pnl) * 0.1) if pnl != 0 else 0
        total_fees = total_volume * 0.003  # 0.3% fees
        impermanent_loss = abs(pnl) * 0.1 if pnl < 0 else 0
        gas_costs = total_swaps * 0.001  # Mock gas costs
        
        results = {
            "pnl": round(pnl, 6),
            "sharpe": round(sharpe, 2),
            "total_fees": round(total_fees, 6),
            "impermanent_loss": round(impermanent_loss, 6),
            "gas_costs": round(gas_costs, 6)
        }
        
        logger.info(f"Backtest completed with PnL: {results['pnl']}")
        return results
        
    except Exception as e:
        logger.error(f"Error in backtest simulation: {e}")
        raise

def parse_pool_input(pool_input: str) -> str:
    """
    Parse pool input and convert known pool names to addresses
    """
    pool_lower = pool_input.lower().strip()
    
    # Check if it's a known pool name
    if pool_lower in KNOWN_POOLS:
        return KNOWN_POOLS[pool_lower]
    
    # Check if it's already a valid address (starts with 0x and has 42 characters)
    if pool_input.startswith("0x") and len(pool_input) == 42:
        return pool_input
    
    # Return as-is and let validation handle it
    return pool_input

def get_default_time_period() -> tuple[int, int]:
    """
    Get default time period (last 30 days)
    """
    end_time = int(datetime.now().timestamp())
    start_time = end_time - (30 * 24 * 60 * 60)  # 30 days ago
    return start_time, end_time
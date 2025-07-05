import json
import os
import logging
import math
from typing import Dict, Any, List, Optional
from datetime import datetime, UTC
import numpy as np

from uagents import Model, Field

from data_storage import BacktestDataManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PoolEvent(Model):
    """Individual pool event model matching the mock data format"""
    amount: int = Field(description="Liquidity amount")
    amount0: int = Field(description="Amount of token0")
    amount1: int = Field(description="Amount of token1")
    eventType: int = Field(description="Event type: 0=mint/burn, 1=swap")
    tickLower: int = Field(description="Lower tick boundary")
    tickUpper: int = Field(description="Upper tick boundary")
    unixTimestamp: int = Field(description="Unix timestamp of event")

class BacktestRequest(Model):
    pool: str = Field(
        description="Uniswap V4 pool address to backtest (e.g., 0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640 for USDC-ETH)",
    )
    start: int = Field(
        description="Start timestamp (Unix timestamp, e.g., 1620158974)",
    )
    end: int = Field(
        description="End timestamp (Unix timestamp, e.g., 1620243233)",
    )
    strategy_params: Dict[str, Any] = Field(
        default={},
        description="Optional strategy parameters for the backtest",
    )

class BacktestResponse(Model):
    kind: str = Field(
        default="backtest_result",
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
    total_events: int = Field(
        default=0,
        description="Total number of events processed",
    )
    swap_events: int = Field(
        default=0,
        description="Number of swap events",
    )
    liquidity_events: int = Field(
        default=0,
        description="Number of liquidity events (mint/burn)",
    )
    volume_token0: float = Field(
        default=0.0,
        description="Total volume in token0",
    )
    volume_token1: float = Field(
        default=0.0,
        description="Total volume in token1",
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

def load_mock_data() -> List[Dict[str, Any]]:
    """Load mock data from the JSON file"""
    try:
        mock_data_path = os.path.join(os.path.dirname(__file__), "../mock_data.json")
        with open(mock_data_path, 'r') as f:
            data = json.load(f)
            return data.get("events", [])
    except Exception as e:
        logger.error(f"Error loading mock data: {e}")
        return []

def filter_events_by_timerange(events: List[Dict[str, Any]], start: int, end: int) -> List[Dict[str, Any]]:
    """Filter events by timestamp range"""
    return [
        event for event in events
        if start <= event.get("unixTimestamp", 0) <= end
    ]

def calculate_price_from_amounts(amount0: int, amount1: int) -> float:
    """Calculate price from token amounts (token1/token0)"""
    if amount0 == 0:
        return 0.0
    return abs(amount1) / abs(amount0)

def calculate_volatility(prices: List[float]) -> float:
    """Calculate price volatility (standard deviation of returns)"""
    if len(prices) < 2:
        return 0.0

    returns = []
    for i in range(1, len(prices)):
        if prices[i-1] != 0:
            returns.append((prices[i] - prices[i-1]) / prices[i-1])

    if not returns:
        return 0.0

    mean_return = sum(returns) / len(returns)
    variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
    return math.sqrt(variance)

async def run_backtest(pool: str, start: int, end: int, strategy_params: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Run a backtest for the specified pool and time period using mock data

    Args:
        pool: Pool address to backtest
        start: Start timestamp
        end: End timestamp
        strategy_params: Optional strategy parameters

    Returns:
        Dictionary with backtest results
    """
    try:
        logger.info(f"Starting enhanced backtest for pool {pool} from {start} to {end}")

        # Validate inputs
        if not pool or not isinstance(pool, str):
            raise ValueError("Pool address must be a valid string")

        if not start or not end or start >= end:
            raise ValueError("Start and end timestamps must be valid and start < end")

        # Log the backtest request
        data_manager.log_agent_activity("EnhancedBacktestService", f"Starting backtest for {pool}")

        # Check if pool is a known pool name and convert to address
        pool_address = KNOWN_POOLS.get(pool.lower(), pool)

        # Load mock data
        all_events = load_mock_data()
        if not all_events:
            raise ValueError("No mock data available")

        # Filter events by time range
        filtered_events = filter_events_by_timerange(all_events, start, end)

        if not filtered_events:
            logger.warning(f"No events found in time range {start} to {end}")
            # Use all events as fallback for demonstration
            filtered_events = all_events
            logger.info(f"Using all {len(filtered_events)} events as fallback")

        logger.info(f"Processing {len(filtered_events)} events for backtest")

        # Run the actual backtest simulation
        backtest_results = await simulate_advanced_backtest(pool_address, filtered_events, strategy_params or {})

        # Store results
        result_data = {
            'pool_address': pool_address,
            'start_time': start,
            'end_time': end,
            'strategy_params': strategy_params or {},
            'results': backtest_results,
            'timestamp': datetime.now(UTC).isoformat()
        }

        data_manager.store_backtest_results(result_data)
        data_manager.log_agent_activity("EnhancedBacktestService", f"Completed backtest for {pool_address}")

        return {
            "kind": "backtest_result",
            "success": True,
            **backtest_results
        }

    except Exception as e:
        error_msg = f"Backtest failed: {str(e)}"
        logger.error(error_msg)

        data_manager.log_agent_activity("EnhancedBacktestService", f"Backtest failed: {error_msg}")

        return {
            "kind": "backtest_result",
            "success": False,
            "error_message": error_msg,
            "pnl": 0.0,
            "sharpe": 0.0,
            "total_fees": 0.0,
            "impermanent_loss": 0.0,
            "gas_costs": 0.0,
            "total_events": 0,
            "swap_events": 0,
            "liquidity_events": 0,
            "volume_token0": 0.0,
            "volume_token1": 0.0
        }

POOL_TOKENS = {
    "usdc-eth":  ("USDC", "ETH"),
    "wbtc-eth":  ("WBTC", "ETH"),
}


TOKEN_DECIMALS = {"ETH": 18, "WETH": 18, "USDC": 6, "WBTC": 8}

def to_units(amount: int, symbol: str) -> float:
    return amount / 10 ** TOKEN_DECIMALS[symbol]

async def simulate_advanced_backtest(pool: str, events: list, p):
    fee_tier     = p.get("fee_tier", 0.003)
    pos_size_eth = p.get("position_size", 1.0)

    # --- NEW 2️⃣  resolve token symbols ------------
    sym0, sym1 = POOL_TOKENS.get(pool.lower(), ("USDC", "ETH"))

    # --- step 1: split & convert units -------------------------------------
    swap_ev   = [e for e in events if e["eventType"] == 1]
    liq_ev    = [e for e in events if e["eventType"] == 0]

    vol0 = vol1 = 0
    prices, ts  = [], []
    for s in swap_ev:
        a0 = to_units(s["amount0"], sym0)
        a1 = to_units(s["amount1"], sym1)
        vol0 += abs(a0)
        vol1 += abs(a1)
        if a0 != 0:
            prices.append(abs(a1)/abs(a0))
            ts.append(s["unixTimestamp"])

    # --- step 2: fee revenue only on your share ----------------------------
    last_price = prices[-1] if prices else 0
    liq_notional = sum(to_units(e["amount0"], sym0) * last_price
                    for e in liq_ev)
    my_notional  = pos_size_eth
    share        = my_notional / (liq_notional + my_notional) if liq_notional else 0
    fee_rev_eth= vol1 * fee_tier * share

    # --- step 3: impermanent loss ------------------------------------------
    il = 0
    if len(prices) >= 2:
        pr = prices[-1]/prices[0]
        il_pct = 2*math.sqrt(pr)/(1+pr) - 1
        il = abs(il_pct) * pos_size_eth

    # --- step 4: gas --------------------------------------------------------
    TX_MINT  = 130_000         # mint or burn once
    GAS_GWEI = 25
    gas_costs = 2 * TX_MINT * GAS_GWEI * 1e-9     # eth

    pnl = fee_rev_eth - il - gas_costs

    # --- step 5: Sharpe -----------------------------------------------------
    returns = np.diff(np.log(prices))
    vol = returns.std() * math.sqrt(365*24*60) if returns.size else 0
    sharpe = pnl/pos_size_eth/vol if vol else 0

    return {
        "pnl": round(pnl, 6),
        "sharpe": round(sharpe, 2),
        "total_fees": round(fee_rev_eth, 6),
        "impermanent_loss": round(il, 6),
        "gas_costs": round(gas_costs, 6),
        "total_events": len(events),
        "swap_events": len(swap_ev),
        "liquidity_events": len(liq_ev),
        "volume_token0": round(vol0, 3),
        "volume_token1": round(vol1, 3)
    }

# Function calling interface for LLM integration
async def backtest_function_call(
    pool: str,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    strategy_type: str = "liquidity_provision",
    fee_tier: float = 0.003,
    position_size: float = 1.0
) -> Dict[str, Any]:
    """
    Function calling interface for backtesting

    Args:
        pool: Pool address or name (e.g., "usdc-eth" or "0x88e6...")
        start_time: Start timestamp (optional, defaults to first event)
        end_time: End timestamp (optional, defaults to last event)
        strategy_type: Strategy type (currently supports "liquidity_provision")
        fee_tier: Fee tier for the strategy (default 0.3%)
        position_size: Position size in ETH equivalent (default 1.0)

    Returns:
        Backtest results dictionary
    """
    try:
        # Load mock data to get default time range if not provided
        all_events = load_mock_data()
        if not all_events:
            raise ValueError("No mock data available")

        # Set default time range if not provided
        if start_time is None:
            start_time = min(e.get("unixTimestamp", 0) for e in all_events)
        if end_time is None:
            end_time = max(e.get("unixTimestamp", 0) for e in all_events)

        # Prepare strategy parameters
        strategy_params = {
            "strategy_type": strategy_type,
            "fee_tier": fee_tier,
            "position_size": position_size
        }

        # Run backtest
        result = await run_backtest(pool, start_time, end_time, strategy_params)

        return result

    except Exception as e:
        logger.error(f"Function call backtest failed: {e}")
        return {
            "kind": "backtest_result",
            "success": False,
            "error_message": str(e),
            "pnl": 0.0,
            "sharpe": 0.0,
            "total_fees": 0.0,
            "impermanent_loss": 0.0,
            "gas_costs": 0.0,
            "total_events": 0,
            "swap_events": 0,
            "liquidity_events": 0,
            "volume_token0": 0.0,
            "volume_token1": 0.0
        }

def get_available_functions() -> List[Dict[str, Any]]:
    """
    Get available function definitions for LLM function calling
    """
    return [
        {
            "name": "backtest_function_call",
            "description": "Run a backtest simulation on Uniswap V4 pool data",
            "parameters": {
                "type": "object",
                "properties": {
                    "pool": {
                        "type": "string",
                        "description": "Pool address or name (e.g., 'usdc-eth' or '0x88e6...')"
                    },
                    "start_time": {
                        "type": "integer",
                        "description": "Start timestamp (Unix timestamp, optional)"
                    },
                    "end_time": {
                        "type": "integer",
                        "description": "End timestamp (Unix timestamp, optional)"
                    },
                    "strategy_type": {
                        "type": "string",
                        "description": "Strategy type",
                        "enum": ["liquidity_provision"],
                        "default": "liquidity_provision"
                    },
                    "fee_tier": {
                        "type": "number",
                        "description": "Fee tier for the strategy (e.g., 0.003 for 0.3%)",
                        "default": 0.003
                    },
                    "position_size": {
                        "type": "number",
                        "description": "Position size in ETH equivalent",
                        "default": 1.0
                    }
                },
                "required": ["pool"]
            }
        }
    ]

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
    Get default time period from mock data
    """
    try:
        events = load_mock_data()
        if events:
            start_time = min(e.get("unixTimestamp", 0) for e in events)
            end_time = max(e.get("unixTimestamp", 0) for e in events)
            return start_time, end_time
    except Exception:
        pass

    # Fallback to current time - 30 days
    end_time = int(datetime.now(UTC).timestamp())
    start_time = end_time - (30 * 24 * 60 * 60)
    return start_time, end_time

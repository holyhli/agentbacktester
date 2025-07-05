import json
import os
import logging
import math
from typing import Dict, Any, List, Optional
from datetime import datetime, UTC
import numpy as np

from uagents import Model, Field

from data_storage import BacktestDataManager
from data_service import fetch_pool_events, get_top_pool, EventType

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
    "usdc-eth": "0x55caabb0d2b704fd0ef8192a7e35d8837e678207",  # Updated to active pool
    "wbtc-eth": "0xcbcdf9626bc03e24f779434178a73a0b4bad62ed",
    "usdc-weth": "0x55caabb0d2b704fd0ef8192a7e35d8837e678207",  # Updated to active pool  
    "wbtc-weth": "0xcbcdf9626bc03e24f779434178a73a0b4bad62ed",
}

def load_mock_data() -> List[Dict[str, Any]]:
    """Load mock data from the JSON file (fallback only)"""
    try:
        mock_data_path = os.path.join(os.path.dirname(__file__), "../mock_data.json")
        with open(mock_data_path, 'r') as f:
            data = json.load(f)
            return data.get("events", [])
    except Exception as e:
        logger.error(f"Error loading mock data: {e}")
        return []

async def fetch_real_pool_data(pool_address: str, start: int, end: int) -> List[Dict[str, Any]]:
    """Fetch real pool data from The Graph and convert to backtest format"""
    try:
        # Calculate minutes back from the time range
        now_ts = int(datetime.now().timestamp())
        
        # Check if the provided timestamps are from the past (like 2024)
        # If so, use a reasonable recent time range instead
        if end < now_ts - (30 * 24 * 60 * 60):  # If end is more than 30 days ago
            logger.info(f"Provided time range is too old ({start} to {end}), using recent data instead")
            # Use last 24 hours instead
            minutes_back = 1440  # 24 hours
            start_ts = now_ts - (24 * 60 * 60)
            end_ts = now_ts
        else:
            # Use provided time range
            end_ts = min(end, now_ts)  # Don't go into the future
            start_ts = max(start, now_ts - (7 * 24 * 60 * 60))  # Don't go more than 7 days back
            minutes_back = max(60, (end_ts - start_ts) // 60)  # At least 1 hour
        
        logger.info(f"Fetching real data for pool {pool_address}, minutes_back: {minutes_back}")
        
        # Fetch events from The Graph (limit to 1000 events max)
        result = await fetch_pool_events(pool_address, minutes_back, max_events=1000)
        events = result["events"]
        
        logger.info(f"Fetched {len(events)} real events from The Graph")
        
        # Convert data_service.PoolEvent objects to dictionary format expected by backtest
        converted_events = []
        current_time = int(datetime.now().timestamp())
        
        for event in events:
            # Validate timestamp (should be reasonable and not in the future)
            timestamp = event.unixTimestamp
            if timestamp > current_time + 3600:  # More than 1 hour in the future
                logger.warning(f"Event timestamp {timestamp} is in the future, adjusting to current time")
                timestamp = current_time
            elif timestamp < current_time - (365 * 24 * 60 * 60):  # More than 1 year old
                logger.warning(f"Event timestamp {timestamp} is very old, adjusting")
                timestamp = current_time - (24 * 60 * 60)  # Set to 24 hours ago
            
            # Convert to the format expected by the backtest simulation
            converted_event = {
                "amount": int(abs(event.amount) * 1000),  # Convert to integer, ensure positive
                "amount0": int(abs(event.amount0) * 1000000) if event.amount0 else 0,  # Scale for precision
                "amount1": int(abs(event.amount1) * 1000000) if event.amount1 else 0,  # Scale for precision
                "eventType": int(event.eventType.value),  # Convert enum to int
                "tickLower": event.tickLower if event.tickLower else 0,
                "tickUpper": event.tickUpper if event.tickUpper else 0,
                "unixTimestamp": timestamp
            }
            converted_events.append(converted_event)
        
        # Filter by actual time range if needed
        if start_ts < end_ts and len(converted_events) > 0:
            filtered_events = [
                event for event in converted_events
                if start_ts <= event["unixTimestamp"] <= end_ts
            ]
            if filtered_events:
                logger.info(f"Filtered to {len(filtered_events)} events in time range {start_ts} to {end_ts}")
                return filtered_events
            else:
                logger.info(f"No events found in time range {start_ts} to {end_ts}, using all {len(converted_events)} events")
        
        logger.info(f"Using all {len(converted_events)} real events (no additional time filtering)")
        
        # Save the raw and converted data for debugging
        await save_debug_data(pool_address, events, converted_events, start_ts, end_ts)
        
        return converted_events
        
    except Exception as e:
        logger.error(f"Error fetching real pool data: {e}")
        logger.info("Falling back to mock data")
        return load_mock_data()

async def save_debug_data(pool_address: str, raw_events: List, converted_events: List[Dict], start_ts: int, end_ts: int):
    """Save debug data for inspection"""
    try:
        # Create debug directory
        debug_dir = os.path.join(os.path.dirname(__file__), "debug_data")
        os.makedirs(debug_dir, exist_ok=True)
        
        # Create timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pool_short = pool_address[:10] if pool_address else "unknown"
        
        # Save raw events data
        raw_data = {
            "pool_address": pool_address,
            "timestamp": timestamp,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "event_count": len(raw_events),
            "raw_events": [
                {
                    "amount": event.amount,
                    "amount0": event.amount0,
                    "amount1": event.amount1,
                    "eventType": event.eventType.value,
                    "unixTimestamp": event.unixTimestamp,
                    "tickLower": event.tickLower,
                    "tickUpper": event.tickUpper
                } for event in raw_events
            ]
        }
        
        raw_file = os.path.join(debug_dir, f"raw_data_{pool_short}_{timestamp}.json")
        with open(raw_file, 'w') as f:
            json.dump(raw_data, f, indent=2)
        
        # Save converted events data
        converted_data = {
            "pool_address": pool_address,
            "timestamp": timestamp,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "event_count": len(converted_events),
            "converted_events": converted_events
        }
        
        converted_file = os.path.join(debug_dir, f"converted_data_{pool_short}_{timestamp}.json")
        with open(converted_file, 'w') as f:
            json.dump(converted_data, f, indent=2)
        
        logger.info(f"💾 Debug data saved to {debug_dir}:")
        logger.info(f"   📄 Raw data: {os.path.basename(raw_file)}")
        logger.info(f"   📄 Converted data: {os.path.basename(converted_file)}")
        
    except Exception as e:
        logger.error(f"Failed to save debug data: {e}")

async def get_real_pool_address(pool: str) -> str:
    """Get real pool address, either from known pools or by auto-selecting top pool"""
    # Check if it's a known pool name
    if pool.lower() in KNOWN_POOLS:
        return KNOWN_POOLS[pool.lower()]
    
    # Check if it's already a valid address
    if pool.startswith("0x") and len(pool) == 42:
        return pool
    
    # Auto-select top pool
    try:
        top_pool = await get_top_pool()
        logger.info(f"Auto-selected top pool: {top_pool['token0']['symbol']}/{top_pool['token1']['symbol']}")
        return top_pool["id"]
    except Exception as e:
        logger.error(f"Error getting top pool: {e}")
        # Fallback to USDC/WETH
        return KNOWN_POOLS["usdc-eth"]

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
    Run a backtest for the specified pool and time period using real data from The Graph

    Args:
        pool: Pool address or name to backtest
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

        # Get real pool address (handles pool names, addresses, or auto-selection)
        pool_address = await get_real_pool_address(pool)
        logger.info(f"Using pool address: {pool_address}")

        # Fetch real data from The Graph
        filtered_events = await fetch_real_pool_data(pool_address, start, end)
        
        if not filtered_events:
            raise ValueError("No events available for the specified time range")

        logger.info(f"Processing {len(filtered_events)} real events for backtest")

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

    logger.info(f"🧮 BACKTEST MATH DEBUG START")
    logger.info(f"📊 Input parameters:")
    logger.info(f"   📍 Pool: {pool}")
    logger.info(f"   💰 Position size (ETH): {pos_size_eth}")
    logger.info(f"   💸 Fee tier: {fee_tier}")
    logger.info(f"   📋 Total events: {len(events)}")

    # --- NEW 2️⃣  resolve token symbols ------------
    sym0, sym1 = POOL_TOKENS.get(pool.lower(), ("USDC", "ETH"))
    logger.info(f"   🪙 Token symbols: {sym0}/{sym1}")

    # --- step 1: split & convert units -------------------------------------
    swap_ev   = [e for e in events if e["eventType"] == 1]
    liq_ev    = [e for e in events if e["eventType"] == 0]
    
    logger.info(f"📈 STEP 1: Event Processing")
    logger.info(f"   🔄 Swap events: {len(swap_ev)}")
    logger.info(f"   💧 Liquidity events: {len(liq_ev)}")

    vol0 = vol1 = 0
    prices, ts  = [], []
    for i, s in enumerate(swap_ev):
        a0 = to_units(s["amount0"], sym0)
        a1 = to_units(s["amount1"], sym1)
        vol0 += abs(a0)
        vol1 += abs(a1)
        if a0 != 0:
            price = abs(a1)/abs(a0)
            prices.append(price)
            ts.append(s["unixTimestamp"])
            
            # Log first few swap events for debugging
            if i < 3:
                logger.info(f"   📊 Swap {i+1}: amount0={a0:.6f} {sym0}, amount1={a1:.6f} {sym1}, price={price:.2f}")

    logger.info(f"   📊 Total volume: {vol0:.3f} {sym0}, {vol1:.3f} {sym1}")
    logger.info(f"   💹 Price range: {min(prices):.2f} to {max(prices):.2f} (from {len(prices)} swaps)")

    # --- step 2: fee revenue only on your share ----------------------------
    logger.info(f"💰 STEP 2: Fee Revenue Calculation")
    last_price = prices[-1] if prices else 0
    logger.info(f"   💵 Last price: {last_price:.2f} {sym1}/{sym0}")
    
    liq_notional = sum(to_units(e["amount0"], sym0) * last_price for e in liq_ev)
    logger.info(f"   💧 Total liquidity notional: {liq_notional:.6f} ETH")
    
    my_notional  = pos_size_eth
    logger.info(f"   👤 My position notional: {my_notional:.6f} ETH")
    
    share = my_notional / (liq_notional + my_notional) if liq_notional else 0
    logger.info(f"   🥧 My share of pool: {share:.6f} ({share*100:.3f}%)")
    
    fee_rev_eth = vol1 * fee_tier * share
    logger.info(f"   💸 Fee revenue: {vol1:.3f} × {fee_tier} × {share:.6f} = {fee_rev_eth:.6f} ETH")

    # --- step 3: impermanent loss ------------------------------------------
    logger.info(f"📉 STEP 3: Impermanent Loss Calculation")
    il = 0
    if len(prices) >= 2:
        initial_price = prices[0]
        final_price = prices[-1]
        pr = final_price / initial_price
        logger.info(f"   📈 Price ratio: {final_price:.2f} / {initial_price:.2f} = {pr:.6f}")
        
        il_pct = 2*math.sqrt(pr)/(1+pr) - 1
        logger.info(f"   📐 IL percentage: 2×√{pr:.6f}/(1+{pr:.6f}) - 1 = {il_pct:.6f}")
        
        il = abs(il_pct) * pos_size_eth
        logger.info(f"   💔 Impermanent loss: {abs(il_pct):.6f} × {pos_size_eth} = {il:.6f} ETH")
    else:
        logger.info(f"   ⚠️ Not enough price data for IL calculation")

    # --- step 4: gas --------------------------------------------------------
    logger.info(f"⛽ STEP 4: Gas Costs Calculation")
    TX_MINT  = 130_000         # mint or burn once
    GAS_GWEI = 25
    gas_costs = 2 * TX_MINT * GAS_GWEI * 1e-9     # eth
    logger.info(f"   ⛽ Gas cost: 2 × {TX_MINT} × {GAS_GWEI} × 1e-9 = {gas_costs:.6f} ETH")

    # --- PnL Calculation ---
    logger.info(f"🏆 STEP 5: Final PnL Calculation")
    logger.info(f"   💰 Fee revenue: +{fee_rev_eth:.6f} ETH")
    logger.info(f"   💔 Impermanent loss: -{il:.6f} ETH")
    logger.info(f"   ⛽ Gas costs: -{gas_costs:.6f} ETH")
    
    pnl = fee_rev_eth - il - gas_costs
    logger.info(f"   🎯 PnL: {fee_rev_eth:.6f} - {il:.6f} - {gas_costs:.6f} = {pnl:.6f} ETH")

    # --- step 5: Sharpe -----------------------------------------------------
    logger.info(f"📊 STEP 6: Sharpe Ratio Calculation")
    returns = np.diff(np.log(prices))
    vol = returns.std() * math.sqrt(365*24*60) if returns.size else 0
    sharpe = pnl/pos_size_eth/vol if vol else 0
    logger.info(f"   📈 Price volatility (annualized): {vol:.6f}")
    logger.info(f"   📊 Sharpe ratio: {pnl:.6f}/{pos_size_eth}/{vol:.6f} = {sharpe:.6f}")
    
    logger.info(f"🧮 BACKTEST MATH DEBUG END")

    # Save detailed calculation results
    calculation_results = {
        "parameters": {
            "pool": pool,
            "position_size_eth": pos_size_eth,
            "fee_tier": fee_tier,
            "symbols": [sym0, sym1]
        },
        "event_analysis": {
            "total_events": len(events),
            "swap_events": len(swap_ev),
            "liquidity_events": len(liq_ev),
            "volume_token0": round(vol0, 3),
            "volume_token1": round(vol1, 3),
            "price_range": {
                "min": round(min(prices), 2) if prices else 0,
                "max": round(max(prices), 2) if prices else 0,
                "initial": round(prices[0], 2) if prices else 0,
                "final": round(prices[-1], 2) if prices else 0
            }
        },
        "calculations": {
            "fee_revenue": {
                "last_price": round(last_price, 6),
                "liquidity_notional": round(liq_notional, 6),
                "my_notional": round(my_notional, 6),
                "pool_share": round(share, 6),
                "fee_revenue_eth": round(fee_rev_eth, 6)
            },
            "impermanent_loss": {
                "price_ratio": round(pr, 6) if len(prices) >= 2 else 0,
                "il_percentage": round(il_pct, 6) if len(prices) >= 2 else 0,
                "il_eth": round(il, 6)
            },
            "gas_costs": {
                "tx_cost": TX_MINT,
                "gas_price_gwei": GAS_GWEI,
                "total_gas_eth": round(gas_costs, 6)
            },
            "risk_metrics": {
                "volatility": round(vol, 6),
                "sharpe_ratio": round(sharpe, 6)
            }
        },
        "final_results": {
            "pnl_eth": round(pnl, 6),
            "pnl_breakdown": {
                "fee_revenue": round(fee_rev_eth, 6),
                "impermanent_loss": round(-il, 6),
                "gas_costs": round(-gas_costs, 6)
            }
        }
    }
    
    await save_calculation_debug(pool, calculation_results)

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

async def save_calculation_debug(pool: str, calculation_results: Dict[str, Any]):
    """Save detailed calculation results for debugging"""
    try:
        debug_dir = os.path.join(os.path.dirname(__file__), "debug_data")
        os.makedirs(debug_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pool_short = pool[:10] if pool.startswith("0x") else pool
        
        calc_file = os.path.join(debug_dir, f"calculations_{pool_short}_{timestamp}.json")
        with open(calc_file, 'w') as f:
            json.dump(calculation_results, f, indent=2)
        
        logger.info(f"💾 Calculation debug saved: {os.path.basename(calc_file)}")
        
    except Exception as e:
        logger.error(f"Failed to save calculation debug: {e}")

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
        # Set default time range if not provided (use last 24 hours)
        if start_time is None or end_time is None:
            now_ts = int(datetime.now().timestamp())
            if start_time is None:
                start_time = now_ts - 86400  # 24 hours ago
            if end_time is None:
                end_time = now_ts

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
    Get default time period (last 24 hours)
    """
    # Use current time - 24 hours as default
    end_time = int(datetime.now(UTC).timestamp())
    start_time = end_time - (24 * 60 * 60)  # 24 hours ago
    return start_time, end_time

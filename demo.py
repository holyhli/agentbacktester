#!/usr/bin/env python3
"""
Enhanced Backtester Agent Demo

This script demonstrates the Function Calling capabilities of the Enhanced Backtester Agent.
It shows how to interact with the agent using both direct function calls and REST API.
"""

import asyncio
import requests
import time
from datetime import datetime

# Import the enhanced backtest service for direct testing
from backtest_service import (
    backtest_function_call,
    get_available_functions,
    load_mock_data,
    get_default_time_period
)

def print_header(title: str):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f"🚀 {title}")
    print("=" * 60)

def print_section(title: str):
    """Print a formatted section"""
    print(f"\n📊 {title}")
    print("-" * 40)

async def demo_direct_function_calls():
    """Demonstrate direct function calls without agent communication"""
    print_header("Direct Function Calls Demo")

    # 1. Show available functions
    print_section("Available Functions")
    functions = get_available_functions()
    for func in functions:
        print(f"🔧 {func['name']}: {func['description']}")
        params = func['parameters']['properties']
        required = func['parameters'].get('required', [])
        print(f"   Parameters: {list(params.keys())}")
        print(f"   Required: {required}")

    # 2. Load and analyze mock data
    print_section("Mock Data Analysis")
    mock_events = load_mock_data()
    if mock_events:
        swap_events = [e for e in mock_events if e.get("eventType") == 1]
        liquidity_events = [e for e in mock_events if e.get("eventType") == 0]

        print(f"📈 Total Events: {len(mock_events)}")
        print(f"🔄 Swap Events: {len(swap_events)} ({len(swap_events)/len(mock_events)*100:.1f}%)")
        print(f"💧 Liquidity Events: {len(liquidity_events)} ({len(liquidity_events)/len(mock_events)*100:.1f}%)")

        start_time, end_time = get_default_time_period()
        start_date = datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S")
        end_date = datetime.fromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S")
        print(f"⏰ Time Range: {start_date} to {end_date}")

        # Show sample events
        print(f"\n📋 Sample Events:")
        for i, event in enumerate(mock_events[:3]):
            event_type = "Swap" if event.get("eventType") == 1 else "Liquidity"
            timestamp = datetime.fromtimestamp(event.get("unixTimestamp", 0)).strftime("%H:%M:%S")
            print(f"   {i+1}. {event_type} at {timestamp} - Amount0: {event.get('amount0', 0)}")

    # 3. Basic function call
    print_section("Basic Function Call")
    print("🚀 Running basic backtest for USDC-ETH...")

    try:
        result = await backtest_function_call(pool="usdc-eth")

        if result.get("success"):
            print("✅ Backtest completed successfully!")
            print(f"💰 PnL: {result.get('pnl', 0):.6f} ETH")
            print(f"📊 Sharpe Ratio: {result.get('sharpe', 0):.2f}")
            print(f"💸 Total Fees: {result.get('total_fees', 0):.6f} ETH")
            print(f"📉 Impermanent Loss: {result.get('impermanent_loss', 0):.6f} ETH")
            print(f"⛽ Gas Costs: {result.get('gas_costs', 0):.6f} ETH")
            print(f"📈 Events Processed: {result.get('total_events', 0)}")
        else:
            print(f"❌ Backtest failed: {result.get('error_message', 'Unknown error')}")

    except Exception as e:
        print(f"❌ Error: {e}")

    # 4. Advanced function call with parameters
    print_section("Advanced Function Call")
    print("🚀 Running advanced backtest with custom parameters...")

    try:
        result = await backtest_function_call(
            pool="wbtc-eth",
            start_time=1620158974,  # From mock data
            end_time=1620243233,    # From mock data
            strategy_type="liquidity_provision",
            fee_tier=0.005,  # 0.5% fee tier
            position_size=2.0  # 2 ETH position
        )

        if result.get("success"):
            print("✅ Advanced backtest completed!")
            print(f"🏊 Pool: WBTC-ETH")
            print(f"💰 Position Size: 2.0 ETH")
            print(f"💸 Fee Tier: 0.5%")
            print(f"📊 Results:")
            print(f"   💰 PnL: {result.get('pnl', 0):.6f} ETH")
            print(f"   📊 Sharpe: {result.get('sharpe', 0):.2f}")
            print(f"   🔄 Swap Events: {result.get('swap_events', 0)}")
            print(f"   💧 Liquidity Events: {result.get('liquidity_events', 0)}")
            print(f"   📈 Volume Token0: {result.get('volume_token0', 0):.2f}")
            print(f"   📈 Volume Token1: {result.get('volume_token1', 0):.2f}")
        else:
            print(f"❌ Advanced backtest failed: {result.get('error_message', 'Unknown error')}")

    except Exception as e:
        print(f"❌ Error: {e}")

def demo_rest_api():
    """Demonstrate REST API calls (requires running agent)"""
    print_header("REST API Demo")

    base_url = "http://localhost:8000"

    # 1. Health check
    print_section("Health Check")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print("✅ Agent is healthy!")
            print(f"📊 Status: {health.get('status', 'unknown')}")
            print(f"📈 Mock Data Loaded: {health.get('mock_data_loaded', False)}")
            print(f"🔢 Total Events: {health.get('total_events', 0)}")
            print(f"🔧 Functions Available: {health.get('functions_available', 0)}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to agent: {e}")
        print("💡 Make sure the Enhanced Backtest Agent is running!")
        return False

    # 2. Get available functions
    print_section("Available Functions")
    try:
        response = requests.get(f"{base_url}/functions", timeout=5)
        if response.status_code == 200:
            functions_data = response.json()
            functions = functions_data.get('functions', [])
            print(f"🔧 Found {len(functions)} available functions:")
            for func in functions:
                print(f"   - {func['name']}: {func['description']}")
        else:
            print(f"❌ Failed to get functions: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Error getting functions: {e}")

    # 3. Run backtest via REST
    print_section("REST Backtest")
    try:
        backtest_data = {
            "pool": "usdc-eth",
            "strategy_type": "liquidity_provision",
            "fee_tier": 0.003,
            "position_size": 1.0
        }

        print("🚀 Sending backtest request...")
        response = requests.post(
            f"{base_url}/backtest",
            json=backtest_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            print("✅ REST backtest completed!")
            print(f"📊 Results:")
            print(f"   💰 PnL: {result.get('pnl', 0):.6f} ETH")
            print(f"   📊 Sharpe: {result.get('sharpe', 0):.2f}")
            print(f"   💸 Total Fees: {result.get('total_fees', 0):.6f} ETH")
            print(f"   📉 IL: {result.get('impermanent_loss', 0):.6f} ETH")
            print(f"   ⛽ Gas: {result.get('gas_costs', 0):.6f} ETH")
        else:
            print(f"❌ REST backtest failed: {response.status_code}")
            print(f"Response: {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"❌ Error running REST backtest: {e}")

    return True

async def demo_comparison():
    """Show comparison between different strategies"""
    print_header("Strategy Comparison Demo")

    strategies = [
        {"name": "Conservative", "fee_tier": 0.003, "position_size": 1.0},
        {"name": "Moderate", "fee_tier": 0.005, "position_size": 1.5},
        {"name": "Aggressive", "fee_tier": 0.01, "position_size": 2.0},
    ]

    print("🔍 Comparing different liquidity provision strategies...")

    results = []

    for strategy in strategies:
        print(f"\n📊 Testing {strategy['name']} Strategy...")
        print(f"   Fee Tier: {strategy['fee_tier']*100:.1f}%")
        print(f"   Position Size: {strategy['position_size']} ETH")

        try:
            result = await backtest_function_call(
                pool="usdc-eth",
                strategy_type="liquidity_provision",
                fee_tier=strategy['fee_tier'],
                position_size=strategy['position_size']
            )

            if result.get("success"):
                results.append({
                    "name": strategy['name'],
                    "pnl": result.get('pnl', 0),
                    "sharpe": result.get('sharpe', 0),
                    "fees": result.get('total_fees', 0),
                    "il": result.get('impermanent_loss', 0)
                })
                print(f"   ✅ PnL: {result.get('pnl', 0):.6f} ETH")
            else:
                print(f"   ❌ Failed: {result.get('error_message', 'Unknown error')}")

        except Exception as e:
            print(f"   ❌ Error: {e}")

    # Show comparison
    if results:
        print_section("Strategy Comparison Results")
        print(f"{'Strategy':<12} {'PnL (ETH)':<12} {'Sharpe':<8} {'Fees (ETH)':<12} {'IL (ETH)':<10}")
        print("-" * 60)

        for result in results:
            print(f"{result['name']:<12} {result['pnl']:<12.6f} {result['sharpe']:<8.2f} {result['fees']:<12.6f} {result['il']:<10.6f}")

        # Find best strategy
        best_pnl = max(results, key=lambda x: x['pnl'])
        best_sharpe = max(results, key=lambda x: x['sharpe'])

        print(f"\n🏆 Best PnL: {best_pnl['name']} ({best_pnl['pnl']:.6f} ETH)")
        print(f"🏆 Best Sharpe: {best_sharpe['name']} ({best_sharpe['sharpe']:.2f})")

async def main():
    """Main demo function"""
    print_header("Enhanced Backtester Agent Demo")
    print("This demo showcases the Function Calling capabilities")
    print("of the Enhanced Backtester Agent with real mock data.")

    # Demo 1: Direct function calls
    await demo_direct_function_calls()

    # Demo 2: Strategy comparison
    await demo_comparison()

    # Demo 3: REST API (if agent is running)
    print("\n" + "⏳" * 20)
    print("Waiting 2 seconds before REST API demo...")
    time.sleep(2)

    rest_success = demo_rest_api()

    # Final summary
    print_header("Demo Summary")
    print("✅ Direct Function Calls: Completed")
    print("✅ Strategy Comparison: Completed")
    print(f"{'✅' if rest_success else '❌'} REST API Demo: {'Completed' if rest_success else 'Failed (agent not running)'}")

    if not rest_success:
        print("\n💡 To test REST API functionality:")
        print("   1. Run: python agent.py")
        print("   2. Wait for agent to start")
        print("   3. Run this demo again")

    print("\n🎉 Demo completed! The Enhanced Backtester Agent is ready for:")
    print("   • LLM Function Calling integration")
    print("   • REST API access")
    print("   • Agent-to-agent communication")
    print("   • Advanced backtesting with real event data")

if __name__ == "__main__":
    print("""
🚀 Enhanced Backtester Agent Demo

This demo will show you:
  1. Direct function calls with mock data
  2. Strategy comparison analysis
  3. REST API integration (if agent is running)

📊 Using real Uniswap V3 event data (28 events)
⏰ Time range: May 2021 (1 day of activity)
🏊 Pools: USDC-ETH, WBTC-ETH

Starting demo in 3 seconds...
    """)

    time.sleep(3)
    asyncio.run(main())

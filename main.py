#!/usr/bin/env python3
"""
Main orchestrator for the UniV4 Backtesting system with uAgents
"""
import asyncio
import argparse
from data_agent import data_agent
from backtest_agent import backtest_agent, BacktestRequest
from chat_agent import chat_agent
from uagents import Context


async def run_agents():
    """Run all agents concurrently"""
    print("Starting UniV4 Backtesting Agents...")
    print(f"DataAgent will run on: http://127.0.0.1:8001")
    print(f"BacktestAgent will run on: http://127.0.0.1:8002")
    print(f"ChatAgent will run on: http://127.0.0.1:8003")
    print("\n🤖 Chat Interface Available!")
    print("You can interact with the system using natural language commands.")
    print("Example: 'backtest USDC-ETH for 1 week'")
    
    # Create tasks for all agents
    tasks = [
        asyncio.create_task(data_agent.run_async()),
        asyncio.create_task(backtest_agent.run_async()),
        asyncio.create_task(chat_agent.run_async())
    ]
    
    try:
        # Wait for all tasks to complete
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        print("\nShutting down agents...")
        # Cancel all tasks
        for task in tasks:
            task.cancel()
        # Wait for tasks to finish cancellation
        await asyncio.gather(*tasks, return_exceptions=True)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="UniV4 Backtesting with uAgents")
    parser.add_argument("--pool", type=str, help="Pool address to backtest")
    parser.add_argument("--start", type=int, help="Start timestamp")
    parser.add_argument("--end", type=int, help="End timestamp")
    
    args = parser.parse_args()
    
    if args.pool and args.start and args.end:
        print(f"Will backtest pool {args.pool} from {args.start} to {args.end}")
    
    # Run the agents
    asyncio.run(run_agents())


if __name__ == "__main__":
    main()

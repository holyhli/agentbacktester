#!/usr/bin/env python3
"""
Example client that demonstrates how to use the UniV4 Backtesting system
"""
from uagents import Agent, Context, Model
from typing import Dict, Any


class BacktestRequest(Model):
    pool: str
    start: int
    end: int
    strategy_params: Dict[str, Any] = {}


class BacktestResults(Model):
    kind: str
    pnl: float
    sharpe: float
    total_fees: float
    impermanent_loss: float
    gas_costs: float
    success: bool
    error_message: str = ""


# BacktestAgent address (you'll need to update this with the actual address)
BACKTEST_AGENT_ADDRESS = "test-agent://agent1qwquu2d237gntfugrnwch38g8jkl3n9dkeq53x64wpw5g68kr0chkv7hw50"

# Create client agent
client = Agent(
    name="BacktestClient",
    port=8003,
    seed="client_secret_phrase",
    endpoint=["http://127.0.0.1:8003/submit"],
)


@client.on_startup()
async def startup_handler(ctx: Context):
    """Send backtest request on startup"""
    ctx.logger.info(f"BacktestClient started with address: {ctx.agent.address}")
    
    # Example backtest request for ETH/USDC 0.05% pool
    request = BacktestRequest(
        pool="0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640",  # ETH/USDC 0.05%
        start=1640995200,  # Jan 1, 2022
        end=1641081600,    # Jan 2, 2022 (short period for testing)
        strategy_params={
            "strategy": "AutoCompound",
            "rebalance_threshold": 0.1
        }
    )
    
    ctx.logger.info(f"Sending backtest request for pool {request.pool}")
    await ctx.send(BACKTEST_AGENT_ADDRESS, request)


@client.on_message(model=BacktestResults)
async def handle_backtest_results(ctx: Context, sender: str, msg: BacktestResults):
    """Handle backtest results"""
    ctx.logger.info(f"Received backtest results from {sender}")
    
    if msg.success:
        ctx.logger.info("=== BACKTEST RESULTS ===")
        ctx.logger.info(f"PnL: {msg.pnl:.6f}")
        ctx.logger.info(f"Sharpe Ratio: {msg.sharpe:.4f}")
        ctx.logger.info(f"Total Fees: {msg.total_fees:.6f}")
        ctx.logger.info(f"Impermanent Loss: {msg.impermanent_loss:.6f}")
        ctx.logger.info(f"Gas Costs: {msg.gas_costs:.6f}")
        ctx.logger.info("========================")
    else:
        ctx.logger.error(f"Backtest failed: {msg.error_message}")


@client.on_interval(period=60.0)
async def periodic_check(ctx: Context):
    """Periodic health check"""
    ctx.logger.info("BacktestClient is running and waiting for results...")


if __name__ == "__main__":
    print("Starting BacktestClient...")
    print("This client will send a backtest request and wait for results.")
    print("Make sure both DataAgent and BacktestAgent are running first!")
    print("Client will run on: http://127.0.0.1:8003")
    
    client.run()
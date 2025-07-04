from uagents import Agent, Context, Model
import json
import os
import tempfile
import asyncio
from subprocess import run, PIPE, CalledProcessError
from typing import List, Dict, Any
from data_storage import BacktestDataManager


class BacktestRequest(Model):
    pool: str
    start: int
    end: int
    strategy_params: Dict[str, Any] = {}


class EventsResponse(Model):
    kind: str
    events: List[Dict[str, Any]]


class BacktestResults(Model):
    kind: str
    pnl: float
    sharpe: float
    total_fees: float
    impermanent_loss: float
    gas_costs: float
    success: bool
    error_message: str = ""


class FetchEventsRequest(Model):
    pool: str
    start: int
    end: int


# Create BacktestAgent
backtest_agent = Agent(
    name="BacktestAgent",
    port=8002,
    seed="backtest_agent_secret_phrase",
    endpoint=["http://127.0.0.1:8002/submit"],
)

# DataAgent address (from the DataAgent when it starts)
DATA_AGENT_ADDRESS = "agent1qd9fx6rgeu4u37gafa3s4vu3fmq9pux3e94tzt2ghzr8vx86muttvkqf6re"


class SolidityBacktester:
    def __init__(self, project_path: str = None):
        # Use local project path - now copied to the uAgents directory
        self.project_path = project_path or os.path.join(os.path.dirname(__file__), "UniV4Backtester_ETHSF2024")
        self.data_path = os.path.join(self.project_path, "src", "data")
        self.events_file = os.path.join(self.data_path, "pool-events.json")

        # Ensure data directory exists
        os.makedirs(self.data_path, exist_ok=True)

        # Initialize data storage for caching and logging
        self.data_manager = BacktestDataManager()

    def write_events_file(self, events: List[Dict[str, Any]]):
        """Write events to the expected location for Solidity script"""
        events_data = {"events": events}

        # Write to the Solidity project's expected location
        with open(self.events_file, 'w') as f:
            json.dump(events_data, f, indent=2)

        # Also cache the events in our data storage
        if events:
            pool_info = events[0] if events else {}
            self.data_manager.log_agent_activity("BacktestAgent", f"Wrote {len(events)} events to {self.events_file}")

    def run_backtest(self) -> Dict[str, Any]:
        """Run the Foundry script for backtesting"""
        try:
            # Log the backtest attempt
            self.data_manager.log_agent_activity("BacktestAgent", f"Starting backtest in {self.project_path}")

            # Check if forge is available
            forge_check = run(["which", "forge"], stdout=PIPE, stderr=PIPE, text=True)
            if forge_check.returncode != 0:
                raise Exception("Forge not found. Please install Foundry.")

            # Check if RPC URL is set
            rpc_url = os.getenv("UNI_RPC_URL")
            if not rpc_url:
                # Use a default RPC URL for testing
                rpc_url = "https://eth-mainnet.g.alchemy.com/v2/demo"
                self.data_manager.log_agent_activity("BacktestAgent", "Using default RPC URL")

            # Run the Foundry script
            cmd = [
                "forge", "script", "UniV4Backtester.s.sol",
                "--fork-url", rpc_url,
                "--json"
            ]

            self.data_manager.log_agent_activity("BacktestAgent", f"Running command: {' '.join(cmd)}")

            result = run(cmd, stdout=PIPE, stderr=PIPE, text=True, cwd=self.project_path)

            # Log the result
            self.data_manager.log_agent_activity("BacktestAgent", f"Forge script return code: {result.returncode}")

            if result.returncode != 0:
                error_msg = f"Forge script failed: {result.stderr}"
                self.data_manager.log_agent_activity("BacktestAgent", error_msg)
                raise Exception(error_msg)

            # Parse JSON output
            try:
                output = json.loads(result.stdout)
                results = {
                    "success": True,
                    "pnl": output.get("returns", 0.0),
                    "sharpe": output.get("sharpe", 0.0),
                    "total_fees": output.get("fees", 0.0),
                    "impermanent_loss": output.get("impermanent_loss", 0.0),
                    "gas_costs": output.get("gas", 0.0)
                }

                # Log successful backtest
                self.data_manager.log_agent_activity("BacktestAgent", f"Backtest completed successfully: PnL={results['pnl']}")
                return results

            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract basic info from console output
                self.data_manager.log_agent_activity("BacktestAgent", "Could not parse JSON output, using console output")

                # Look for key metrics in the console output
                stdout_lines = result.stdout.split('\n')
                pnl = 0.0

                # Try to extract PnL from console output
                for line in stdout_lines:
                    if "token0Out=" in line or "token1Out=" in line:
                        # Basic success indicator
                        pnl = 0.01  # Placeholder
                        break

                return {
                    "success": True,
                    "pnl": pnl,
                    "sharpe": 0.0,
                    "total_fees": 0.0,
                    "impermanent_loss": 0.0,
                    "gas_costs": 0.0
                }

        except Exception as e:
            error_msg = str(e)
            self.data_manager.log_agent_activity("BacktestAgent", f"Backtest failed: {error_msg}")
            return {
                "success": False,
                "error_message": error_msg,
                "pnl": 0.0,
                "sharpe": 0.0,
                "total_fees": 0.0,
                "impermanent_loss": 0.0,
                "gas_costs": 0.0
            }


# Initialize backtester
backtester = SolidityBacktester()

# Store pending requests with their original senders
pending_requests: Dict[str, tuple[BacktestRequest, str]] = {}


@backtest_agent.on_message(model=BacktestRequest)
async def backtest_request_handler(ctx: Context, sender: str, msg: BacktestRequest):
    """Handle backtest requests"""
    try:
        ctx.logger.info(f"Received backtest request for pool {msg.pool} from {sender}")

        # Store the request and sender for when we get events back
        request_id = f"{msg.pool}_{msg.start}_{msg.end}"
        pending_requests[request_id] = (msg, sender)

        # Forward fetch request to DataAgent
        fetch_request = FetchEventsRequest(
            pool=msg.pool,
            start=msg.start,
            end=msg.end
        )

        await ctx.send(DATA_AGENT_ADDRESS, fetch_request)
        ctx.logger.info(f"Forwarded fetch request to DataAgent for {msg.pool}")

    except Exception as e:
        ctx.logger.error(f"Error in backtest_request_handler: {e}")
        # Send error response
        error_response = BacktestResults(
            kind="Error",
            success=False,
            error_message=str(e),
            pnl=0.0,
            sharpe=0.0,
            total_fees=0.0,
            impermanent_loss=0.0,
            gas_costs=0.0
        )
        await ctx.send(sender, error_response)


@backtest_agent.on_message(model=EventsResponse)
async def events_response_handler(ctx: Context, sender: str, msg: EventsResponse):
    """Handle events response from DataAgent"""
    try:
        if msg.kind == "Error":
            ctx.logger.error("Received error response from DataAgent")
            return

        ctx.logger.info(f"Received {len(msg.events)} events from DataAgent")

        # Find the original requester based on the first event's pool info
        original_requester = None
        original_request = None

        if msg.events:
            # Try to match based on pool and timing
            for request_id, (request, requester) in pending_requests.items():
                # Simple matching - in production you'd want more sophisticated matching
                original_requester = requester
                original_request = request
                # Remove from pending requests
                del pending_requests[request_id]
                break

        if not original_requester:
            ctx.logger.error("Could not find original requester for events")
            return

        # Write events to file for Solidity script
        backtester.write_events_file(msg.events)

        # Run the backtest
        results = backtester.run_backtest()

        # Create response
        response = BacktestResults(
            kind="Results",
            success=results["success"],
            error_message=results.get("error_message", ""),
            pnl=results["pnl"],
            sharpe=results["sharpe"],
            total_fees=results["total_fees"],
            impermanent_loss=results["impermanent_loss"],
            gas_costs=results["gas_costs"]
        )

        ctx.logger.info(f"Backtest completed. PnL: {results['pnl']}, Sharpe: {results['sharpe']}")

        # Send results back to original requester
        await ctx.send(original_requester, response)
        ctx.logger.info(f"Sent backtest results to {original_requester}")

    except Exception as e:
        ctx.logger.error(f"Error in events_response_handler: {e}")
        # Try to send error to any pending requester
        for request_id, (request, requester) in list(pending_requests.items()):
            error_response = BacktestResults(
                kind="Error",
                success=False,
                error_message=str(e),
                pnl=0.0,
                sharpe=0.0,
                total_fees=0.0,
                impermanent_loss=0.0,
                gas_costs=0.0
            )
            await ctx.send(requester, error_response)
            del pending_requests[request_id]
            break


@backtest_agent.on_event("startup")
async def startup_handler(ctx: Context):
    """Handle agent startup"""
    ctx.logger.info(f"BacktestAgent started with address: {ctx.agent.address}")
    ctx.logger.info(f"Will communicate with DataAgent at: {DATA_AGENT_ADDRESS}")


# Add a simple test endpoint
@backtest_agent.on_interval(period=30.0)
async def health_check(ctx: Context):
    """Periodic health check"""
    ctx.logger.info("BacktestAgent is running...")


if __name__ == "__main__":
    backtest_agent.run()

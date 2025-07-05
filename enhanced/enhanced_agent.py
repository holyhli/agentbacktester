from enum import Enum
from typing import Dict, Any, List

from uagents import Agent, Context, Model, Field
from uagents.experimental.quota import QuotaProtocol, RateLimit
from uagents_core.models import ErrorMessage

from chat_proto import chat_proto, struct_output_client_proto
from enhanced_backtest_service import (
    run_backtest,
    BacktestRequest,
    BacktestResponse,
    backtest_function_call,
    get_available_functions,
    load_mock_data,
    get_default_time_period
)

# Enhanced agent with Function Calling capabilities
agent = Agent(
    name="enhanced_backtest_agent",
    seed="enhanced_backtest_agent_unique_seed_phrase_2024",
    mailbox=True
)

# Rate limiting protocol
proto = QuotaProtocol(
    storage_reference=agent.storage,
    name="Enhanced-Backtest-Protocol",
    version="1.0.0",
    default_rate_limit=RateLimit(window_size_minutes=60, max_requests=50),
)

# Function calling models
class FunctionCall(Model):
    """Function call request model"""
    function_name: str = Field(description="Name of the function to call")
    parameters: Dict[str, Any] = Field(description="Function parameters")

class FunctionCallResponse(Model):
    """Function call response model"""
    function_name: str = Field(description="Name of the called function")
    result: Dict[str, Any] = Field(description="Function execution result")
    success: bool = Field(description="Whether the function call succeeded")
    error_message: str = Field(default="", description="Error message if failed")

class AvailableFunctions(Model):
    """Available functions list"""
    functions: List[Dict[str, Any]] = Field(description="List of available functions")

class GetFunctionsRequest(Model):
    """Request to get available functions"""
    pass

# Enhanced backtest request handler
@proto.on_message(
    BacktestRequest, replies={BacktestResponse, ErrorMessage}
)
async def handle_backtest_request(ctx: Context, sender: str, msg: BacktestRequest):
    ctx.logger.info(f"Received enhanced backtest request for pool: {msg.pool}")
    try:
        result = await run_backtest(msg.pool, msg.start, msg.end, msg.strategy_params)
        ctx.logger.info(f"Successfully completed enhanced backtest for {msg.pool}")
        await ctx.send(sender, BacktestResponse(**result))
    except Exception as err:
        ctx.logger.error(f"Enhanced backtest failed: {err}")
        await ctx.send(sender, ErrorMessage(error=str(err)))

# Function calling handler
@proto.on_message(
    FunctionCall, replies={FunctionCallResponse, ErrorMessage}
)
async def handle_function_call(ctx: Context, sender: str, msg: FunctionCall):
    ctx.logger.info(f"Received function call: {msg.function_name}")

    try:
        if msg.function_name == "backtest_function_call":
            # Extract parameters
            params = msg.parameters
            pool = params.get("pool")
            start_time = params.get("start_time")
            end_time = params.get("end_time")
            strategy_type = params.get("strategy_type", "liquidity_provision")
            fee_tier = params.get("fee_tier", 0.003)
            position_size = params.get("position_size", 1.0)

            if not pool:
                raise ValueError("Pool parameter is required")

            # Call the function
            result = await backtest_function_call(
                pool=pool,
                start_time=start_time,
                end_time=end_time,
                strategy_type=strategy_type,
                fee_tier=fee_tier,
                position_size=position_size
            )

            response = FunctionCallResponse(
                function_name=msg.function_name,
                result=result,
                success=result.get("success", False)
            )

            ctx.logger.info(f"Function call completed successfully: {msg.function_name}")
            await ctx.send(sender, response)

        else:
            raise ValueError(f"Unknown function: {msg.function_name}")

    except Exception as err:
        ctx.logger.error(f"Function call failed: {err}")
        error_response = FunctionCallResponse(
            function_name=msg.function_name,
            result={},
            success=False,
            error_message=str(err)
        )
        await ctx.send(sender, error_response)

# Get available functions handler
@proto.on_message(
    GetFunctionsRequest, replies={AvailableFunctions}
)
async def handle_get_functions(ctx: Context, sender: str, msg: GetFunctionsRequest):
    ctx.logger.info("Received request for available functions")

    functions = get_available_functions()
    response = AvailableFunctions(functions=functions)

    await ctx.send(sender, response)

# Startup handler with enhanced logging
@agent.on_event("startup")
async def startup(ctx: Context):
    ctx.logger.info(f"🚀 Enhanced Backtest Agent {agent.name} started: {agent.address}")
    ctx.logger.info("📊 Enhanced backtest agent with Function Calling is ready")

    # Load and log mock data info
    try:
        mock_events = load_mock_data()
        if mock_events:
            start_time, end_time = get_default_time_period()
            ctx.logger.info(f"📈 Loaded {len(mock_events)} mock events")
            ctx.logger.info(f"⏰ Time range: {start_time} to {end_time}")

            # Log event type breakdown
            swap_events = len([e for e in mock_events if e.get("eventType") == 1])
            liquidity_events = len([e for e in mock_events if e.get("eventType") == 0])
            ctx.logger.info(f"📊 Event breakdown: {swap_events} swaps, {liquidity_events} liquidity events")
        else:
            ctx.logger.warning("⚠️ No mock data loaded")
    except Exception as e:
        ctx.logger.error(f"❌ Error loading mock data: {e}")

    # Log available functions
    functions = get_available_functions()
    ctx.logger.info(f"🔧 Available functions: {len(functions)}")
    for func in functions:
        ctx.logger.info(f"  - {func['name']}: {func['description']}")

@agent.on_event("shutdown")
async def shutdown(ctx: Context):
    ctx.logger.info("🛑 Enhanced backtest agent shutting down...")
    ctx.logger.info("🧹 Cleaning up resources...")

# Health check functionality
def agent_is_healthy() -> bool:
    """
    Enhanced health check logic
    """
    try:
        # Check if mock data is available
        mock_events = load_mock_data()
        if not mock_events:
            return False

        # Check if functions are available
        functions = get_available_functions()
        if not functions:
            return False

        return True
    except Exception:
        return False

class HealthCheck(Model):
    pass

class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"

class AgentHealth(Model):
    agent_name: str = Field(description="Name of the agent")
    status: HealthStatus = Field(description="Health status")
    mock_data_loaded: bool = Field(default=False, description="Whether mock data is loaded")
    total_events: int = Field(default=0, description="Total number of mock events")
    functions_available: int = Field(default=0, description="Number of available functions")

health_protocol = QuotaProtocol(
    storage_reference=agent.storage,
    name="Enhanced-HealthProtocol",
    version="1.0.0"
)

@health_protocol.on_message(HealthCheck, replies={AgentHealth})
async def handle_health_check(ctx: Context, sender: str, msg: HealthCheck):
    status = HealthStatus.UNHEALTHY
    mock_data_loaded = False
    total_events = 0
    functions_available = 0

    try:
        if agent_is_healthy():
            status = HealthStatus.HEALTHY

            # Get mock data info
            mock_events = load_mock_data()
            if mock_events:
                mock_data_loaded = True
                total_events = len(mock_events)

            # Get functions info
            functions = get_available_functions()
            functions_available = len(functions)

    except Exception as err:
        ctx.logger.error(f"Health check error: {err}")

    health_response = AgentHealth(
        agent_name="enhanced_backtest_agent",
        status=status,
        mock_data_loaded=mock_data_loaded,
        total_events=total_events,
        functions_available=functions_available
    )

    await ctx.send(sender, health_response)

# REST API endpoints for direct access
@agent.on_rest_get("/health", AgentHealth)
async def health_endpoint(ctx: Context) -> AgentHealth:
    """REST endpoint for health check"""
    ctx.logger.info("🏥 REST health check requested")

    status = HealthStatus.HEALTHY if agent_is_healthy() else HealthStatus.UNHEALTHY
    mock_data_loaded = False
    total_events = 0
    functions_available = 0

    try:
        mock_events = load_mock_data()
        if mock_events:
            mock_data_loaded = True
            total_events = len(mock_events)

        functions = get_available_functions()
        functions_available = len(functions)
    except Exception as e:
        ctx.logger.error(f"Health endpoint error: {e}")

    return AgentHealth(
        agent_name="enhanced_backtest_agent",
        status=status,
        mock_data_loaded=mock_data_loaded,
        total_events=total_events,
        functions_available=functions_available
    )

@agent.on_rest_get("/functions", AvailableFunctions)
async def functions_endpoint(ctx: Context) -> AvailableFunctions:
    """REST endpoint to get available functions"""
    ctx.logger.info("🔧 REST functions list requested")

    functions = get_available_functions()
    return AvailableFunctions(functions=functions)

class BacktestRequestREST(Model):
    """REST-specific backtest request model"""
    pool: str = Field(description="Pool address or name")
    start_time: int = Field(default=0, description="Start timestamp (optional)")
    end_time: int = Field(default=0, description="End timestamp (optional)")
    strategy_type: str = Field(default="liquidity_provision", description="Strategy type")
    fee_tier: float = Field(default=0.003, description="Fee tier")
    position_size: float = Field(default=1.0, description="Position size")

@agent.on_rest_post("/backtest", BacktestRequestREST, BacktestResponse)
async def backtest_endpoint(ctx: Context, request: BacktestRequestREST) -> BacktestResponse:
    """REST endpoint for running backtests"""
    ctx.logger.info(f"🚀 REST backtest requested for pool: {request.pool}")

    try:
        # Use default time range if not provided
        start_time = request.start_time
        end_time = request.end_time

        if start_time == 0 or end_time == 0:
            default_start, default_end = get_default_time_period()
            start_time = start_time or default_start
            end_time = end_time or default_end

        # Run backtest using function call interface
        result = await backtest_function_call(
            pool=request.pool,
            start_time=start_time,
            end_time=end_time,
            strategy_type=request.strategy_type,
            fee_tier=request.fee_tier,
            position_size=request.position_size
        )

        return BacktestResponse(**result)

    except Exception as e:
        ctx.logger.error(f"REST backtest failed: {e}")
        return BacktestResponse(
            success=False,
            error_message=str(e),
            pnl=0.0,
            sharpe=0.0,
            total_fees=0.0,
            impermanent_loss=0.0,
            gas_costs=0.0
        )

# Include all protocols
agent.include(proto, publish_manifest=True)
agent.include(health_protocol, publish_manifest=True)
agent.include(chat_proto, publish_manifest=True)
agent.include(struct_output_client_proto, publish_manifest=True)

if __name__ == "__main__":
    print("""
🤖 Starting Enhanced Backtest Agent...

🔧 Features:
  • Function Calling interface for LLM integration
  • Advanced backtest simulation with real event data
  • REST API endpoints for direct access
  • Enhanced health monitoring
  • Mock data processing with 28 events

📡 REST Endpoints:
  • GET  /health    - Health check
  • GET  /functions - Available functions
  • POST /backtest  - Run backtest

💬 Agent Communication:
  • BacktestRequest/Response - Standard backtest
  • FunctionCall/Response - Function calling interface
  • HealthCheck/Status - Health monitoring

🛑 Stop with Ctrl+C
    """)
    agent.run()

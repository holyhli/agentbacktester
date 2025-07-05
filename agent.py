import os
from enum import Enum

from uagents import Agent, Context, Model
from uagents.experimental.quota import QuotaProtocol, RateLimit
from uagents_core.models import ErrorMessage

from chat_proto import chat_proto, struct_output_client_proto
from backtest_service import run_backtest, BacktestRequest, BacktestResponse

agent = Agent(
    name="backtest_agent",
    seed="backtest_agent_unique_seed_phrase_2024",
    mailbox=True  # Enable for Agentverse integration
)

proto = QuotaProtocol(
    storage_reference=agent.storage,
    name="Backtest-Protocol",
    version="0.1.0",
    default_rate_limit=RateLimit(window_size_minutes=60, max_requests=30),
)

@proto.on_message(
    BacktestRequest, replies={BacktestResponse, ErrorMessage}
)
async def handle_request(ctx: Context, sender: str, msg: BacktestRequest):
    ctx.logger.info(f"Received backtest request for pool: {msg.pool}")
    try:
        result = await run_backtest(msg.pool, msg.start, msg.end, msg.strategy_params)
        ctx.logger.info(f"Successfully completed backtest for {msg.pool}")
        await ctx.send(sender, BacktestResponse(**result))
    except Exception as err:
        ctx.logger.error(err)
        await ctx.send(sender, ErrorMessage(error=str(err)))

# ALWAYS include startup/shutdown handlers
@agent.on_event("startup")
async def startup(ctx: Context):
    ctx.logger.info(f"Agent {agent.name} started: {agent.address}")
    ctx.logger.info("Backtest agent with mailbox functionality is ready")

@agent.on_event("shutdown")
async def shutdown(ctx: Context):
    ctx.logger.info("Cleaning up resources...")

agent.include(proto, publish_manifest=True)

### Health check related code
def agent_is_healthy() -> bool:
    """
    Implement the actual health check logic here.
    For example, check if the agent can connect to data sources.
    """
    try:
        # Add actual health check logic here
        return True
    except Exception:
        return False

class HealthCheck(Model):
    pass

class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"

class AgentHealth(Model):
    agent_name: str
    status: HealthStatus

health_protocol = QuotaProtocol(
    storage_reference=agent.storage, name="HealthProtocol", version="0.1.0"
)

@health_protocol.on_message(HealthCheck, replies={AgentHealth})
async def handle_health_check(ctx: Context, sender: str, msg: HealthCheck):
    status = HealthStatus.UNHEALTHY
    try:
        if agent_is_healthy():
            status = HealthStatus.HEALTHY
    except Exception as err:
        ctx.logger.error(err)
    finally:
        await ctx.send(sender, AgentHealth(agent_name="backtest_agent", status=status))

agent.include(health_protocol, publish_manifest=True)
agent.include(chat_proto, publish_manifest=True)
agent.include(struct_output_client_proto, publish_manifest=True)

if __name__ == "__main__":
    agent.run()
from datetime import datetime
from uuid import uuid4
from typing import Any

from uagents import Context, Model, Protocol

# Import the necessary components of the chat protocol
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    EndSessionContent,
    StartSessionContent,
    TextContent,
    chat_protocol_spec,
)

from backtest_service import run_backtest, BacktestRequest

# AI Agent Address for structured output processing
AI_AGENT_ADDRESS = 'agent1q0h70caed8ax769shpemapzkyk65uscw4xwk6dc4t3emvp5jdcvqs9xs32y'

if not AI_AGENT_ADDRESS:
    raise ValueError("AI_AGENT_ADDRESS not set")

def create_text_chat(text: str, end_session: bool = False) -> ChatMessage:
    content = [TextContent(type="text", text=text)]
    if end_session:
        content.append(EndSessionContent(type="end-session"))
    return ChatMessage(
        timestamp=datetime.utcnow(),
        msg_id=uuid4(),
        content=content,
    )

chat_proto = Protocol(spec=chat_protocol_spec)
struct_output_client_proto = Protocol(
    name="StructuredOutputClientProtocol", version="0.1.0"
)

class StructuredOutputPrompt(Model):
    prompt: str
    output_schema: dict[str, Any]

class StructuredOutputResponse(Model):
    output: dict[str, Any]

@chat_proto.on_message(ChatMessage)
async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
    ctx.logger.info(f"Got a message from {sender}: {msg}")
    ctx.storage.set(str(ctx.session), sender)
    await ctx.send(
        sender,
        ChatAcknowledgement(timestamp=datetime.utcnow(), acknowledged_msg_id=msg.msg_id),
    )

    for item in msg.content:
        if isinstance(item, StartSessionContent):
            ctx.logger.info(f"Got a start session message from {sender}")
            welcome_text = """🤖 **Welcome to UniV4 Backtest Agent!**

I can help you run backtests on Uniswap V4 pools! Here are some examples:

• "Backtest USDC-ETH pool for the last week"
• "Run a simulation on WBTC-ETH for 30 days"
• "Test pool 0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640 for 1 month"

Just describe what you want to backtest and I'll help you!"""
            await ctx.send(sender, create_text_chat(welcome_text))
            continue
        elif isinstance(item, TextContent):
            ctx.logger.info(f"Got a message from {sender}: {item.text}")
            ctx.storage.set(str(ctx.session), sender)
            await ctx.send(
                AI_AGENT_ADDRESS,
                StructuredOutputPrompt(
                    prompt=item.text, output_schema=BacktestRequest.schema()
                ),
            )
        else:
            ctx.logger.info(f"Got unexpected content from {sender}")

@chat_proto.on_message(ChatAcknowledgement)
async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    ctx.logger.info(
        f"Got an acknowledgement from {sender} for {msg.acknowledged_msg_id}"
    )

@struct_output_client_proto.on_message(StructuredOutputResponse)
async def handle_structured_output_response(
    ctx: Context, sender: str, msg: StructuredOutputResponse
):
    session_sender = ctx.storage.get(str(ctx.session))
    if session_sender is None:
        ctx.logger.error(
            "Discarding message because no session sender found in storage"
        )
        return

    if "<UNKNOWN>" in str(msg.output):
        await ctx.send(
            session_sender,
            create_text_chat(
                "Sorry, I couldn't process your backtest request. Please specify a pool (like USDC-ETH) and time period (like 'last week')."
            ),
        )
        return

    try:
        # Parse the structured output to get the backtest parameters
        backtest_request = BacktestRequest.parse_obj(msg.output)
        
        if not backtest_request.pool:
            await ctx.send(
                session_sender,
                create_text_chat(
                    "Sorry, I couldn't identify the pool. Please specify a pool like 'USDC-ETH' or provide a pool address."
                ),
            )
            return
        
        # Send confirmation message
        pool_display = backtest_request.pool[:10] + "..." if len(backtest_request.pool) > 10 else backtest_request.pool
        start_date = datetime.fromtimestamp(backtest_request.start).strftime('%Y-%m-%d')
        end_date = datetime.fromtimestamp(backtest_request.end).strftime('%Y-%m-%d')
        
        await ctx.send(
            session_sender, 
            create_text_chat(f"🚀 Starting backtest for pool {pool_display} from {start_date} to {end_date}. This may take a few moments...")
        )
        
        # Run the backtest
        result = await run_backtest(
            backtest_request.pool, 
            backtest_request.start, 
            backtest_request.end, 
            backtest_request.strategy_params or {}
        )
        
        # Format the results
        if result.get('success', False):
            response_text = f"""🎉 **Backtest Complete!**

📊 **Results Summary:**
• **PnL**: {result.get('pnl', 0):.4f} ({'+' if result.get('pnl', 0) >= 0 else ''}{result.get('pnl', 0)*100:.2f}%)
• **Sharpe Ratio**: {result.get('sharpe', 0):.2f}
• **Total Fees**: {result.get('total_fees', 0):.6f} ETH
• **Impermanent Loss**: {result.get('impermanent_loss', 0):.4f}
• **Gas Costs**: {result.get('gas_costs', 0):.6f} ETH

{'🟢 Profitable Strategy!' if result.get('pnl', 0) > 0 else '🔴 Loss-making Strategy' if result.get('pnl', 0) < 0 else '⚪ Break-even Strategy'}

Want to run another backtest? Just ask!"""
        else:
            error_msg = result.get('error_message', 'Unknown error occurred')
            response_text = f"❌ **Backtest Failed**\n\nError: {error_msg}\n\nPlease try again with different parameters."
        
        # Send the results back to the user
        await ctx.send(session_sender, create_text_chat(response_text))
        
    except Exception as err:
        ctx.logger.error(err)
        await ctx.send(
            session_sender,
            create_text_chat(
                "Sorry, I couldn't complete the backtest. Please try again later or check your parameters."
            ),
        )
        return
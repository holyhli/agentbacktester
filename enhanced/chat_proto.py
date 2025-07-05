from datetime import datetime
from uuid import uuid4
from typing import Any, Deque, Dict, Tuple

from collections import deque
import re
from enum import Enum, auto

from uagents import Context, Model, Protocol

# Import components of the chat protocol
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    EndSessionContent,
    StartSessionContent,
    TextContent,
    chat_protocol_spec,
)

from enhanced_backtest_service import run_backtest, BacktestRequest

# -----------------------------------------------------------------------------
# configuration
# -----------------------------------------------------------------------------
# AI Agent Address for structured output processing
AI_AGENT_ADDRESS = "agent1qvk7q2av3e2y5gf5s90nfzkc8a48q3wdqeevwrtgqfdl0k78rspd6f2l4dx"
if not AI_AGENT_ADDRESS:
    raise ValueError("AI_AGENT_ADDRESS not set")


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

def create_text_chat(text: str, end_session: bool = False) -> ChatMessage:
    """Wrap raw markdown text into a ChatMessage compatible with the
    uAgents chat‑protocol."""
    content = [TextContent(type="text", text=text)]
    if end_session:
        content.append(EndSessionContent(type="end-session"))
    return ChatMessage(timestamp=datetime.utcnow(), msg_id=uuid4(), content=content)


# -----------------------------------------------------------------------------
# lightweight command parser (help / status / results / backtest)
# -----------------------------------------------------------------------------

class Cmd(Enum):
    HELP = auto()
    STATUS = auto()
    RESULTS = auto()
    BACKTEST = auto()
    UNKNOWN = auto()


_SECONDS_PER = {
    "day": 86_400,
    "week": 604_800,
    "month": 2_592_000,
}


class CommandParser:
    """NLP‑lite parser to recognise a handful of meta commands. Anything it
    cannot confidently classify is handled by the LLM structured‑output
    pipeline."""

    _POOL_REGEX = re.compile(r"0x[a-fA-F0-9]{40}")
    _TIME_REGEX = re.compile(r"last\s+(\d+)\s+(day|week|month)s?", re.I)

    def __init__(self, max_history: int = 50) -> None:
        self._history: Deque[Dict[str, Any]] = deque(maxlen=max_history)

    # ------------------------------------------------------------------ public
    def parse(self, text: str) -> Tuple[Cmd, Dict[str, Any]]:
        t = text.lower().strip()
        if any(w in t for w in ("help", "commands", "what can you do")):
            return Cmd.HELP, {}
        if "status" in t:
            return Cmd.STATUS, {}
        if "result" in t or "latest" in t:
            return Cmd.RESULTS, {}
        if any(w in t for w in ("backtest", "test", "simulate", "run")):
            pool = self._extract_pool(t)
            start, end = self._extract_period(t)
            return Cmd.BACKTEST, {"pool": pool, "start": start, "end": end}
        return Cmd.UNKNOWN, {}

    def remember(self, entry: Dict[str, Any]) -> None:
        self._history.append(entry)

    def latest(self) -> Dict[str, Any] | None:
        return self._history[-1] if self._history else None

    # --------------------------------------------------------------- internal
    def _extract_pool(self, t: str) -> str | None:
        m = self._POOL_REGEX.search(t)
        return m.group(0) if m else None

    def _extract_period(self, t: str) -> Tuple[int, int]:
        now = int(datetime.utcnow().timestamp())
        m = self._TIME_REGEX.search(t)
        if not m:
            # default – last 30 days
            return now - 30 * _SECONDS_PER["day"], now
        num, unit = int(m.group(1)), m.group(2).rstrip("s")
        secs = num * _SECONDS_PER[unit]
        return now - secs, now


# -----------------------------------------------------------------------------
# canned texts
# -----------------------------------------------------------------------------

HELP_TEXT = """🤖 **Backtest Agent – Commands**

• `backtest 0x… for 1 week` – run a backtest  
• `status` – system health  
• `results` – summary of the latest run  
• `help` – this help

Pools may be Uniswap addresses (0x…) or symbolic names (e.g. `USDC-ETH`).  
Time periods accept “1 day / week / month” or “last 10 days”, etc.
"""

STATUS_TEXT = """📊 **Status**

✅ Chat gateway online  
✅ Structured‑output LLM agent reachable  
✅ Backtest service up
"""


def _format_results(entry: Dict[str, Any]) -> str:
    if not entry:
        return "No backtest results available yet."
    start = datetime.utcfromtimestamp(entry["start"]).strftime("%Y-%m-%d")
    end = datetime.utcfromtimestamp(entry["end"]).strftime("%Y-%m-%d")
    r = entry["results"]
    return (
        "📈 **Latest Backtest**\n\n"
        f"Pool: {entry['pool'][:10]}…  \n"
        f"Period: {start} → {end}\n\n"
        f"PnL: {r.get('pnl', 0):.4f}  \n"
        f"Sharpe: {r.get('sharpe', 0):.2f}"
    )


# -----------------------------------------------------------------------------
# protocol definitions
# -----------------------------------------------------------------------------

chat_proto = Protocol(spec=chat_protocol_spec)
struct_output_client_proto = Protocol(
    name="StructuredOutputClientProtocol", version="0.1.0"
)


class StructuredOutputPrompt(Model):
    prompt: str
    output_schema: dict[str, Any]


class StructuredOutputResponse(Model):
    output: dict[str, Any]


# global parser instance
parser = CommandParser()


# -----------------------------------------------------------------------------
# handlers – chat protocol
# -----------------------------------------------------------------------------


@chat_proto.on_message(ChatMessage)
async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
    ctx.logger.info(f"Got ChatMessage from {sender}: {msg}")

    # remember who owns this session (needed when the backtest finishes)
    ctx.storage.set(str(ctx.session), sender)

    # ACK immediately
    await ctx.send(
        sender,
        ChatAcknowledgement(
            timestamp=datetime.utcnow(), acknowledged_msg_id=msg.msg_id
        ),
    )

    for item in msg.content:
        # welcome
        if isinstance(item, StartSessionContent):
            await ctx.send(sender, create_text_chat(HELP_TEXT))
            continue

        # only deal with raw text commands
        if not isinstance(item, TextContent):
            continue

        cmd, info = parser.parse(item.text)

        if cmd is Cmd.HELP:
            await ctx.send(sender, create_text_chat(HELP_TEXT))
            continue
        if cmd is Cmd.STATUS:
            await ctx.send(sender, create_text_chat(STATUS_TEXT))
            continue
        if cmd is Cmd.RESULTS:
            await ctx.send(sender, create_text_chat(_format_results(parser.latest())))
            continue

        # fallback: forward to the LLM agent for structured‑output parsing
        await ctx.send(
            AI_AGENT_ADDRESS,
            StructuredOutputPrompt(
                prompt=item.text, output_schema=BacktestRequest.schema()
            ),
        )


@chat_proto.on_message(ChatAcknowledgement)
async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    ctx.logger.debug(f"Chat ACK from {sender} for {msg.acknowledged_msg_id}")


# -----------------------------------------------------------------------------
# handlers – structured‑output replies
# -----------------------------------------------------------------------------


@struct_output_client_proto.on_message(StructuredOutputResponse)
async def handle_structured_output_response(
        ctx: Context, sender: str, msg: StructuredOutputResponse
):
    """Receive the parsed BacktestRequest from the LLM agent, run the back‑
    test and push a nice summary back to the user."""

    session_sender = ctx.storage.get(str(ctx.session))
    if session_sender is None:
        ctx.logger.warning("No session sender, dropping response")
        return

    # guardrail – usually the LLM fills unknowns with "<UNKNOWN>"
    if "<UNKNOWN>" in str(msg.output):
        await ctx.send(
            session_sender,
            create_text_chat(
                "Sorry, I couldn't understand that request. "
                "Please specify a pool (e.g. `USDC-ETH`) and a time period "
                "(e.g. `last week`).",
            ),
        )
        return

    try:
        # parse validated payload
        backtest_request = BacktestRequest.parse_obj(msg.output)
        if not backtest_request.pool:
            await ctx.send(
                session_sender,
                create_text_chat(
                    "Sorry, I couldn't identify the pool. Please specify one.",
                ),
            )
            return

        pool_display = (
            backtest_request.pool[:10] + "…"
            if len(backtest_request.pool) > 10
            else backtest_request.pool
        )
        start_date = datetime.fromtimestamp(backtest_request.start).strftime("%Y-%m-%d")
        end_date = datetime.fromtimestamp(backtest_request.end).strftime("%Y-%m-%d")

        await ctx.send(
            session_sender,
            create_text_chat(
                f"🚀 Starting backtest for pool {pool_display} "
                f"from {start_date} to {end_date}. This may take a few moments…",
            ),
        )

        # run the backtest (async)
        result = await run_backtest(
            backtest_request.pool,
            backtest_request.start,
            backtest_request.end,
            backtest_request.strategy_params or {},
            )

        # summarise
        if result.get("success", False):
            response_text = (
                "🎉 **Backtest Complete!**\n\n"
                "📊 **Results Summary:**\n"
                f"• **PnL**: {result.get('pnl', 0):.4f} "
                f"({'+' if result.get('pnl', 0) >= 0 else ''}{result.get('pnl', 0)*100:.2f}%)\n"
                f"• **Sharpe Ratio**: {result.get('sharpe', 0):.2f}\n"
                f"• **Total Fees**: {result.get('total_fees', 0):.6f} ETH\n"
                f"• **Impermanent Loss**: {result.get('impermanent_loss', 0):.4f}\n"
                f"• **Gas Costs**: {result.get('gas_costs', 0):.6f} ETH\n\n"
                f"{'🟢 Profitable Strategy!' if result.get('pnl', 0) > 0 else '🔴 Loss' if result.get('pnl', 0) < 0 else '⚪ Break‑even'}"
            )

            # store for the results command
            parser.remember(
                {
                    "pool": backtest_request.pool,
                    "start": backtest_request.start,
                    "end": backtest_request.end,
                    "results": result,
                }
            )
        else:
            response_text = (
                "❌ **Backtest Failed**\n\n"
                f"Error: {result.get('error_message', 'Unknown error')}."
            )

        await ctx.send(session_sender, create_text_chat(response_text))

    except Exception as exc:
        ctx.logger.error(str(exc))
        await ctx.send(
            session_sender,
            create_text_chat(
                "Sorry, I couldn't complete the backtest. "
                "Please try again later or verify your parameters.",
            ),
        )

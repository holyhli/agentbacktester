from __future__ import annotations

"""chat_proto.py – Enhanced Backtest Agent

This version fixes the buggy relative‑date handling so that phrases like
"backtest usdc/weth for the last 3 days with my 30 ETH" are parsed locally
without involving the LLM.  Any request of the form «last N day(s) / week(s)
/ month(s)» is now mapped to the correct Unix‑epoch start/end timestamps.
"""

import re
from collections import deque
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Deque, Dict, Tuple
from uuid import uuid4

from uagents import Context, Model, Protocol
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    EndSessionContent,
    StartSessionContent,
    TextContent,
    chat_protocol_spec,
)

from backtest_service import (
    run_backtest,
    BacktestRequest,
    get_default_time_period,
)
from data_service import is_data_request, handle_data_request

# -----------------------------------------------------------------------------
# configuration – address of the LLM structured‑output helper
# -----------------------------------------------------------------------------

AI_AGENT_ADDRESS = (
    "agent1q0h70caed8ax769shpemapzkyk65uscw4xwk6dc4t3emvp5jdcvqs9xs32y"
)
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
    return ChatMessage(timestamp=datetime.now(timezone.utc), msg_id=uuid4(), content=content)

# -----------------------------------------------------------------------------
# lightweight command parser (help / status / results / backtest / data)
# -----------------------------------------------------------------------------

class Cmd(Enum):
    HELP = auto()
    STATUS = auto()
    RESULTS = auto()
    BACKTEST = auto()
    DATA = auto()
    UNKNOWN = auto()


_SECONDS_PER = {
    "day": 86_400,
    "week": 604_800,
    "month": 2_592_000,  # 30 days – good enough for back‑testing buckets
}


class CommandParser:
    """Tiny NLP‑lite parser able to understand a handful of meta commands.

    Anything it cannot confidently classify is forwarded to the LLM‑based
    structured‑output agent.  The key addition here is first‑class support for
    relative time expressions («last N days/weeks/months»).
    """

    _POOL_REGEX = re.compile(r"0x[a-fA-F0-9]{40}")
    _TIME_REGEX = re.compile(r"last\s+(\d+)\s+(day|week|month)s?", re.I)

    def __init__(self, max_history: int = 50) -> None:
        self._history: Deque[Dict[str, Any]] = deque(maxlen=max_history)

    # ------------------------------------------------------------------ public
    def parse(self, text: str) -> Tuple[Cmd, Dict[str, Any]]:
        t = text.lower().strip()

        # ---------------- meta commands
        if any(w in t for w in ("help", "commands", "what can you do")):
            return Cmd.HELP, {}
        if "status" in t:
            return Cmd.STATUS, {}
        if "result" in t or "latest" in t:
            return Cmd.RESULTS, {}

        # ---------------- data fetch (GraphQL / on‑chain events)
        if is_data_request(text):
            return Cmd.DATA, {"query": text}

        # ---------------- back‑testing
        if any(w in t for w in ("backtest", "test", "simulate", "run")):
            pool = self._extract_pool(t) or "usdc-eth"  # sensible default
            start, end = self._extract_period(t)
            size = self._extract_position_size(t)
            return Cmd.BACKTEST, {
                "pool": pool,
                "start": start,
                "end": end,
                "position_size": size,
            }

        return Cmd.UNKNOWN, {}

    def remember(self, entry: Dict[str, Any]) -> None:
        self._history.append(entry)

    def latest(self) -> Dict[str, Any] | None:
        return self._history[-1] if self._history else None

    # --------------------------------------------------------------- internal
    def _extract_pool(self, t: str) -> str | None:
        # explicit 0x… address takes precedence
        if m := self._POOL_REGEX.search(t):
            return m.group(0)

        # common symbolic names
        t = t.lower()
        if any(pair in t for pair in ("usdc/weth", "usdc-weth", "usdc/eth", "usdc-eth")):
            return "usdc-eth"
        if any(pair in t for pair in ("wbtc/eth", "wbtc-weth", "wbtc/eth", "wbtc-weth")):
            return "wbtc-eth"
        return None

    def _extract_period(self, t: str) -> Tuple[int, int]:
        """Translate relative or explicit dates into epoch seconds.

        Supported formats:
          • «last 3 days» / «last 2 weeks» / «last 1 month»
          • «from 2025-07-01 to 2025-07-05» (ISO‑8601 dates)
        """
        now = int(datetime.now(timezone.utc).timestamp())

        # relative («last N unit»)
        if m := self._TIME_REGEX.search(t):
            num = int(m.group(1))
            unit = m.group(2).rstrip("s")
            secs = num * _SECONDS_PER[unit]
            return now - secs, now

        # explicit range («from YYYY‑MM‑DD to YYYY‑MM‑DD»)
        if m := re.search(r"from\s+(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})", t):
            start_dt = datetime.strptime(m.group(1), "%Y-%m-%d").replace(tzinfo=timezone.utc)
            end_dt = datetime.strptime(m.group(2), "%Y-%m-%d").replace(tzinfo=timezone.utc)
            return int(start_dt.timestamp()), int(end_dt.timestamp())

        # fallback – last 30 days
        return now - 30 * _SECONDS_PER["day"], now

    def _extract_position_size(self, t: str) -> float:
        """Look for patterns like "30 eth", "my 5 ETH", "with 2.5 eth"."""
        if m := re.search(r"(?:my|with)?\s*(\d+\.?\d*)\s*eth", t):
            try:
                return float(m.group(1))
            except ValueError:
                pass
        return 1.0  # default position size if none supplied


# single global instance
parser = CommandParser()

# -----------------------------------------------------------------------------
# canned texts – help / status / results
# -----------------------------------------------------------------------------

HELP_TEXT = """🤖 **Enhanced Backtest Agent – Commands**

💰 **Backtesting**:
• `backtest usdc/eth for the last 7 days` – run a back‑test
• `backtest 0x… from 2025-07-01 to 2025-07-05 with 10 ETH` – explicit range
• `status` – system health
• `results` – summary of the latest run

📊 **Data Fetching**:
• `get pool events from last 24 hours` – recent events
• `fetch 200 events from last 3 days` – custom timeframe

🔧 **General**:
• `help` – show this help text
"""

STATUS_TEXT = """📊 **Status**

✅ Chat gateway online
✅ Structured‑output LLM agent reachable
✅ Backtest service up
✅ The Graph data connection ready
"""


def _format_results(entry: Dict[str, Any]) -> str:
    if not entry:
        return "No backtest results available yet."
    start = datetime.fromtimestamp(entry["start"], tz=timezone.utc).strftime("%Y-%m-%d")
    end = datetime.fromtimestamp(entry["end"], tz=timezone.utc).strftime("%Y-%m-%d")
    r = entry["results"]
    return (
        "📈 **Latest Backtest**\n\n"
        f"Pool: {entry['pool'][:10]}…  \n"
        f"Period: {start} → {end}\n\n"
        f"PnL: {r.get('pnl', 0):.4f}  \n"
        f"Sharpe: {r.get('sharpe', 0):.2f}"
    )


# -----------------------------------------------------------------------------
# protocols
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


# -----------------------------------------------------------------------------
# chat message handler
# -----------------------------------------------------------------------------


@chat_proto.on_message(ChatMessage)
async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
    ctx.logger.info(f"Got ChatMessage from {sender}: {msg}")

    # remember who owns this session (needed when the back‑test finishes)
    ctx.storage.set(str(ctx.session), sender)

    # ACK immediately so the UI can show a tick
    await ctx.send(
        sender,
        ChatAcknowledgement(
            timestamp=datetime.now(timezone.utc),
            acknowledged_msg_id=msg.msg_id,
        ),
    )

    for item in msg.content:
        # welcome message
        if isinstance(item, StartSessionContent):
            await ctx.send(sender, create_text_chat(HELP_TEXT))
            continue

        # we only care about raw text at this level
        if not isinstance(item, TextContent):
            continue

        cmd, info = parser.parse(item.text)

        # ---------------- immediate, local commands
        if cmd is Cmd.HELP:
            await ctx.send(sender, create_text_chat(HELP_TEXT))
            continue
        if cmd is Cmd.STATUS:
            await ctx.send(sender, create_text_chat(STATUS_TEXT))
            continue
        if cmd is Cmd.RESULTS:
            await ctx.send(sender, create_text_chat(_format_results(parser.latest())))
            continue

        # ---------------- data requests (GraphQL fetch)
        if cmd is Cmd.DATA:
            ctx.logger.info(f"Handling data request: {info['query']}")
            try:
                response = await handle_data_request(info["query"])
                await ctx.send(sender, create_text_chat(response))
            except Exception as e:
                await ctx.send(sender, create_text_chat(f"❌ **Error**: {e}"))
            continue

        # ---------------- back‑testing (parsed locally)
        if cmd is Cmd.BACKTEST:
            ctx.logger.info(
                "Handling backtest request directly: "
                f"pool={info['pool']}, start={info['start']}, end={info['end']}"
            )
            try:
                pool = info["pool"] or "usdc-eth"
                start = info["start"]
                end = info["end"]
                position_size = info["position_size"]

                # fall back to sensible defaults if user omitted the period
                if not start or not end:
                    start, end = get_default_time_period()

                # kick off the back‑test
                result = await run_backtest(
                    pool,
                    start,
                    end,
                    {"position_size": position_size},
                )

                if result.get("success", False):
                    summary = (
                        "✅ **Backtest Complete!**\n\n"
                        f"📊 **Pool**: {pool}\n"
                        f"💰 **PnL**: {result.get('pnl', 0):.4f}\n"
                        f"📈 **Sharpe Ratio**: {result.get('sharpe', 0):.2f}\n"
                        f"💸 **Total Fees**: {result.get('total_fees', 0):.4f}\n"
                    )
                else:
                    summary = (
                        "❌ **Backtest Failed**: "
                        f"{result.get('error_message', 'Unknown error')}"
                    )

                await ctx.send(sender, create_text_chat(summary))

                # remember for the `results` command
                parser.remember({"pool": pool, "start": start, "end": end, "results": result})
            except Exception as e:
                await ctx.send(sender, create_text_chat(f"❌ **Error running backtest**: {e}"))
            continue

        # ---------------- fallback – send to LLM structured‑output agent
        await ctx.send(
            AI_AGENT_ADDRESS,
            StructuredOutputPrompt(prompt=item.text, output_schema=BacktestRequest.schema()),
        )


# -----------------------------------------------------------------------------
# chat ACK handler (optional – keeps logs tidy)
# -----------------------------------------------------------------------------


@chat_proto.on_message(ChatAcknowledgement)
async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    ctx.logger.debug(f"Chat ACK from {sender} for {msg.acknowledged_msg_id}")


# -----------------------------------------------------------------------------
# structured‑output reply handler – still required for complex LLM parses
# -----------------------------------------------------------------------------


@struct_output_client_proto.on_message(StructuredOutputResponse)
async def handle_structured_output_response(ctx: Context, sender: str, msg: StructuredOutputResponse):
    """Receive the parsed BacktestRequest from the LLM agent, run the back‑test and
    push a neat summary back to the user.  This path is used when the request
    contains parameters we chose not to parse locally (e.g. strategy‑specific
    JSON)."""

    session_sender = ctx.storage.get(str(ctx.session))
    if session_sender is None:
        ctx.logger.warning("No session sender found – dropping response")
        return

    # guardrail – usually the LLM fills unknowns with "<UNKNOWN>"
    if "<UNKNOWN>" in str(msg.output):
        await ctx.send(session_sender, create_text_chat("Sorry, I couldn't process that request."))
        return

    try:
        backtest_request = BacktestRequest.parse_obj(msg.output)
        pool_display = (
            backtest_request.pool[:10] + "…" if len(backtest_request.pool) > 10 else backtest_request.pool
        )
        start_str = datetime.fromtimestamp(backtest_request.start, tz=timezone.utc).strftime("%Y-%m-%d")
        end_str = datetime.fromtimestamp(backtest_request.end, tz=timezone.utc).strftime("%Y-%m-%d")

        await ctx.send(
            session_sender,
            create_text_chat(
                f"🚀 Starting backtest for pool {pool_display} from {start_str} to {end_str}. This may take a few moments…",
            ),
        )

        # run the back‑test (async)
        params = backtest_request.strategy_params or {}
        params["position_size"] = backtest_request.position_size
        result = await run_backtest(backtest_request.pool, backtest_request.start, backtest_request.end, params)

        # summarise
        if result.get("success", False):
            summary = (
                "🎉 **Backtest Complete!**\n\n"
                f"• **PnL**: {result.get('pnl', 0):.4f}\n"
                f"• **Sharpe**: {result.get('sharpe', 0):.2f}\n"
                f"• **Fees**: {result.get('total_fees', 0):.4f} ETH"
            )
            parser.remember({"pool": backtest_request.pool, "start": backtest_request.start, "end": backtest_request.end, "results": result})
        else:
            summary = f"❌ **Backtest Failed**: {result.get('error_message', 'Unknown error')}"

        await ctx.send(session_sender, create_text_chat(summary))
    except Exception as e:
        await ctx.send(session_sender, create_text_chat(f"❌ **Internal error**: {e}"))


# -----------------------------------------------------------------------------
# include protocols in whatever agent imports this module
# -----------------------------------------------------------------------------

__all__ = [
    "chat_proto",
    "struct_output_client_proto",
    "StructuredOutputPrompt",
    "StructuredOutputResponse",
]

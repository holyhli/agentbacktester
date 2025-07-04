from uagents import Agent, Context, Model
import json
import re
from typing import Dict, Any, List, Optional, Literal
from datetime import datetime, timedelta
from data_storage import BacktestDataManager

# AI Engine imports for proper chat protocol
try:
    from ai_engine.chitchat import ChitChatDialogue
    from ai_engine.messages import DialogueMessage
    from ai_engine.types import UAgentResponse, UAgentResponseType
    AI_ENGINE_AVAILABLE = True
except ImportError:
    AI_ENGINE_AVAILABLE = False
    print("⚠️ AI Engine not available. Install with: pip install uagents-ai-engine")


class ChatMessage(Model):
    message: str
    user_id: str = "user"


class ChatResponse(Model):
    message: str
    agent_name: str = "BacktestChatAgent"


# AI Engine dialogue message
if AI_ENGINE_AVAILABLE:
    class BacktestDialogueMessage(DialogueMessage):
        """Backtest dialogue message for AI Engine integration"""
        pass


# Define dialogue messages for AI Engine
class InitiateBacktestDialogue(Model):
    """I initiate Backtest dialogue request"""
    pass


class AcceptBacktestDialogue(Model):
    """I accept Backtest dialogue request"""
    pass


class ConcludeBacktestDialogue(Model):
    """I conclude Backtest dialogue request"""
    pass


class RejectBacktestDialogue(Model):
    """I reject Backtest dialogue request"""
    pass


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


# Create ChatAgent
chat_agent = Agent(
    name="BacktestChatAgent",
    port=8003,
    seed="chat_agent_secret_phrase",
    endpoint=["http://127.0.0.1:8003/submit"],
    agentverse="https://agentverse.ai",
)

# BacktestAgent address - will be updated dynamically
BACKTEST_AGENT_ADDRESS = "agent1qwlxe6jxh6pyqpwp6ne00zj9fl4w2hnw4646j3ww402fz5cw6zkygcls6tn"

# Data manager for logging
data_manager = BacktestDataManager()

# Store active chat sessions
active_sessions: Dict[str, Dict[str, Any]] = {}


class ChatCommandParser:
    """Parse natural language commands for backtesting"""
    
    def __init__(self):
        # Common pool addresses for easy reference
        self.known_pools = {
            "usdc-eth": "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640",
            "wbtc-eth": "0xcbcdf9626bc03e24f779434178a73a0b4bad62ed",
            "usdc-weth": "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640",
            "wbtc-weth": "0xcbcdf9626bc03e24f779434178a73a0b4bad62ed",
        }
        
        # Time period mappings
        self.time_periods = {
            "1 day": 86400,
            "1 week": 604800,
            "1 month": 2592000,
            "3 months": 7776000,
            "6 months": 15552000,
            "1 year": 31536000,
        }
    
    def parse_command(self, message: str) -> Dict[str, Any]:
        """Parse user message and extract backtest parameters"""
        message = message.lower().strip()
        
        result = {
            "action": "unknown",
            "pool": None,
            "start": None,
            "end": None,
            "error": None
        }
        
        # Check for help commands
        if any(word in message for word in ["help", "commands", "what can you do"]):
            result["action"] = "help"
            return result
        
        # Check for status commands
        if any(word in message for word in ["status", "running", "active"]):
            result["action"] = "status"
            return result
        
        # Check for backtest commands
        if any(word in message for word in ["backtest", "test", "simulate", "run"]):
            result["action"] = "backtest"
            
            # Extract pool
            pool = self._extract_pool(message)
            if pool:
                result["pool"] = pool
            else:
                result["error"] = "Could not identify the pool. Please specify a pool like 'USDC-ETH' or provide a pool address."
                return result
            
            # Extract time period
            start_time, end_time = self._extract_time_period(message)
            if start_time and end_time:
                result["start"] = start_time
                result["end"] = end_time
            else:
                # Default to last 30 days
                end_time = int(datetime.now().timestamp())
                start_time = end_time - (30 * 24 * 60 * 60)  # 30 days ago
                result["start"] = start_time
                result["end"] = end_time
            
            return result
        
        # Check for results commands
        if any(word in message for word in ["results", "show results", "latest", "last backtest"]):
            result["action"] = "show_results"
            return result
        
        result["error"] = "I didn't understand that command. Type 'help' to see available commands."
        return result
    
    def _extract_pool(self, message: str) -> str:
        """Extract pool information from message"""
        # Check for known pool names
        for pool_name, address in self.known_pools.items():
            if pool_name in message:
                return address
        
        # Check for pool address pattern (0x followed by 40 hex characters)
        pool_pattern = r'0x[a-fA-F0-9]{40}'
        match = re.search(pool_pattern, message)
        if match:
            return match.group(0)
        
        return None
    
    def _extract_time_period(self, message: str) -> tuple[int, int]:
        """Extract time period from message"""
        now = datetime.now()
        
        # Check for specific time periods
        for period_name, seconds in self.time_periods.items():
            if period_name in message:
                end_time = int(now.timestamp())
                start_time = end_time - seconds
                return start_time, end_time
        
        # Check for "last X days/weeks/months"
        patterns = [
            (r'last (\d+) days?', lambda x: int(x) * 86400),
            (r'last (\d+) weeks?', lambda x: int(x) * 604800),
            (r'last (\d+) months?', lambda x: int(x) * 2592000),
        ]
        
        for pattern, multiplier in patterns:
            match = re.search(pattern, message)
            if match:
                duration = multiplier(match.group(1))
                end_time = int(now.timestamp())
                start_time = end_time - duration
                return start_time, end_time
        
        return None, None


# Initialize command parser
parser = ChatCommandParser()

# Initialize AI Engine dialogue if available
if AI_ENGINE_AVAILABLE:
    # Create the dialogue
    backtest_dialogue = ChitChatDialogue(
        version="0.1",
        storage=chat_agent.storage,
    )
    
    @backtest_dialogue.on_initiate_session(InitiateBacktestDialogue)
    async def start_backtest_dialogue(ctx: Context, sender: str, _msg: InitiateBacktestDialogue):
        ctx.logger.info(f"Received dialogue init from {sender}")
        await ctx.send(sender, AcceptBacktestDialogue())
    
    @backtest_dialogue.on_start_dialogue(AcceptBacktestDialogue)
    async def accepted_backtest_dialogue(ctx: Context, sender: str, _msg: AcceptBacktestDialogue):
        ctx.logger.info(f"Dialogue with {sender} was accepted")
    
    @backtest_dialogue.on_continue_dialogue(BacktestDialogueMessage)
    async def continue_backtest_dialogue(ctx: Context, sender: str, msg: BacktestDialogueMessage):
        """Handle AI Engine dialogue messages"""
        try:
            user_message = msg.user_message or ""
            ctx.logger.info(f"Received dialogue message: {user_message} from: {sender}")
            
            # Parse the command using our existing parser
            parsed = parser.parse_command(user_message)
            
            # Generate response based on parsed command
            if parsed["action"] == "help":
                response_text = """🤖 **Backtest Chat Agent Help**

I can help you run backtests on Uniswap V4 pools! Here are the commands I understand:

**Backtest Commands:**
• `backtest USDC-ETH for 1 week` - Run backtest on USDC-ETH pool for the last week
• `test WBTC-ETH for 1 month` - Run backtest on WBTC-ETH pool for the last month
• `simulate 0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640 for 3 months` - Use specific pool address

**Supported Pools:**
• USDC-ETH, WBTC-ETH, USDC-WETH, WBTC-WETH
• Or provide any pool address (0x...)

**Time Periods:**
• 1 day, 1 week, 1 month, 3 months, 6 months, 1 year
• Or specify: "last 5 days", "last 2 weeks", etc.

**Other Commands:**
• `status` - Check system status
• `results` - Show latest backtest results
• `help` - Show this help message

**Example:**
"Please backtest the USDC-ETH pool for the last 2 weeks" """
                
            elif parsed["action"] == "status":
                response_text = """📊 **System Status**

✅ Chat Agent: Online
✅ Data Agent: Connected  
✅ Backtest Agent: Connected
✅ Storage System: Active

Ready to process backtest requests!"""
                
            elif parsed["action"] == "show_results":
                # Get latest results from storage
                results = data_manager.get_all_backtest_results()
                if results:
                    latest = results[-1]  # Get most recent
                    response_text = f"""📈 **Latest Backtest Results**

Pool: {latest.get('pool_address', 'Unknown')}
Period: {datetime.fromtimestamp(latest.get('start_time', 0)).strftime('%Y-%m-%d')} to {datetime.fromtimestamp(latest.get('end_time', 0)).strftime('%Y-%m-%d')}

Results:
• PnL: {latest['results'].get('pnl', 0):.4f}
• Sharpe Ratio: {latest['results'].get('sharpe', 0):.2f}
• Total Fees: {latest['results'].get('total_fees', 0):.6f}
• Impermanent Loss: {latest['results'].get('impermanent_loss', 0):.4f}
• Gas Costs: {latest['results'].get('gas_costs', 0):.6f}
• Success: {'✅' if latest['results'].get('success') else '❌'}"""
                else:
                    response_text = "No backtest results found. Run a backtest first!"
                    
            elif parsed["action"] == "backtest":
                if parsed["error"]:
                    response_text = f"❌ Error: {parsed['error']}"
                else:
                    # Store session info
                    session_id = f"{sender}_{int(datetime.now().timestamp())}"
                    active_sessions[session_id] = {
                        "user": sender,
                        "pool": parsed["pool"],
                        "start": parsed["start"],
                        "end": parsed["end"],
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Send backtest request to BacktestAgent
                    if BACKTEST_AGENT_ADDRESS:
                        backtest_request = BacktestRequest(
                            pool=parsed["pool"],
                            start=parsed["start"],
                            end=parsed["end"]
                        )
                        await ctx.send(BACKTEST_AGENT_ADDRESS, backtest_request)
                    
                    # Format response
                    pool_name = "Unknown Pool"
                    for name, addr in parser.known_pools.items():
                        if addr == parsed["pool"]:
                            pool_name = name.upper()
                            break
                    
                    start_date = datetime.fromtimestamp(parsed["start"]).strftime('%Y-%m-%d')
                    end_date = datetime.fromtimestamp(parsed["end"]).strftime('%Y-%m-%d')
                    
                    response_text = f"""🚀 **Backtest Started!**

Pool: {pool_name} ({parsed["pool"][:10]}...)
Period: {start_date} to {end_date}

I'm now fetching historical data and running the backtest simulation. This may take a few moments...

I'll send you the results as soon as they're ready! 📊"""
                    
            else:
                response_text = parsed.get("error", "I didn't understand that. Type 'help' for available commands.")
            
            # Send dialogue response
            await ctx.send(
                sender,
                BacktestDialogueMessage(
                    type="agent_message",
                    agent_message=response_text,
                ),
            )
            
            # Log the interaction
            data_manager.log_agent_activity("ChatAgent", f"Processed dialogue message from {sender}")
            
        except Exception as e:
            ctx.logger.error(f"Error in dialogue handler: {e}")
            await ctx.send(sender, ConcludeBacktestDialogue())
    
    @backtest_dialogue.on_end_session(ConcludeBacktestDialogue)
    async def conclude_backtest_dialogue(ctx: Context, sender: str, _msg: ConcludeBacktestDialogue):
        ctx.logger.info(f"Dialogue concluded with {sender}")
    
    # Include the dialogue in the agent
    chat_agent.include(backtest_dialogue, publish_manifest=True)


# Also add UAgentResponse handler for AI Engine compatibility
if AI_ENGINE_AVAILABLE:
    @chat_agent.on_message(model=UAgentResponse)
    async def handle_uagent_response(ctx: Context, sender: str, msg: UAgentResponse):
        """Handle UAgentResponse messages for AI Engine compatibility"""
        ctx.logger.info(f"Received UAgentResponse from {sender}: {msg.message}")
        
        # Process the message like a regular chat message
        if msg.message:
            parsed = parser.parse_command(msg.message)
            
            # Create appropriate response
            if parsed["action"] == "help":
                response_message = """🤖 **Backtest Chat Agent Help**

I can help you run backtests on Uniswap V4 pools! Here are the commands I understand:

**Backtest Commands:**
• `backtest USDC-ETH for 1 week` - Run backtest on USDC-ETH pool for the last week
• `test WBTC-ETH for 1 month` - Run backtest on WBTC-ETH pool for the last month

**Other Commands:**
• `status` - Check system status
• `results` - Show latest backtest results
• `help` - Show this help message

**Example:**
"Please backtest the USDC-ETH pool for the last 2 weeks" """
                
                response = UAgentResponse(
                    type=UAgentResponseType.FINAL,
                    message=response_message,
                    agent_address=ctx.agent.address
                )
            else:
                # Handle other commands similarly
                response_message = "I can help you with backtesting. Type 'help' for available commands."
                response = UAgentResponse(
                    type=UAgentResponseType.FINAL,
                    message=response_message,
                    agent_address=ctx.agent.address
                )
            
            await ctx.send(sender, response)


@chat_agent.on_message(model=ChatMessage, replies=ChatResponse)
async def handle_chat_message(ctx: Context, sender: str, msg: ChatMessage):
    """Handle incoming chat messages"""
    try:
        ctx.logger.info(f"Received chat message from {sender}: {msg.message}")
        
        # Log the interaction
        data_manager.log_agent_activity("ChatAgent", f"Received message from {sender}: {msg.message}")
        
        # Parse the command
        parsed = parser.parse_command(msg.message)
        
        if parsed["action"] == "help":
            response_text = """
🤖 **Backtest Chat Agent Help**

I can help you run backtests on Uniswap V4 pools! Here are the commands I understand:

**Backtest Commands:**
• `backtest USDC-ETH for 1 week` - Run backtest on USDC-ETH pool for the last week
• `test WBTC-ETH for 1 month` - Run backtest on WBTC-ETH pool for the last month
• `simulate 0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640 for 3 months` - Use specific pool address

**Supported Pools:**
• USDC-ETH, WBTC-ETH, USDC-WETH, WBTC-WETH
• Or provide any pool address (0x...)

**Time Periods:**
• 1 day, 1 week, 1 month, 3 months, 6 months, 1 year
• Or specify: "last 5 days", "last 2 weeks", etc.

**Other Commands:**
• `status` - Check system status
• `results` - Show latest backtest results
• `help` - Show this help message

**Example:**
"Please backtest the USDC-ETH pool for the last 2 weeks"
            """
            
        elif parsed["action"] == "status":
            response_text = """
📊 **System Status**

✅ Chat Agent: Online
✅ Data Agent: Connected
✅ Backtest Agent: Connected
✅ Storage System: Active

Ready to process backtest requests!
            """
            
        elif parsed["action"] == "show_results":
            # Get latest results from storage
            results = data_manager.get_all_backtest_results()
            if results:
                latest = results[-1]  # Get most recent
                response_text = f"""
📈 **Latest Backtest Results**

Pool: {latest.get('pool_address', 'Unknown')}
Period: {datetime.fromtimestamp(latest.get('start_time', 0)).strftime('%Y-%m-%d')} to {datetime.fromtimestamp(latest.get('end_time', 0)).strftime('%Y-%m-%d')}

Results:
• PnL: {latest['results'].get('pnl', 0):.4f}
• Sharpe Ratio: {latest['results'].get('sharpe', 0):.2f}
• Total Fees: {latest['results'].get('total_fees', 0):.6f}
• Impermanent Loss: {latest['results'].get('impermanent_loss', 0):.4f}
• Gas Costs: {latest['results'].get('gas_costs', 0):.6f}
• Success: {'✅' if latest['results'].get('success') else '❌'}
                """
            else:
                response_text = "No backtest results found. Run a backtest first!"
                
        elif parsed["action"] == "backtest":
            if parsed["error"]:
                response_text = f"❌ Error: {parsed['error']}"
            else:
                # Store session info
                session_id = f"{sender}_{int(datetime.now().timestamp())}"
                active_sessions[session_id] = {
                    "user": sender,
                    "pool": parsed["pool"],
                    "start": parsed["start"],
                    "end": parsed["end"],
                    "timestamp": datetime.now().isoformat()
                }
                
                # Send backtest request to BacktestAgent
                backtest_request = BacktestRequest(
                    pool=parsed["pool"],
                    start=parsed["start"],
                    end=parsed["end"]
                )
                
                await ctx.send(BACKTEST_AGENT_ADDRESS, backtest_request)
                
                # Format response
                pool_name = "Unknown Pool"
                for name, addr in parser.known_pools.items():
                    if addr == parsed["pool"]:
                        pool_name = name.upper()
                        break
                
                start_date = datetime.fromtimestamp(parsed["start"]).strftime('%Y-%m-%d')
                end_date = datetime.fromtimestamp(parsed["end"]).strftime('%Y-%m-%d')
                
                response_text = f"""
🚀 **Backtest Started!**

Pool: {pool_name} ({parsed["pool"][:10]}...)
Period: {start_date} to {end_date}

I'm now fetching historical data and running the backtest simulation. This may take a few moments...

I'll send you the results as soon as they're ready! 📊
                """
                
        else:
            response_text = parsed.get("error", "I didn't understand that. Type 'help' for available commands.")
        
        # Send response
        response = ChatResponse(message=response_text.strip())
        await ctx.send(sender, response)
        
        # Log the response
        data_manager.log_agent_activity("ChatAgent", f"Sent response to {sender}")
        
    except Exception as e:
        ctx.logger.error(f"Error handling chat message: {e}")
        error_response = ChatResponse(
            message=f"Sorry, I encountered an error: {str(e)}. Please try again."
        )
        await ctx.send(sender, error_response)


@chat_agent.on_message(model=BacktestResults)
async def handle_backtest_results(ctx: Context, sender: str, msg: BacktestResults):
    """Handle backtest results from BacktestAgent"""
    try:
        ctx.logger.info(f"Received backtest results from {sender}")
        
        # Find the user who requested this backtest
        # In a real implementation, you'd have better session tracking
        target_user = None
        for session_id, session_info in active_sessions.items():
            # Simple matching - in production you'd want more sophisticated tracking
            target_user = session_info["user"]
            break
        
        if not target_user:
            ctx.logger.warning("Could not find target user for backtest results")
            return
        
        # Format results message
        if msg.success:
            response_text = f"""
🎉 **Backtest Complete!**

📊 **Results Summary:**
• **PnL**: {msg.pnl:.4f} ({'+' if msg.pnl >= 0 else ''}{msg.pnl*100:.2f}%)
• **Sharpe Ratio**: {msg.sharpe:.2f}
• **Total Fees**: {msg.total_fees:.6f} ETH
• **Impermanent Loss**: {msg.impermanent_loss:.4f}
• **Gas Costs**: {msg.gas_costs:.6f} ETH

{'🟢 Profitable Strategy!' if msg.pnl > 0 else '🔴 Loss-making Strategy' if msg.pnl < 0 else '⚪ Break-even Strategy'}

Want to run another backtest? Just ask!
            """
        else:
            response_text = f"""
❌ **Backtest Failed**

Error: {msg.error_message}

Please try again with different parameters or check if the pool address is correct.
            """
        
        # Send results to user
        response = ChatResponse(message=response_text.strip())
        await ctx.send(target_user, response)
        
        # Clean up session
        active_sessions.clear()  # Simple cleanup
        
        # Log the interaction
        data_manager.log_agent_activity("ChatAgent", f"Sent backtest results to {target_user}")
        
    except Exception as e:
        ctx.logger.error(f"Error handling backtest results: {e}")


@chat_agent.on_event("startup")
async def startup_handler(ctx: Context):
    """Handle agent startup"""
    ctx.logger.info(f"BacktestChatAgent started with address: {ctx.agent.address}")
    ctx.logger.info("Ready to handle chat messages for backtesting!")
    
    # Log startup
    data_manager.log_agent_activity("ChatAgent", "Agent started and ready for chat")


@chat_agent.on_interval(period=60.0)
async def health_check(ctx: Context):
    """Periodic health check and session cleanup"""
    ctx.logger.info("ChatAgent health check - ready for conversations!")
    
    # Clean up old sessions (older than 1 hour)
    current_time = datetime.now()
    expired_sessions = []
    
    for session_id, session_info in active_sessions.items():
        session_time = datetime.fromisoformat(session_info["timestamp"])
        if (current_time - session_time).total_seconds() > 3600:  # 1 hour
            expired_sessions.append(session_id)
    
    for session_id in expired_sessions:
        del active_sessions[session_id]
    
    if expired_sessions:
        ctx.logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")


if __name__ == "__main__":
    print(f"Chat Agent Address: {chat_agent.address}")
    print("Chat Agent is ready! You can now send messages to interact with the backtesting system.")
    print("\nExample messages:")
    print("- 'help' - Show available commands")
    print("- 'backtest USDC-ETH for 1 week' - Run a backtest")
    print("- 'status' - Check system status")
    chat_agent.run()

#!/usr/bin/env python3
"""
Simple chat client to interact with the BacktestChatAgent
Demonstrates how users can send natural language commands to run backtests
"""
import asyncio
from uagents import Agent, Context, Model
from typing import Optional


class ChatMessage(Model):
    message: str
    user_id: str = "user"


class ChatResponse(Model):
    message: str
    agent_name: str = "BacktestChatAgent"


# Create a simple client agent with mailbox enabled
client_agent = Agent(
    name="ChatClient",
    port=8004,
    seed="chat_client_secret_phrase",
    endpoint=["http://127.0.0.1:8004/submit"],
)

# Chat agent address (you'll need to update this with the actual address)
CHAT_AGENT_ADDRESS = "agent1qtn5xfe4qz8jelr75wmhgtlszrae8zysztf0g8yr3k8j0gwctggqv6x4ax8"

# Store conversation state
conversation_active = False
waiting_for_response = False


@client_agent.on_message(model=ChatResponse)
async def handle_chat_response(ctx: Context, sender: str, msg: ChatResponse):
    """Handle responses from the chat agent"""
    global waiting_for_response
    waiting_for_response = False
    
    print(f"\n🤖 {msg.agent_name}:")
    print(msg.message)
    print("\n" + "="*50)


async def send_message(ctx: Context, message: str):
    """Send a message to the chat agent"""
    global waiting_for_response
    
    chat_msg = ChatMessage(message=message, user_id="demo_user")
    await ctx.send(CHAT_AGENT_ADDRESS, chat_msg)
    waiting_for_response = True
    
    print(f"💬 You: {message}")
    print("⏳ Waiting for response...")


async def interactive_chat():
    """Run interactive chat session"""
    global conversation_active, waiting_for_response
    
    print("🚀 Backtest Chat Client Started!")
    print("="*50)
    print("Welcome to the Uniswap V4 Backtest Chat Interface!")
    print("Type 'help' to see available commands, or 'quit' to exit.")
    print("="*50)
    
    # Build context
    ctx = client_agent._build_context()
    
    conversation_active = True
    
    while conversation_active:
        try:
            # Get user input
            user_input = input("\n💬 You: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                conversation_active = False
                break
            
            # Send message to chat agent
            await send_message(ctx, user_input)
            
            # Wait for response (with timeout)
            timeout_counter = 0
            while waiting_for_response and timeout_counter < 30:  # 30 second timeout
                await asyncio.sleep(1)
                timeout_counter += 1
            
            if waiting_for_response:
                print("⚠️ Response timeout. The chat agent might be busy.")
                waiting_for_response = False
                
        except KeyboardInterrupt:
            print("\n👋 Chat session ended.")
            conversation_active = False
            break
        except Exception as e:
            print(f"❌ Error: {e}")


async def demo_conversation():
    """Run a demo conversation with predefined messages"""
    print("🎬 Running Demo Conversation...")
    print("="*50)
    
    ctx = client_agent._build_context()
    
    demo_messages = [
        "help",
        "status", 
        "backtest USDC-ETH for 1 week",
        "results"
    ]
    
    for message in demo_messages:
        print(f"\n💬 Demo User: {message}")
        await send_message(ctx, message)
        
        # Wait for response
        timeout_counter = 0
        while waiting_for_response and timeout_counter < 15:
            await asyncio.sleep(1)
            timeout_counter += 1
        
        if waiting_for_response:
            print("⚠️ Response timeout")
            waiting_for_response = False
        
        # Wait between messages
        await asyncio.sleep(2)
    
    print("\n🎬 Demo completed!")


@client_agent.on_event("startup")
async def startup_handler(ctx: Context):
    """Handle client startup"""
    ctx.logger.info(f"ChatClient started with address: {ctx.agent.address}")
    print(f"📍 Client Address: {ctx.agent.address}")


async def main():
    """Main function to run the chat client"""
    import sys
    
    # Start the client agent in the background
    client_task = asyncio.create_task(client_agent.run_async())
    
    # Wait a moment for startup
    await asyncio.sleep(2)
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        await demo_conversation()
    else:
        await interactive_chat()
    
    # Cancel the client task
    client_task.cancel()
    try:
        await client_task
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    print("🔧 Note: Make sure to update CHAT_AGENT_ADDRESS with the actual chat agent address!")
    print("You can find it when you run chat_agent.py\n")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

#!/usr/bin/env python3
"""
Helper script to discover agent addresses and update configuration
Run this after starting the agents to get their addresses
"""
import re
import sys
from data_agent import data_agent
from backtest_agent import backtest_agent
from chat_agent import chat_agent


def get_agent_addresses():
    """Get all agent addresses"""
    addresses = {
        "DataAgent": data_agent.address,
        "BacktestAgent": backtest_agent.address,
        "ChatAgent": chat_agent.address,
    }
    return addresses


def update_chat_client_address(chat_agent_address):
    """Update the chat client with the correct chat agent address"""
    try:
        with open("chat_client.py", "r") as f:
            content = f.read()
        
        # Replace the placeholder address
        pattern = r'CHAT_AGENT_ADDRESS = "test-agent://[^"]*"'
        replacement = f'CHAT_AGENT_ADDRESS = "test-agent://{chat_agent_address}"'
        
        updated_content = re.sub(pattern, replacement, content)
        
        with open("chat_client.py", "w") as f:
            f.write(updated_content)
        
        print(f"✅ Updated chat_client.py with ChatAgent address: {chat_agent_address}")
        return True
    except Exception as e:
        print(f"❌ Error updating chat_client.py: {e}")
        return False


def update_backtest_agent_address(data_agent_address):
    """Update the backtest agent with the correct data agent address"""
    try:
        with open("backtest_agent.py", "r") as f:
            content = f.read()
        
        # Replace the placeholder address
        pattern = r'DATA_AGENT_ADDRESS = "test-agent://[^"]*"'
        replacement = f'DATA_AGENT_ADDRESS = "test-agent://{data_agent_address}"'
        
        updated_content = re.sub(pattern, replacement, content)
        
        with open("backtest_agent.py", "w") as f:
            f.write(updated_content)
        
        print(f"✅ Updated backtest_agent.py with DataAgent address: {data_agent_address}")
        return True
    except Exception as e:
        print(f"❌ Error updating backtest_agent.py: {e}")
        return False


def update_chat_agent_address(backtest_agent_address):
    """Update the chat agent with the correct backtest agent address"""
    try:
        with open("chat_agent.py", "r") as f:
            content = f.read()
        
        # Replace the placeholder address
        pattern = r'BACKTEST_AGENT_ADDRESS = None'
        replacement = f'BACKTEST_AGENT_ADDRESS = "test-agent://{backtest_agent_address}"'
        
        updated_content = re.sub(pattern, replacement, content)
        
        with open("chat_agent.py", "w") as f:
            f.write(updated_content)
        
        print(f"✅ Updated chat_agent.py with BacktestAgent address: {backtest_agent_address}")
        return True
    except Exception as e:
        print(f"❌ Error updating chat_agent.py: {e}")
        return False


def main():
    """Main function"""
    print("🔍 Agent Address Discovery Tool")
    print("=" * 50)
    
    # Get all addresses
    addresses = get_agent_addresses()
    
    print("📍 Agent Addresses:")
    for name, address in addresses.items():
        print(f"  {name}: {address}")
    
    print("\n🔧 Configuration Updates:")
    
    # Update configurations
    success_count = 0
    
    # Update backtest agent with data agent address
    if update_backtest_agent_address(addresses["DataAgent"]):
        success_count += 1
    
    # Update chat agent with backtest agent address
    if update_chat_agent_address(addresses["BacktestAgent"]):
        success_count += 1
    
    # Update chat client with chat agent address
    if update_chat_client_address(addresses["ChatAgent"]):
        success_count += 1
    
    print(f"\n✅ Successfully updated {success_count}/3 configuration files")
    
    if success_count == 3:
        print("\n🎉 All configurations updated successfully!")
        print("You can now run the chat client:")
        print("  python3 chat_client.py")
    else:
        print("\n⚠️  Some configurations failed to update. Please check manually.")
    
    print("\n📋 Quick Reference:")
    print("  Start system: python3 main.py")
    print("  Chat client:  python3 chat_client.py")
    print("  Demo mode:    python3 chat_client.py demo")


if __name__ == "__main__":
    main()

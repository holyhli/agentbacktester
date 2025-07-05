#!/usr/bin/env python3
"""
Data storage utilities for uAgents
Demonstrates how agents can read/write local files for persistent data storage
"""
import json
import os
import pickle
from typing import Dict, Any, List
from datetime import datetime
import csv


class AgentDataStorage:
    """Utility class for agent data persistence"""
    
    def __init__(self, storage_dir: str = "agent_data"):
        self.storage_dir = storage_dir
        self.ensure_storage_dir()
    
    def ensure_storage_dir(self):
        """Ensure storage directory exists"""
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)
    
    def save_json(self, filename: str, data: Dict[str, Any]) -> bool:
        """Save data as JSON file"""
        try:
            filepath = os.path.join(self.storage_dir, filename)
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving JSON to {filename}: {e}")
            return False
    
    def load_json(self, filename: str) -> Dict[str, Any]:
        """Load data from JSON file"""
        try:
            filepath = os.path.join(self.storage_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            print(f"Error loading JSON from {filename}: {e}")
            return {}
    
    def save_pickle(self, filename: str, data: Any) -> bool:
        """Save data as pickle file (for complex Python objects)"""
        try:
            filepath = os.path.join(self.storage_dir, filename)
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            return True
        except Exception as e:
            print(f"Error saving pickle to {filename}: {e}")
            return False
    
    def load_pickle(self, filename: str) -> Any:
        """Load data from pickle file"""
        try:
            filepath = os.path.join(self.storage_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    return pickle.load(f)
            return None
        except Exception as e:
            print(f"Error loading pickle from {filename}: {e}")
            return None
    
    def save_csv(self, filename: str, data: List[Dict[str, Any]]) -> bool:
        """Save data as CSV file"""
        try:
            if not data:
                return False
            
            filepath = os.path.join(self.storage_dir, filename)
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
            return True
        except Exception as e:
            print(f"Error saving CSV to {filename}: {e}")
            return False
    
    def load_csv(self, filename: str) -> List[Dict[str, Any]]:
        """Load data from CSV file"""
        try:
            filepath = os.path.join(self.storage_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    reader = csv.DictReader(f)
                    return list(reader)
            return []
        except Exception as e:
            print(f"Error loading CSV from {filename}: {e}")
            return []
    
    def append_to_log(self, filename: str, log_entry: str) -> bool:
        """Append log entry to file"""
        try:
            filepath = os.path.join(self.storage_dir, filename)
            timestamp = datetime.now().isoformat()
            with open(filepath, 'a') as f:
                f.write(f"[{timestamp}] {log_entry}\n")
            return True
        except Exception as e:
            print(f"Error appending to log {filename}: {e}")
            return False
    
    def file_exists(self, filename: str) -> bool:
        """Check if file exists in storage directory"""
        filepath = os.path.join(self.storage_dir, filename)
        return os.path.exists(filepath)
    
    def delete_file(self, filename: str) -> bool:
        """Delete file from storage directory"""
        try:
            filepath = os.path.join(self.storage_dir, filename)
            if os.path.exists(filepath):
                os.remove(filepath)
                return True
            return False
        except Exception as e:
            print(f"Error deleting file {filename}: {e}")
            return False
    
    def list_files(self) -> List[str]:
        """List all files in storage directory"""
        try:
            return [f for f in os.listdir(self.storage_dir) if os.path.isfile(os.path.join(self.storage_dir, f))]
        except Exception as e:
            print(f"Error listing files: {e}")
            return []
    
    def get_file_size(self, filename: str) -> int:
        """Get file size in bytes"""
        try:
            filepath = os.path.join(self.storage_dir, filename)
            if os.path.exists(filepath):
                return os.path.getsize(filepath)
            return 0
        except Exception as e:
            print(f"Error getting file size for {filename}: {e}")
            return 0


class BacktestDataManager:
    """Specialized data manager for backtest results"""
    
    def __init__(self, storage_dir: str = "backtest_data"):
        self.storage = AgentDataStorage(storage_dir)
    
    def save_backtest_result(self, pool_address: str, start_time: int, end_time: int, results: Dict[str, Any]) -> bool:
        """Save backtest results with metadata"""
        result_data = {
            "pool_address": pool_address,
            "start_time": start_time,
            "end_time": end_time,
            "timestamp": datetime.now().isoformat(),
            "results": results
        }
        
        filename = f"backtest_{pool_address}_{start_time}_{end_time}.json"
        return self.storage.save_json(filename, result_data)
    
    def load_backtest_result(self, pool_address: str, start_time: int, end_time: int) -> Dict[str, Any]:
        """Load specific backtest result"""
        filename = f"backtest_{pool_address}_{start_time}_{end_time}.json"
        return self.storage.load_json(filename)
    
    def save_pool_events(self, pool_address: str, events: List[Dict[str, Any]]) -> bool:
        """Save pool events data"""
        filename = f"events_{pool_address}.json"
        return self.storage.save_json(filename, {"events": events})
    
    def load_pool_events(self, pool_address: str) -> List[Dict[str, Any]]:
        """Load pool events data"""
        filename = f"events_{pool_address}.json"
        data = self.storage.load_json(filename)
        return data.get("events", [])
    
    def cache_graph_data(self, pool_address: str, start_time: int, end_time: int, events: List[Dict[str, Any]]) -> bool:
        """Cache The Graph data to avoid repeated API calls"""
        cache_data = {
            "pool_address": pool_address,
            "start_time": start_time,
            "end_time": end_time,
            "cached_at": datetime.now().isoformat(),
            "events": events
        }
        
        filename = f"graph_cache_{pool_address}_{start_time}_{end_time}.json"
        return self.storage.save_json(filename, cache_data)
    
    def get_cached_graph_data(self, pool_address: str, start_time: int, end_time: int) -> List[Dict[str, Any]]:
        """Get cached The Graph data"""
        filename = f"graph_cache_{pool_address}_{start_time}_{end_time}.json"
        data = self.storage.load_json(filename)
        return data.get("events", [])
    
    def log_agent_activity(self, agent_name: str, activity: str) -> bool:
        """Log agent activity"""
        filename = f"{agent_name}_activity.log"
        return self.storage.append_to_log(filename, activity)
    
    def store_backtest_results(self, result_data: Dict[str, Any]) -> bool:
        """Store backtest results (alias for save_backtest_result)"""
        return self.save_backtest_result(
            result_data['pool_address'],
            result_data['start_time'],
            result_data['end_time'],
            result_data['results']
        )
    
    def get_all_backtest_results(self) -> List[Dict[str, Any]]:
        """Get all backtest results"""
        files = self.storage.list_files()
        backtest_files = [f for f in files if f.startswith("backtest_") and f.endswith(".json")]
        
        results = []
        for filename in backtest_files:
            data = self.storage.load_json(filename)
            if data:
                results.append(data)
        
        return results


# Example usage in an agent
if __name__ == "__main__":
    # Initialize storage
    storage = AgentDataStorage("test_agent_data")
    backtest_manager = BacktestDataManager("test_backtest_data")
    
    # Example data operations
    print("=== Testing Data Storage ===")
    
    # Save JSON data
    test_data = {
        "agent_name": "TestAgent",
        "status": "active",
        "last_update": datetime.now().isoformat()
    }
    storage.save_json("agent_status.json", test_data)
    
    # Load JSON data
    loaded_data = storage.load_json("agent_status.json")
    print(f"Loaded data: {loaded_data}")
    
    # Save backtest result
    backtest_result = {
        "pnl": 0.05,
        "sharpe": 1.2,
        "total_fees": 0.001,
        "success": True
    }
    backtest_manager.save_backtest_result(
        "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640",
        1640995200,
        1641081600,
        backtest_result
    )
    
    # Log activity
    backtest_manager.log_agent_activity("TestAgent", "Started backtest simulation")
    
    # List all files
    print(f"Storage files: {storage.list_files()}")
    print(f"Backtest files: {backtest_manager.storage.list_files()}")
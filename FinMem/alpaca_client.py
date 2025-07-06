"""
Alpaca API Client for FinMem
Handles authentication, data retrieval, and order execution.
"""

class AlpacaClient:
    def __init__(self, api_key, api_secret, base_url):
        """Initialize Alpaca API client."""
        # TODO: Set up Alpaca API connection
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url

    def get_account(self):
        """Fetch account information."""
        # TODO: Implement
        return {"account_id": "mock_account"}

    def get_positions(self):
        """Fetch current positions (mocked)."""
        # Return a mock position for testing
        return [{"symbol": "AAPL"}]

    def get_historical_data(self, symbol, timeframe, start, end):
        """Fetch historical market data (mocked)."""
        # Return a mock price series
        return [100, 105, 102, 110, 108, 115]

    def submit_order(self, symbol, qty, side, type, time_in_force):
        """Place an order (mocked)."""
        # TODO: Implement
        return {"status": "submitted", "symbol": symbol, "qty": qty, "side": side} 
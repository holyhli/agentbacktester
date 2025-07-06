"""
FinMem Trading Agent
Main entry point for orchestrating trading, news, and risk modules.
"""
from FinMem.config import Config
from FinMem.alpaca_client import AlpacaClient
from FinMem.news_graph import NewsGraphClient
from FinMem.risk_profile import RiskProfiler
from FinMem.utils import get_logger

class FinMemAgent:
    def __init__(self, config):
        """Initialize agent with config."""
        self.config = config
        self.logger = get_logger("FinMemAgent")
        self.alpaca = AlpacaClient(
            config.ALPACA_API_KEY,
            config.ALPACA_API_SECRET,
            config.ALPACA_BASE_URL
        )
        self.news_graph = NewsGraphClient(
            endpoint=config.NEWS_GRAPH_ENDPOINT,
            mock=config.USE_MOCK_NEWS_GRAPH
        )
        self.risk_profiler = RiskProfiler()

    def run(self):
        """Main loop for the trading agent (single iteration for demo)."""
        self.logger.info("Starting trading loop...")
        # 1. Fetch positions and market data
        positions = self.alpaca.get_positions()
        if not positions:
            self.logger.info("No positions found. Exiting.")
            return
        for pos in positions:
            symbol = pos.get('symbol')
            self.logger.info(f"Processing {symbol}")
            # 2. Get historical prices
            prices = self.alpaca.get_historical_data(symbol, '1D', '2024-01-01', '2024-06-01')
            # 3. Query news graph for sentiment/news
            news = self.news_graph.get_news_for_ticker(symbol)
            sentiments = [n['sentiment'] for n in news]
            # 4. Compute risk profile
            risk = self.risk_profiler.calculate(prices, sentiments)
            self.logger.info(f"Risk profile for {symbol}: {risk}")
            # 5. Make trade decision (stub)
            action = 'hold'  # TODO: Implement real decision logic
            self.logger.info(f"Action for {symbol}: {action}")
            # 6. Log actions and risk profiles (already logged)

if __name__ == "__main__":
    agent = FinMemAgent(Config)
    agent.run() 
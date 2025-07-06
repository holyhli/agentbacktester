"""
Configuration loader for FinMem
Loads API keys and settings from environment variables or .env file.
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    ALPACA_API_KEY = os.getenv('ALPACA_API_KEY', '')
    ALPACA_API_SECRET = os.getenv('ALPACA_API_SECRET', '')
    ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    NEWS_GRAPH_ENDPOINT = os.getenv('NEWS_GRAPH_ENDPOINT', None)
    USE_MOCK_NEWS_GRAPH = os.getenv('USE_MOCK_NEWS_GRAPH', 'True') == 'True' 
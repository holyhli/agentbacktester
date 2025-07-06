# FinMem Trading Agent

A modular trading agent for Alpaca, news graph sentiment, and risk profiling.

## Setup

1. Clone the repo and create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. Set up your environment variables (or create a `.env` file):
   ```env
   ALPACA_API_KEY=your_key
   ALPACA_API_SECRET=your_secret
   ALPACA_BASE_URL=https://paper-api.alpaca.markets
   NEWS_GRAPH_ENDPOINT=your_news_graph_endpoint
   USE_MOCK_NEWS_GRAPH=True
   ```

## Usage

**Run the agent from the project root:**
```bash
python -m FinMem.agent
```
This ensures all package imports work correctly.

## Testing

Run all tests:
```bash
pytest
```

## Structure
- `agent.py`: Main agent logic
- `alpaca_client.py`: Alpaca API integration
- `news_graph.py`: News/sentiment graph integration
- `risk_profile.py`: Risk metrics
- `config.py`: Configuration loader
- `utils.py`: Logging and helpers
- `tests/`: Unit tests 
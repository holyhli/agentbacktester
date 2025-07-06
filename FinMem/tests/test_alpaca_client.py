import pytest
from FinMem.alpaca_client import AlpacaClient

def test_alpaca_client_init():
    client = AlpacaClient('key', 'secret', 'https://paper-api.alpaca.markets')
    assert client is not None 
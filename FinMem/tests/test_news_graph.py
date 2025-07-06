import pytest
from FinMem.news_graph import NewsGraphClient

def test_news_graph_client_mock():
    client = NewsGraphClient(mock=True)
    news = client.get_news_for_ticker('AAPL')
    assert isinstance(news, list)
    assert 'headline' in news[0] 
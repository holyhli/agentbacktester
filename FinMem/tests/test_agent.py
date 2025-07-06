import pytest
from FinMem.agent import FinMemAgent
from FinMem.config import Config

def test_agent_init():
    agent = FinMemAgent(Config)
    assert agent is not None 
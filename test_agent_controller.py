import pandas as pd
from unittest.mock import MagicMock
from agent_controller import AgentController
from evaluator import Signal


def make_df():
    return pd.DataFrame({
        "Close": [100.0, 110.0],
        "SMA20": [None, 105.0],
        "SMA50": [None, 100.0],
    })


def test_run_returns_error_when_fetch_fails():
    tools = MagicMock()
    tools.fetch_stock_data.return_value = (None, "API down")
    agent = AgentController(llm_client=MagicMock(), tools=tools, evaluator=MagicMock())

    result = agent.run(symbol="AAPL", question="?")
    assert result["status"] == "error"
    assert result["error"] == "API down"


def test_run_happy_path_composes_pipeline():
    tools = MagicMock()
    tools.fetch_stock_data.return_value = (make_df(), None)

    evaluator = MagicMock()
    evaluator.evaluate.return_value = Signal(action="BUY", confidence=0.8, reasons=["r1"])

    llm = MagicMock()
    llm.generate.return_value = "Narrative text."

    agent = AgentController(llm_client=llm, tools=tools, evaluator=evaluator)
    result = agent.run(symbol="AAPL", question="trend?")

    assert result["status"] == "success"
    assert result["signal"].action == "BUY"
    assert result["narrative"] == "Narrative text."
    assert result["latest_price"] == 110.0
    # LLM must receive the pre-computed signal, not make the decision
    _, kwargs = llm.generate.call_args
    assert kwargs["signal"].action == "BUY"


def test_run_does_not_call_llm_when_fetch_fails():
    tools = MagicMock()
    tools.fetch_stock_data.return_value = (None, "err")
    llm = MagicMock()
    agent = AgentController(llm_client=llm, tools=tools, evaluator=MagicMock())

    agent.run(symbol="AAPL", question="?")
    llm.generate.assert_not_called()

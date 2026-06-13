from typing import Any
from finance_tools import FinanceDataTool
from evaluator import Evaluator
from llm_client import LLMClient


class AgentController:
    """Orchestrates: fetch market data → evaluate signal → generate narrative."""

    def __init__(self, llm_client: LLMClient, tools: FinanceDataTool, evaluator: Evaluator):
        self.llm_client = llm_client
        self.tools = tools
        self.evaluator = evaluator

    def run(self, symbol: str, question: str) -> dict[str, Any]:
        df, err = self.tools.fetch_stock_data(symbol)
        if df is None:
            return {"status": "error", "error": err}

        signal = self.evaluator.evaluate(df)

        latest = df.dropna(subset=["SMA20", "SMA50"]).iloc[-1]
        narrative = self.llm_client.generate(
            symbol=symbol,
            question=question,
            price=latest["Close"],
            sma20=latest["SMA20"],
            sma50=latest["SMA50"],
            signal=signal,
        )

        return {
            "status": "success",
            "signal": signal,
            "narrative": narrative,
            "df": df,
            "latest_price": latest["Close"],
            "sma20": latest["SMA20"],
            "sma50": latest["SMA50"],
        }

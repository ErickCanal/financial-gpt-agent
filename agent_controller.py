"""Orchestration layer: composes the data, decision, and narrative layers.

`AgentController.run()` runs the pipeline in order — fetch market data, evaluate the signal,
then generate narrative — and short-circuits to an error result if the data fetch fails
(so no LLM call is wasted on missing data). Returns a single dict the UI can render directly.
"""
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

        try:
            signal = self.evaluator.evaluate(df)
        except ValueError as e:
            return {"status": "error", "error": str(e)}

        latest = df.dropna(subset=["SMA20", "SMA50"]).iloc[-1]

        # The narrative is optional context. If every LLM provider fails, we still
        # return the deterministic signal — the decision never depended on the LLM.
        try:
            narrative = self.llm_client.generate(
                symbol=symbol,
                question=question,
                price=latest["Close"],
                sma20=latest["SMA20"],
                sma50=latest["SMA50"],
                signal=signal,
            )
        except Exception as e:
            narrative = (
                f"_Analyst narrative unavailable: {e}_\n\n"
                "The signal above is computed from technical rules and does not depend "
                "on the AI narrative."
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

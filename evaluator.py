from dataclasses import dataclass, field
import pandas as pd


@dataclass
class Signal:
    action: str          # "BUY" | "SELL" | "HOLD"
    confidence: float    # 0.0 – 1.0
    reasons: list = field(default_factory=list)


class Evaluator:
    """Deterministic technical-signal evaluator. No LLM involved."""

    def evaluate(self, df: pd.DataFrame) -> Signal:
        latest = df.dropna(subset=["SMA20", "SMA50"]).iloc[-1]
        price = latest["Close"]
        sma20 = latest["SMA20"]
        sma50 = latest["SMA50"]

        bullish = 0
        bearish = 0
        reasons = []

        if price > sma20:
            bullish += 1
            reasons.append(f"Price ${price:.2f} is above SMA20 ${sma20:.2f} (short-term bullish)")
        else:
            bearish += 1
            reasons.append(f"Price ${price:.2f} is below SMA20 ${sma20:.2f} (short-term bearish)")

        if sma20 > sma50:
            bullish += 1
            reasons.append(f"SMA20 ${sma20:.2f} is above SMA50 ${sma50:.2f} (golden-cross trend)")
        else:
            bearish += 1
            reasons.append(f"SMA20 ${sma20:.2f} is below SMA50 ${sma50:.2f} (death-cross trend)")

        # Both indicators agree → high-confidence signal; mixed → HOLD
        if bullish == 2:
            return Signal(action="BUY", confidence=0.80, reasons=reasons)
        elif bearish == 2:
            return Signal(action="SELL", confidence=0.80, reasons=reasons)
        else:
            return Signal(action="HOLD", confidence=0.50, reasons=reasons)

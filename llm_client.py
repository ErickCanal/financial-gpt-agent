"""Narrative layer: turns a pre-computed Signal into readable analyst context.

Crucially, this layer does NOT make the trading decision — it receives a Signal that the
Evaluator already produced and only explains it in prose. Supports four LLM providers
(OpenAI, Claude, Deepseek, Mistral) tried in order as a fallback chain, so a quota error or
outage on one provider transparently falls through to the next.
"""
import os
from evaluator import Signal


class LLMClient:
    """Multi-provider LLM client with automatic fallback. Generates narrative context around a signal."""

    def __init__(self, openai_key: str | None = None, anthropic_key: str | None = None,
                 deepseek_key: str | None = None, mistral_key: str | None = None):
        self.providers = []

        openai_key = openai_key or os.getenv("OPENAI_API_KEY")
        anthropic_key = anthropic_key or os.getenv("ANTHROPIC_API_KEY")
        deepseek_key = deepseek_key or os.getenv("DEEPSEEK_API_KEY")
        mistral_key = mistral_key or os.getenv("MISTRAL_API_KEY")

        if openai_key:
            try:
                from langchain_openai import ChatOpenAI
                self.providers.append(("OpenAI", ChatOpenAI(temperature=0, model="gpt-4", openai_api_key=openai_key)))
            except Exception as e:
                print(f"Warning: OpenAI provider failed to initialize: {e}")

        if anthropic_key:
            try:
                from langchain_anthropic import ChatAnthropic
                self.providers.append(("Claude", ChatAnthropic(temperature=0, model="claude-3-5-sonnet-20241022", api_key=anthropic_key)))
            except Exception as e:
                print(f"Warning: Claude provider failed to initialize: {e}")

        if deepseek_key:
            try:
                from langchain_openai import ChatOpenAI
                # Deepseek uses OpenAI-compatible API
                self.providers.append(("Deepseek", ChatOpenAI(
                    temperature=0,
                    model="deepseek-chat",
                    openai_api_key=deepseek_key,
                    base_url="https://api.deepseek.com",
                )))
            except Exception as e:
                print(f"Warning: Deepseek provider failed to initialize: {e}")

        if mistral_key:
            try:
                from langchain_mistralai import ChatMistralAI
                self.providers.append(("Mistral", ChatMistralAI(temperature=0, model="mistral-large-latest", api_key=mistral_key)))
            except Exception as e:
                print(f"Warning: Mistral provider failed to initialize: {e}")

        if not self.providers:
            raise ValueError("No LLM provider keys configured. Set at least one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, DEEPSEEK_API_KEY, MISTRAL_API_KEY")

    def generate(self, symbol: str, question: str, price: float,
                 sma20: float, sma50: float, signal: Signal) -> str:
        prompt = (
            f"You are a financial analyst providing context for a technical trading signal.\n\n"
            f"Stock: {symbol}\n"
            f"Current Price: ${price:.2f}\n"
            f"20-day SMA: ${sma20:.2f}\n"
            f"50-day SMA: ${sma50:.2f}\n"
            f"Technical Signal: {signal.action} (confidence: {signal.confidence:.0%})\n"
            f"Signal Basis: {'; '.join(signal.reasons)}\n\n"
            f"User Question: {question}\n\n"
            f"Provide a concise narrative (3-5 sentences) explaining what these technical indicators "
            f"suggest about the stock's current trend, any relevant market context, and caveats the "
            f"investor should keep in mind. Do NOT make the buy/sell/hold recommendation — that has "
            f"already been determined by the technical model above."
        )

        for provider_name, llm in self.providers:
            try:
                response = llm.invoke(prompt)
                return response.content
            except Exception as e:
                print(f"Warning: {provider_name} failed, trying next provider: {e}")
                continue

        raise RuntimeError("All LLM providers failed. Check your API keys and quotas.")

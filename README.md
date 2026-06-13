# Financial GPT Agent

A stock analysis tool that gives you a **deterministic** BUY / SELL / HOLD signal from
technical indicators — and uses an LLM only to *explain* the signal, never to make the call.

## Why this design

The original version (`financial_gpt_alpha_app.py`) handed price data to GPT-4 and asked
"should I invest?" — letting a non-deterministic language model own the trading decision.
That's untestable and unpredictable: the same inputs could yield different recommendations.

This refactor **rips the decision out of the LLM** into a pure, testable `evaluate()` function.
The LLM is demoted to what it's actually good at: writing readable narrative context around a
decision that's already been made by transparent rules.

```
main_app.py (Streamlit UI)
   │
   └─ AgentController.run(symbol, question)
         ├─ FinanceDataTool.fetch_stock_data()  → DataFrame + SMA20/SMA50
         ├─ Evaluator.evaluate(df)              → Signal(action, confidence, reasons)   ← deterministic
         └─ LLMClient.generate(..., signal)     → narrative text                        ← explanation only
```

## The decision rules

The signal is computed from two simple moving-average (SMA) comparisons:

| Price vs SMA20 | SMA20 vs SMA50 | Signal | Confidence |
|----------------|----------------|--------|------------|
| Above (bullish) | Above (golden cross) | **BUY** | 80% |
| Below (bearish) | Below (death cross) | **SELL** | 80% |
| Mixed signals | | **HOLD** | 50% |

Every signal includes human-readable `reasons` explaining exactly why it fired.

## Project layout

| File | Layer | Responsibility |
|------|-------|----------------|
| `finance_tools.py` | Data | Fetch Alpha Vantage data, compute SMA20/SMA50 |
| `evaluator.py` | Decision | Deterministic BUY/SELL/HOLD logic (no LLM) |
| `llm_client.py` | Narrative | Multi-provider LLM, generates context only |
| `agent_controller.py` | Orchestration | Wires the pipeline together |
| `main_app.py` | UI | Streamlit front end (the entry point) |
| `financial_gpt_alpha_app.py` | — | Original monolith, kept for comparison |

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure secrets

Create `.streamlit/secrets.toml` (already gitignored — never commit it):

```toml
APP_PASSWORD = "your_password"
ALPHA_VANTAGE_KEY = "your_alpha_vantage_key"

# At least one LLM provider is required. They're tried in this order
# as a fallback chain — if one hits a quota/outage, the next is used.
OPENAI_API_KEY    = "sk-..."
ANTHROPIC_API_KEY = "sk-ant-..."
DEEPSEEK_API_KEY  = "sk-..."
MISTRAL_API_KEY   = "..."
```

| Key | Where to get it |
|-----|-----------------|
| Alpha Vantage | https://www.alphavantage.co/support/#api-key (free) |
| OpenAI | https://platform.openai.com/account/api-keys |
| Anthropic (Claude) | https://console.anthropic.com/account/keys |
| Deepseek | https://platform.deepseek.com/api_keys |
| Mistral | https://console.mistral.ai/api-keys/ |

## Run

```bash
streamlit run main_app.py
```

Enter your password, type a stock symbol (e.g. `AAPL`), ask a question, and click **Analyze**.

## Test

```bash
pip install -r requirements-dev.txt
python -m pytest -v
```

15 tests cover the decision logic, data-layer error handling, and orchestration. All external
calls (HTTP, LLM) are mocked, so the suite runs offline in ~1 second and consumes no API quota.

CI runs the same suite on every push/PR across Python 3.10–3.12 (see `.github/workflows/tests.yml`).

## Multi-provider fallback

`LLMClient` accepts keys for up to four providers and tries them in order
(OpenAI → Claude → Deepseek → Mistral). If a provider fails — quota exceeded, outage, bad key —
it transparently falls back to the next one. You only need one key for the app to work; add more
for resilience.

## Disclaimer

This is an educational/analytical tool. The signals are based on simple moving-average crossovers
and are **not** financial advice. Do your own research before making investment decisions.

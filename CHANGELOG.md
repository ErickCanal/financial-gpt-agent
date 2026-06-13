# Changelog

All notable changes to this project are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- **Layered architecture** replacing the original monolith. The buy/sell decision was
  extracted from the LLM into a deterministic `Evaluator.evaluate()` function:
  - `finance_tools.py` — Alpha Vantage data fetch + SMA20/SMA50 computation
  - `evaluator.py` — deterministic BUY/SELL/HOLD signal from SMA crossover rules
  - `llm_client.py` — LLM client that generates *narrative context only*
  - `agent_controller.py` — orchestrator wiring fetch → evaluate → narrate
  - `main_app.py` — new Streamlit entry point with signal badge, chart, and narrative
- **Multi-provider LLM fallback** in `llm_client.py`. Supports OpenAI, Claude (Anthropic),
  Deepseek, and Mistral, tried in order so a quota error on one provider no longer breaks
  the app.
- **Test suite** (15 tests) covering the decision logic, data-layer error handling, and
  orchestration. All external calls are mocked; runs offline with no API keys.
  - Includes a regression guard asserting the LLM receives the pre-computed signal and
    never makes the decision itself.
- **CI**: GitHub Actions workflow running pytest on push/PR across Python 3.10–3.12.
- `requirements.txt` / `requirements-dev.txt` dependency manifests.
- `.gitignore` excluding secrets and Python caches.
- `README.md` documenting architecture, setup, decision rules, and usage.

### Changed
- LLM role redefined: previously owned the buy/sell recommendation; now only explains a
  signal that deterministic rules have already produced.

### Notes
- The original `financial_gpt_alpha_app.py` is retained unchanged for comparison.
- Known issue: the OpenAI key on the maintainer's account has hit a quota limit
  (`insufficient_quota`). The multi-provider fallback exists partly to work around this —
  add an Anthropic, Deepseek, or Mistral key to keep the app functional.

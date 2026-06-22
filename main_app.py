import streamlit as st

st.set_page_config(page_title="Financial GPT Agent", layout="centered")
st.title("Financial GPT Agent")

from dotenv import load_dotenv
load_dotenv()

# --- Auth ---
password = st.text_input("Enter access password", type="password")
if password != st.secrets.get("APP_PASSWORD"):
    if password:  # only complain once the user has actually typed something
        st.warning("Access denied. Incorrect password.")
    st.stop()

st.success("Access granted.")

# --- Lazy imports after auth so failed auth exits fast ---
from agent_controller import AgentController
from finance_tools import FinanceDataTool
from evaluator import Evaluator
from llm_client import LLMClient

@st.cache_resource
def build_agent():
    return AgentController(
        llm_client=LLMClient(
            openai_key=st.secrets.get("OPENAI_API_KEY"),
            anthropic_key=st.secrets.get("ANTHROPIC_API_KEY"),
            deepseek_key=st.secrets.get("DEEPSEEK_API_KEY"),
            mistral_key=st.secrets.get("MISTRAL_API_KEY"),
        ),
        tools=FinanceDataTool(api_key=st.secrets.get("ALPHA_VANTAGE_KEY")),
        evaluator=Evaluator(),
    )

agent = build_agent()

# --- Inputs ---
symbol = st.text_input("Stock symbol", value="AAPL").upper().strip()
question = st.text_area("Your question", value="What does the current trend suggest?")

if st.button("Analyze"):
    with st.spinner(f"Fetching data and evaluating {symbol}..."):
        result = agent.run(symbol=symbol, question=question)

    if result["status"] == "error":
        st.error(f"Could not analyze {symbol}: {result['error']}")
        st.stop()

    signal = result["signal"]
    df = result["df"]

    # --- Signal badge ---
    color = {"BUY": "green", "SELL": "red", "HOLD": "orange"}.get(signal.action, "gray")
    st.markdown(
        f"<h2 style='color:{color}; text-align:center;'>"
        f"Signal: {signal.action} &nbsp; ({signal.confidence:.0%} confidence)"
        f"</h2>",
        unsafe_allow_html=True,
    )

    # Signal reasons
    st.markdown("**Why:**")
    for reason in signal.reasons:
        st.markdown(f"- {reason}")

    st.divider()

    # --- Price chart ---
    st.subheader(f"{symbol} — Close price with SMAs")
    st.line_chart(df[["Close", "SMA20", "SMA50"]].dropna())

    st.caption(
        f"Latest close: ${result['latest_price']:.2f} &nbsp;|&nbsp; "
        f"SMA20: ${result['sma20']:.2f} &nbsp;|&nbsp; "
        f"SMA50: ${result['sma50']:.2f}"
    )

    st.divider()

    # --- LLM narrative (context only, not the decision) ---
    st.subheader("Analyst Context")
    st.markdown(result["narrative"])
    st.caption("Note: the BUY/SELL/HOLD signal above is computed from technical rules, not from the AI narrative.")

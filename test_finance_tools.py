from unittest.mock import patch, MagicMock
from finance_tools import FinanceDataTool


SAMPLE_SERIES = {
    "Time Series (Daily)": {
        f"2024-01-{day:02d}": {
            "1. open": "100.0", "2. high": "101.0", "3. low": "99.0",
            "4. close": str(100 + day), "5. volume": "1000000",
        }
        for day in range(1, 26)
    }
}


def test_missing_api_key_returns_error():
    df, err = FinanceDataTool(api_key=None).fetch_stock_data("AAPL")
    assert df is None
    assert "API key is required" in err


@patch("finance_tools.requests.get")
def test_successful_fetch_computes_smas(mock_get):
    mock_get.return_value = MagicMock(json=lambda: SAMPLE_SERIES)
    df, err = FinanceDataTool(api_key="fake").fetch_stock_data("AAPL")
    assert err is None
    assert "SMA20" in df.columns
    assert "SMA50" in df.columns
    assert "Close" in df.columns
    # 25 rows → SMA20 has values, SMA50 (needs 50) stays NaN
    assert df["SMA20"].notna().any()


@patch("finance_tools.requests.get")
def test_api_error_message_propagated(mock_get):
    mock_get.return_value = MagicMock(json=lambda: {"Error Message": "Invalid symbol"})
    df, err = FinanceDataTool(api_key="fake").fetch_stock_data("BADSYM")
    assert df is None
    assert "Invalid symbol" in err


@patch("finance_tools.requests.get")
def test_rate_limit_note_propagated(mock_get):
    mock_get.return_value = MagicMock(json=lambda: {"Note": "rate limit hit"})
    df, err = FinanceDataTool(api_key="fake").fetch_stock_data("AAPL")
    assert df is None
    assert "rate limit" in err.lower()


@patch("finance_tools.requests.get", side_effect=Exception("connection refused"))
def test_network_error_handled(mock_get):
    df, err = FinanceDataTool(api_key="fake").fetch_stock_data("AAPL")
    assert df is None
    assert "Network error" in err

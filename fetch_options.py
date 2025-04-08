import yfinance as yf
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
import logging
import re
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
# ✅ Suppress yfinance's built-in messages
logging.getLogger("yfinance").setLevel(logging.ERROR)

# # ✅ Load trained profitability model and scaler
# profit_model = joblib.load("best_model_v5_war.pkl")  # Load the latest model
scaler = StandardScaler()

tickers = ['LHX','GE','RTX','ESLT','MRCY','DRS','NOC','LMT','ERJ','HWM','HEI','VOOG','CDRE','WWD','AXON','CW','TDG','GD']

expected_features = [
    "delta", "rho", "stock_price", "strike", "time_to_expiration",
    "gamma", "theta", "vega"
]

PROFIT_THRESHOLD = 0.4  # Optimized from previous training results

# ✅ Configure logging
logging.basicConfig(filename="trading_bot_war.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

import numpy as np
from scipy.stats import norm

def compute_greeks(S, K, T, r, sigma=0.2):
    if T <= 0:
        return 0, 0, 0, 0, 0  # Avoid division by zero errors for expired options

    # Compute d1 and d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Calculate Greeks
    delta = norm.cdf(d1)  # Call Delta
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    vega = S * np.sqrt(T) * norm.pdf(d1)  # Vega measures sensitivity to IV changes

    return delta, gamma, theta, rho, vega


def get_risk_free_rate():
    try:
        t_bill = yf.Ticker("^TNX")
        t_bill_data = t_bill.history(period="1d")
        if not t_bill_data.empty:
            return t_bill_data["Close"].iloc[-1] / 100
    except Exception as e:
        print(f"⚠️ Error fetching risk-free rate: {e}. Using fallback value.")
    return 0.03

def get_stock_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility_y'] = df['Daily_Return'].rolling(window=10).std()
    return df[['Close', 'Volatility_y']].dropna()

def get_options_data(ticker, risk_free_rate):
    stock = yf.Ticker(ticker)
    today = datetime.today()
    min_expiration_date = today
    expirations = [exp for exp in stock.options if datetime.strptime(exp, "%Y-%m-%d") > min_expiration_date]
    options_data = []
    stock_price = stock.history(period="1d")["Close"].iloc[-1]
    start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
    end_date = datetime.today().strftime('%Y-%m-%d')
    stock_data = get_stock_data(ticker, start_date, end_date)

    for exp in tqdm(expirations, desc=f"Fetching {ticker} options", unit="exp"):
        opt_chain = stock.option_chain(exp)
        calls = opt_chain.calls[['contractSymbol', 'strike', 'lastPrice', 'openInterest', 'volume', 'bid', 'ask']].copy()
        puts = opt_chain.puts[['contractSymbol', 'strike', 'lastPrice', 'openInterest', 'volume', 'bid', 'ask']].copy()
        calls['expiration'], calls['type'] = exp, 'call'
        puts['expiration'], puts['type'] = exp, 'put'
        options_data.extend([calls, puts])

    options_df = pd.concat(options_data, ignore_index=True)
    options_df['time_to_expiration'] = (pd.to_datetime(options_df['expiration']) - datetime.today()).dt.days / 365
    options_df.rename(columns={"contractSymbol": "symbol"}, inplace=True)
    options_df['Volatility_y'] = stock_data['Volatility_y'].mean()
    options_df['quote_date'] = pd.Timestamp.today().normalize()
    options_df = merge_historical_volatility(options_df, ticker)

    
    options_df[['delta', 'gamma', 'theta', 'rho', 'vega']] = options_df.apply(
        lambda row: compute_greeks(stock_price, row["strike"], row["time_to_expiration"], risk_free_rate, row['Volatility_y']), 
        axis=1, result_type="expand"
    )

    #filters:

    options_df = options_df[options_df['volume'] >= 0]
    #options_df = options_df[(options_df["delta"] > 0.05) & (options_df["delta"] < 0.95)]
    # percent_range = 0.5  # 50%
    # options_df = options_df[
    #     (options_df['strike'] >= stock_price * (1 - percent_range)) &
    #     (options_df['strike'] <= stock_price * (1 + percent_range))
    # ]


    
    return options_df

def predict_profitability(options_df):
    """ Predicts profitability for each option contract using the updated model. """

    prediction_df = options_df[['delta', 'rho', 'stock_price', 'strike', 
                                'time_to_expiration', 'gamma', 'theta', 'vega']].copy()

    # Ensure numerical columns are properly formatted
    for col in expected_features:
        prediction_df[col] = pd.to_numeric(options_df[col], errors="coerce")

    # Prepare data for model prediction
    X = prediction_df[expected_features]
    X_scaled = scaler.fit_transform(X)

    
    # Predict profit potential (no probability score needed)
    profit_predictions = profit_model.predict(X_scaled)

    # Store predictions in the DataFrame
    options_df["profit_prediction"] = profit_predictions

    print(options_df["profit_prediction"].describe())

    # Apply a filtering threshold based on the trained model's optimized settings
    options_df = options_df[options_df["profit_prediction"] >= PROFIT_THRESHOLD]
    
    # Sort the profitable options by predicted profit in descending order
    return options_df.sort_values(by="profit_prediction", ascending=False)

def print_top_option(profitable_options):
    if profitable_options.empty:
        print("⚠️ No high-confidence profitable options found.")
        return
    
    print("profitable options before filtering: ", len(profitable_options))

    # Example: Running the filter with real-time stock prices
    filtered_options= get_filtered_sorted_options(profitable_options)

    if not filtered_options.empty:
        top_option = filtered_options.iloc[0]
        print(f"✅ Recommended Option: {top_option['symbol']}")
        print(f"   - Expiration Date: {top_option['expiration']}")
        print(f"   - Strike Price: {top_option['strike']}")
        print(f"   - Price: ${top_option['lastPrice']*100:.2f}")
        print(f"   - Type: {top_option['type']}")
    else:
        print("⚠️ No suitable options found within expiration and liquidity criteria.")



def get_current_stock_prices(tickers):
    """Fetches the latest stock prices for multiple tickers using yfinance."""
    stock_prices = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        try:
            stock_prices[ticker] = stock.history(period="1d")["Close"].iloc[-1]
        except Exception as e:
            print(f"⚠️ Warning: Could not fetch price for {ticker} - {e}")
            stock_prices[ticker] = None  # Handle missing data
    return stock_prices

def get_single_stock_price(ticker):
    """Fetches the latest stock prices for multiple tickers using yfinance."""
    stock = yf.Ticker(ticker)
    try:
        stock_price = stock.history(period="1d")["Close"].iloc[-1]
    except Exception as e:
        print(f"⚠️ Warning: Could not fetch price for {ticker} - {e}")
        stock_price = None  # Handle missing data
    return stock_price


def get_filtered_sorted_options(options_df, min_profit=PROFIT_THRESHOLD, max_price=475, momentum_threshold=0.03):
    """ Filters, ranks, and sorts option contracts based on updated profitability metrics. """
    
    today = datetime.today()

    # Convert expiration dates to datetime
    options_df['expiration'] = pd.to_datetime(options_df['expiration'], errors='coerce')

    # Define an expiration window (between 2 weeks & 3 months from today)
    # min_expiration_date = today + timedelta(days=14)
    # max_expiration_date = today + timedelta(days=90)

    # # Filter based on expiration window & liquidity requirements
    # filtered_df = options_df[
    #     (options_df['expiration'] >= min_expiration_date) & 
    #     (options_df['expiration'] <= max_expiration_date) & 
    #     (options_df['volume'] > 100)  # Ensures sufficient liquidity
    # ]

    # Apply profit threshold for filtering
    filtered_df = options_df[options_df["profit_prediction"] >= min_profit]
    print(f"{len(filtered_df)} options after profit threshold.")


    # Filter out options with very high costs (e.g., > nax_price per contract)
    price_filtered_df = filtered_df[filtered_df["lastPrice"] * 100 <= max_price]
    print(f"{len(price_filtered_df)} options after price threshold.")

    if (price_filtered_df.size == 0):
        sorted_filtered_options = filtered_df.sort_values(by=["lastPrice"], ascending=True)
        print(sorted_filtered_options[["symbol", "expiration", "strike", "lastPrice", "profit_prediction"]].head(10))

    # Sort by predicted profitability (highest first)
    sorted_filtered_options = price_filtered_df.sort_values(by=["profit_prediction"], ascending=False)

    # Display top options for debugging purposes
    print("\n📊 Remaining Options After Filtering:")
    print(sorted_filtered_options[["symbol", "expiration", "strike", "lastPrice", "profit_prediction"]].head(10))

    return sorted_filtered_options

def compute_volatility_features(options_df, ticker):
    today = datetime.today()
    one_year_ago = today - timedelta(days=365)
    hist = yf.download(ticker, start=one_year_ago.strftime('%Y-%m-%d'), end=today.strftime('%Y-%m-%d'), progress=False)

    hist['daily_return'] = hist['Close'].pct_change()
    hist['hv_1w'] = hist['daily_return'].rolling(window=5).std()
    hist['hv_1m'] = hist['daily_return'].rolling(window=20).std()
    hist['hv_rolling_1m'] = hist['daily_return'].rolling(window=20).std()

    hv_year_low = hist['hv_rolling_1m'].min()
    hv_year_high = hist['hv_rolling_1m'].max()
    hv_current = hist['hv_rolling_1m'].iloc[-1]
    hv_change_1w = hv_current - hist['hv_1w'].iloc[-6]  # 6 days back
    hv_change_1m = hv_current - hist['hv_1m'].iloc[-21]  # 21 days back

    options_df['hv_current'] = hv_current
    options_df['hv_change_1w'] = hv_change_1w
    options_df['hv_change_1m'] = hv_change_1m
    options_df['hv_year_low'] = hv_year_low
    options_df['hv_year_high'] = hv_year_high
    options_df['hv_range_pct'] = (hv_current - hv_year_low) / (hv_year_high - hv_year_low + 1e-6)

    return options_df

historical_vol_df = None  # global cache

def merge_historical_volatility(options_df, ticker, hist_csv_path="data/merged_war_comp.csv"):
    global historical_vol_df
    if historical_vol_df is None:
        historical_vol_df = pd.read_csv(hist_csv_path, parse_dates=['date', 'expiration'])
        historical_vol_df['call_put'] = historical_vol_df['call_put'].map({'Call': 'C', 'Put': 'P', 'Put': 'P', 'P': 'P', 'C': 'C'})
        historical_vol_df['act_symbol'] = historical_vol_df['act_symbol'].str.upper()
        historical_vol_df['merge_key'] = (
            historical_vol_df['date'].dt.normalize().astype(str) + "_" +
            historical_vol_df['act_symbol'] + "_" +
            historical_vol_df['expiration'].astype(str) + "_" +
            historical_vol_df['strike'].astype(str) + "_" +
            historical_vol_df['call_put']
        )

    options_df['quote_date'] = pd.to_datetime(options_df['quote_date']).dt.normalize()
    options_df['act_symbol'] = ticker.upper()
    options_df['call_put'] = options_df['type'].str[0].str.upper()  # 'call' or 'put' → 'C' or 'P'
    options_df['merge_key'] = (
        options_df['quote_date'].astype(str) + "_" +
        options_df['act_symbol'] + "_" +
        options_df['expiration'].astype(str) + "_" +
        options_df['strike'].astype(str) + "_" +
        options_df['call_put']
    )

    merged = options_df.merge(
        historical_vol_df.drop(columns=["stock_price", "bid", "ask", "delta", "gamma", "theta", "vega", "rho"], errors="ignore"),
        how="left",
        on="merge_key",
        suffixes=('', '_hist')
    )

    return merged.drop(columns=["merge_key", "act_symbol", "call_put"])

def prep_features(options_df, ticker):
    stock_price = get_single_stock_price(ticker)
    options_df['stock_price'] = stock_price
    options_df['bid_ask_spread'] = options_df['ask'] - options_df['bid']
    options_df['moneyness'] = options_df['stock_price'] / options_df['strike']
    
    # Pull IV from option chain if available
    if 'impliedVolatility' in options_df.columns:
        options_df['iv_current'] = options_df['impliedVolatility']
    else:
        options_df['iv_current'] = 0.3  # fallback

    # Mock historical IV for now — future: fetch historical snapshots
    options_df["iv_week_ago"] = options_df["iv_current"] * 0.95
    options_df["iv_month_ago"] = options_df["iv_current"] * 0.9
    options_df["iv_year_low"] = options_df["iv_current"] * 0.75
    options_df["iv_year_high"] = options_df["iv_current"] * 1.25

    options_df["iv_change_1w"] = options_df["iv_current"] - options_df["iv_week_ago"]
    options_df["iv_change_1m"] = options_df["iv_current"] - options_df["iv_month_ago"]
    options_df["iv_range_pct"] = (options_df["iv_current"] - options_df["iv_year_low"]) / (
        options_df["iv_year_high"] - options_df["iv_year_low"] + 1e-6)

    options_df = compute_volatility_features(options_df, ticker)
    options_df = add_spy_pct_5d(options_df)

    return options_df


# def prep_features(options_df, ticker):
#     stock_price = get_single_stock_price(ticker)
#     options_df['stock_price'] = stock_price
#     print("columns: ",options_df.columns)
#     options_df['bid_ask_spread'] = options_df['ask'] - options_df['bid']
#     options_df = add_spy_pct_5d(options_df)
#     print("🧪 Final columns in options_df:", options_df.columns)
#     print("📊 Sample of spy_pct_5d:", options_df.get("spy_pct_5d", "❌ Not Found").head())

#     return options_df
    
import yfinance as yf

def add_spy_pct_5d(options_df):
    # Normalize quote_date
    options_df['quote_date'] = pd.to_datetime(options_df['quote_date']).dt.normalize()

    # Fetch extended SPY history
    start = options_df['quote_date'].min() - pd.Timedelta(days=10)
    end = options_df['quote_date'].max() + pd.Timedelta(days=1)
    spy_data = yf.download('SPY', start=start, end=end, auto_adjust=False, progress=False)

    if 'Close' not in spy_data.columns:
        raise ValueError("SPY data download failed or missing 'Close' column.")

    # Compute 5-day return and shift
    spy_returns = spy_data['Close'].pct_change(5)
    spy_df = spy_returns.reset_index()
    spy_df.rename(columns={"SPY": "spy_pct_5d"}, inplace=True)
    spy_df['Date'] = pd.to_datetime(spy_df['Date']).dt.normalize()

    # Match datetime64[ns] resolution
    options_df['quote_date'] = options_df['quote_date'].astype("datetime64[ns]")
    spy_df['Date'] = spy_df['Date'].astype("datetime64[ns]")

    print("🔍 Quote date sample:", options_df['quote_date'].drop_duplicates().sort_values().tail(10))
    print("📈 SPY return index sample:", spy_df['Date'].drop_duplicates().sort_values().tail(10))


    # Merge asof
    merged = pd.merge_asof(
        options_df.sort_values("quote_date"),
        spy_df.sort_values("Date"),
        left_on="quote_date",
        right_on="Date",
        direction="backward"
    ).drop(columns=["Date"])

    return merged







# def main():
#     tickers = ['LHX','GE','RTX','ESLT','MRCY','DRS','NOC','LMT','ERJ','HWM','HEI','VOOG','CDRE','WWD','AXON','CW','TDG','GD']
#     risk_free_rate = get_risk_free_rate()
#     all_profitable_options = []

#     for ticker in tickers:
#         options_df = get_options_data(ticker, risk_free_rate)
#         options_df = prep_features(options_df, ticker)
#         if not options_df.empty:
#             profitable_options = predict_profitability(options_df)
#             all_profitable_options.append(profitable_options)

#     if all_profitable_options:
#         combined_profitable_options = pd.concat(all_profitable_options, ignore_index=True)
#         print_top_option(combined_profitable_options)
#     else:
#         print("⚠️ No profitable options found across all tickers.")

# main()

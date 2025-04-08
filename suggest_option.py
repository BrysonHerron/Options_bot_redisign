import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import fetch_options as fo

# Load models
clf_model_C = load_model("models/keras_clf_model_C.h5")
clf_model_P = load_model("models/keras_clf_model_P.h5")
reg_model_C = load_model("models/keras_reg_model_C.h5")
reg_model_P = load_model("models/keras_reg_model_P.h5")

# === Features used during training ===
FEATURE_COLUMNS = [
    "strike", "bid", "ask",
    "delta", "gamma", "theta", "vega", "rho",
    "iv_current", "iv_change_1w", "iv_change_1m", "iv_range_pct",
    "hv_current", "hv_change_1w", "hv_change_1m", "hv_range_pct",
    "time_to_expiration", "bid_ask_spread", "moneyness", "spy_pct_5d"
]


def predict_profitability(options_df):
    scaler = StandardScaler()
    predictions = []

    # Scale features for the entire dataframe
    X = options_df[FEATURE_COLUMNS].copy()

    print("head: ", X.head())

    # Drop any rows with NaNs or infinite values
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    valid_mask = X.notnull().all(axis=1)
    X = X[valid_mask]
    options_df = options_df[valid_mask]

    # Now scale
    X_scaled = scaler.fit_transform(X)

    # Prediction value filter
    MIN_PROB = 0.85

    for features, (_, row) in zip(X_scaled, options_df.iterrows()):
        features = features.reshape(1, -1)
        opt_type = row['type'].upper()

        if opt_type == 'CALL' or opt_type == 'C':
            clf_pred = clf_model_C.predict(features, verbose=0)[0][0]
            print("  🔎 clf_pred prob:", clf_pred)
            if clf_pred < MIN_PROB:
                predictions.append((0, 0))
                continue
            reg_pred = reg_model_C.predict(features, verbose=0)[0][0]
            predictions.append((1, reg_pred))
        elif opt_type == 'PUT' or opt_type == 'P':
            clf_pred = clf_model_P.predict(features, verbose=0)[0][0]
            if clf_pred < MIN_PROB:
                predictions.append((0, 0))
                continue
            reg_pred = reg_model_P.predict(features, verbose=0)[0][0]
            predictions.append((1, reg_pred))
        else:
            predictions.append((0, 0))

    if predictions:
        clf_preds, reg_preds = zip(*predictions)
        options_df['clf_pred'] = clf_preds
        options_df['predicted_return'] = reg_preds
    else:
        options_df['clf_pred'] = []
        options_df['predicted_return'] = []

    return options_df[options_df['clf_pred'] == 1].sort_values(by='predicted_return', ascending=False)

# === Example usage ===
tickers = ['LHX','GE','RTX','ESLT','MRCY','DRS','NOC','LMT','ERJ','HWM','HEI','VOOG','CDRE','WWD','AXON','CW','TDG','GD']
risk_free_rate = fo.get_risk_free_rate()
all_profitable_options = []

for ticker in tickers:
    options_df = fo.get_options_data(ticker, risk_free_rate)
    options_df = fo.prep_features(options_df, ticker)
    if not options_df.empty:
        options_df = predict_profitability(options_df)  # keep full DF with clf_pred and predicted_return
        profitable_options = options_df[options_df['clf_pred'] == 1].sort_values(by='predicted_return', ascending=False)
        print(f"🔍 {ticker}:")
        print("  ✅ Profitable (clf_pred == 1):", (options_df['clf_pred'] == 1).sum())
        print("  📉 Unprofitable (clf_pred == 0):", (options_df['clf_pred'] == 0).sum())
        all_profitable_options.append(profitable_options)


if all_profitable_options:
    combined_profitable_options = pd.concat(all_profitable_options, ignore_index=True)
    ranked_df = combined_profitable_options.sort_values(by='predicted_return', ascending=False)
    # Price filter
    ranked_df = ranked_df[ranked_df['lastPrice'] < 5]
    print("\n🏁 Final Ranked Options:")
    top_options = ranked_df.head(5)
    for _, top_option in top_options.iterrows():
        print(f"✅ Recommended Option: {top_option['symbol']}")
        print(f"   - Expiration Date: {top_option['expiration']}")
        print(f"   - Profit: {top_option['predicted_return']}")
        print(f"   - Strike Price: {top_option['strike']}")
        print(f"   - Price: ${top_option['lastPrice']:.2f}")
        print(f"   - Type: {top_option['type']}")
        print()

else:
    print("⚠️ No profitable options found across all tickers.")


# === Refactored Options Training Script with Separate Call/Put Combo Models and Keras Tuner ===
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, classification_report
from sklearn.preprocessing import StandardScaler
import optuna
import yfinance as yf
import keras_tuner as kt
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

# Force TensorFlow to use GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print("✅ GPU detected and enabled.")
    except Exception as e:
        print("⚠️ GPU configuration error:", e)
else:
    print("⚠️ No GPU found. Training will use CPU.")

LOG_FILE = "results/model_training_results.txt"

def log_result(message):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} — {message}\n")

# === Load Historical Options Data with Macro Features ===
def load_historical_data(csv_path):
    df = pd.read_csv(csv_path, parse_dates=['date', 'expiration'])
    df['call_put'] = df['call_put'].map({'Call': 'C', 'Put': 'P'})
    print("Initial CSV rows:", len(df))

    df['time_to_expiration'] = (df['expiration'] - df['date']).dt.days / 365
    df['bid_ask_spread'] = df['ask'] - df['bid']
    df['moneyness'] = df['stock_price'] / df['strike']

    # IV Features
    df["iv_change_1w"] = df["iv_current"] - df["iv_week_ago"]
    df["iv_change_1m"] = df["iv_current"] - df["iv_month_ago"]
    df["iv_range_pct"] = (df["iv_current"] - df["iv_year_low"]) / (df["iv_year_high"] - df["iv_year_low"])

    # HV Features
    df["hv_change_1w"] = df["hv_current"] - df["hv_week_ago"]
    df["hv_change_1m"] = df["hv_current"] - df["hv_month_ago"]
    df["hv_range_pct"] = (df["hv_current"] - df["hv_year_low"]) / (df["hv_year_high"] - df["hv_year_low"])

    # Normalize price change direction based on call/put
    df['future_bid'] = df.groupby(['act_symbol', 'strike', 'call_put'])['bid'].shift(-10)
    direction_factor = df['call_put'].map({'C': 1, 'P': -1})
    df['price_change'] = direction_factor * (df['future_bid'] - df['bid'])

    df['target_reg'] = np.sign(df['price_change']) * np.log1p(abs(df['price_change'] / df['bid']))
    df['target_reg'] = np.clip(df['target_reg'], -2, 2)
    df['target_clf'] = (df['price_change'] > 0).astype(int)

    print("After future_bid shift and price_change calc:", len(df))

    # Download and process SPY context
    spy_data = yf.download('SPY', start=df['date'].min(), end=df['date'].max(), auto_adjust=False)

    if 'Close' not in spy_data.columns:
        raise ValueError("SPY data download failed or missing 'Close' column.")

    spy_series = spy_data['Close']
    if isinstance(spy_series, pd.DataFrame):
        spy_series = spy_series.iloc[:, 0]

    spy_series = spy_series.pct_change(5)
    spy_series.name = 'spy_pct_5d'
    spy_df = pd.DataFrame(spy_series)

    # Ensure datetime alignment
    df['date'] = pd.to_datetime(df['date']).dt.normalize().dt.tz_localize(None)
    spy_df.index = pd.to_datetime(spy_df.index).normalize().tz_localize(None)

    print("SPY index range:", spy_df.index.min(), "to", spy_df.index.max())
    print("DF date range:", df['date'].min(), "to", df['date'].max())

    df = df.merge(spy_df, how='left', left_on='date', right_index=True, validate='many_to_one')
    print("Post-merge shape:", df.shape)
    print("Columns after merge:", df.columns.tolist())

    if 'spy_pct_5d' not in df.columns:
        raise ValueError("spy_pct_5d not found in merged DataFrame.")

    print("spy_pct_5d NaNs:", df['spy_pct_5d'].isna().sum())

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    print("Final cleaned dataset length:", len(df))
    return df

# === Feature Definitions ===
#rm 'vol'
FEATURE_COLUMNS = [
    "strike", "bid", "ask",
    "delta", "gamma", "theta", "vega", "rho",
    "iv_current", "iv_change_1w", "iv_change_1m", "iv_range_pct",
    "hv_current", "hv_change_1w", "hv_change_1m", "hv_range_pct",
    "time_to_expiration", "bid_ask_spread", "moneyness", "spy_pct_5d"
]

# === Load & Prepare Data ===
csv_path = "data/merged_war_comp.csv"
df = load_historical_data(csv_path)

# === Keras Model for Classification with Tuning ===
def build_clf_model(hp):
    model = keras.Sequential()
    model.add(layers.Input(shape=(len(FEATURE_COLUMNS),)))
    for i in range(hp.Int("num_layers", 1, 3)):
        model.add(layers.Dense(
            units=hp.Int(f"units_{i}", min_value=32, max_value=256, step=32),
            activation=hp.Choice("activation", ["relu", "tanh"])
        ))
        model.add(layers.Dropout(hp.Float("dropout", 0.0, 0.5, step=0.1)))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Float("lr", 1e-4, 1e-2, sampling="log")),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# === Keras Model for Regression with Tuning ===
def build_reg_model(hp):
    model = keras.Sequential()
    model.add(layers.Input(shape=(len(FEATURE_COLUMNS),)))
    for i in range(hp.Int("num_layers", 1, 3)):
        model.add(layers.Dense(
            units=hp.Int(f"units_{i}", min_value=32, max_value=256, step=32),
            activation=hp.Choice("activation", ["relu", "tanh"])
        ))
        model.add(layers.Dropout(hp.Float("dropout", 0.0, 0.5, step=0.1)))
    model.add(layers.Dense(1, activation="linear"))
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Float("lr", 1e-4, 1e-2, sampling="log")),
        loss="mean_squared_error",
        metrics=["mae", "mse"]
    )
    return model

# === Train and Save Separate Models for Calls and Puts ===
for opt_type in ['C', 'P']:
    print(f"Tuning Keras model for {'Calls' if opt_type == 'C' else 'Puts'}")
    subset = df[df['call_put'] == opt_type].copy()

    X = subset[FEATURE_COLUMNS]
    y = subset['target_clf']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    tuner = kt.Hyperband(
        build_clf_model,
        objective="val_accuracy",
        max_epochs=20,
        factor=3,
        directory="keras_tuner",
        project_name=f"options_clf_{opt_type}"
    )

    tuner.search(X_train, y_train, epochs=20, validation_data=(X_val, y_val))
    best_model = tuner.get_best_models(1)[0]

    loss, acc = best_model.evaluate(X_val, y_val)
    log_result(f"Classifier accuracy for {opt_type}: {acc:.4f}")

    best_model.save(f"models/keras_clf_model_{opt_type}.h5")
    print(f"📁 Saved keras_clf_model_{opt_type}.h5")

    # === Clean Data & Train and Save Regressor ===
    subset = subset.replace([np.inf, -np.inf], np.nan).dropna(subset=['target_reg'])
    y_reg = subset['target_reg']
    X = subset[FEATURE_COLUMNS]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train_r, X_val_r, y_train_r, y_val_r = train_test_split(
        X_scaled, y_reg, test_size=0.2, random_state=42
    )

    reg_tuner = kt.Hyperband(
        build_reg_model,
        objective="val_loss",
        max_epochs=20,
        factor=3,
        directory="keras_tuner",
        project_name=f"options_reg_{opt_type}"
    )

    reg_tuner.search(X_train_r, y_train_r, epochs=20, validation_data=(X_val_r, y_val_r))
    best_reg_model = reg_tuner.get_best_models(1)[0]

    loss, mae, mse = best_reg_model.evaluate(X_val_r, y_val_r)
    log_result(f"Regressor for {opt_type} — MAE: {mae:.4f}, MSE: {mse:.4f}")

    best_reg_model.save(f"models/keras_reg_model_{opt_type}.h5")
    print(f"💾 Saved keras_reg_model_{opt_type}.h5")

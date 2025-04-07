import matplotlib.pyplot as plt
import joblib

# Load models
clf_call = joblib.load("clf_model_C.pkl")
clf_put = joblib.load("clf_model_P.pkl")

# Feature names (ensure order matches training)
features = [
    "strike", "delta", "gamma", "theta", "vega", "rho",
    "stock_price", "time_to_expiration", "bid_ask_spread", "spy_pct_5d"
]

# Plot feature importance for Calls
plt.figure(figsize=(10, 5))
plt.barh(features, clf_call.feature_importances_)
plt.title("Call Classifier Feature Importance")
plt.xlabel("Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Plot feature importance for Puts
plt.figure(figsize=(10, 5))
plt.barh(features, clf_put.feature_importances_)
plt.title("Put Classifier Feature Importance")
plt.xlabel("Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import joblib
import os

# fetching TMPV.NS data
print("fetching TMPV.NS data...")
df = yf.download("TMPV.NS", start="2019-01-01", end="2026-06-25")

print("actual date range:", df.index.min(), "to", df.index.max())
print("total trading days:", len(df))
print("shape:", df.shape)

print("\nmissing values:")
print(df.isnull().sum())

plt.figure(figsize=(14, 5))
plt.plot(df["Close"], color="steelblue", linewidth=1.3)
plt.title("TMPV.NS - Closing Price")
plt.xlabel("Date")
plt.ylabel("Price (₹)")
plt.tight_layout()
plt.savefig("closing_price_plot.png")
plt.show()

close_prices = df[["Close"]].values
print("close prices shape:", close_prices.shape)

lookback = 60

X_raw, y_raw = [], []
for i in range(lookback, len(close_prices)):
    X_raw.append(close_prices[i-lookback:i, 0])
    y_raw.append(close_prices[i, 0])

X_raw = np.array(X_raw)
y_raw = np.array(y_raw)

print("X_raw shape:", X_raw.shape)
print("y_raw shape:", y_raw.shape)

# strict time-based split
split = int(len(X_raw) * 0.80)

X_train_raw = X_raw[:split]
X_test_raw  = X_raw[split:]
y_train_raw = y_raw[:split]
y_test_raw  = y_raw[split:]

print("train samples:", len(X_train_raw))
print("test samples:", len(X_test_raw))

scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

# fit and transform X — fit on train only, transform test
X_train_scaled = scaler_X.fit_transform(X_train_raw)
X_test_scaled  = scaler_X.transform(X_test_raw)

y_train = scaler_y.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
y_test  = scaler_y.transform(y_test_raw.reshape(-1, 1)).flatten()

print("scalers fit on training data only")

X_train = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test  = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

os.makedirs("data", exist_ok=True)

np.save("data/X_train.npy", X_train)
np.save("data/X_test.npy", X_test)
np.save("data/y_train.npy", y_train)
np.save("data/y_test.npy", y_test)
np.save("data/close_prices.npy", close_prices)

joblib.dump(scaler_X, "data/scaler_X.pkl")
joblib.dump(scaler_y, "data/scaler_y.pkl")
print("all arrays and scalers saved to data/")
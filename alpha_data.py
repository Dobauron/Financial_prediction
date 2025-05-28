import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# === 1. Ustawienia ===
API_KEY = "d0njs2hr01qn5ghk6m7gd0njs2hr01qn5ghk6m80"
symbol = "TSLA"

# === 2. Pobieranie danych ===
url = f"https://finnhub.io/api/v1/stock/candle?symbol={symbol}&resolution=D&from=1451606400&to=9999999999&token={API_KEY}"
# from=1451606400 to 9999999999 oznacza zakres od 2016-01-01 do teraz (timestampy)

response = requests.get(url)
data = response.json()
print(data)
if data.get("s") != "ok":
    raise ValueError(f"Błąd pobierania danych: {data.get('error', 'nieznany błąd')}")

# === 3. Tworzenie DataFrame ===
df = pd.DataFrame({
    "date": pd.to_datetime(data['t'], unit='s'),
    "open": data['o'],
    "high": data['h'],
    "low": data['l'],
    "close": data['c'],
    "volume": data['v']
})

df = df.sort_values('date').reset_index(drop=True)

# === 4. Przygotowanie cechy i targetu ===
df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
df = df.dropna()

features = ['open', 'high', 'low', 'close', 'volume']
X = df[features]
y = df['target']

# === 5. Podział na trening i test ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# === 6. Trening modelu ===
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# === 7. Ewaluacja ===
y_pred = model.predict(X_test)
print("=== Raport klasyfikacji ===")
print(classification_report(y_test, y_pred))

# === 8. Wykres ===
plt.figure(figsize=(12, 4))
plt.plot(y_test.values, label='Prawdziwy kierunek')
plt.plot(y_pred, label='Przewidziany kierunek', alpha=0.7)
plt.title(f"Predykcja kierunku ceny akcji {symbol} (Finnhub)")
plt.legend()
plt.grid()
plt.show()

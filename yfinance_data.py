import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

symbol = "TSLA"
ticker = yf.Ticker(symbol)
# === 1. Dane giełdowe ===
print(ticker.info)
print(ticker.fast_info)
price_df = ticker.history(period="3y")
price_df = price_df.reset_index()
price_df['target'] = (price_df['Close'].shift(-1) > price_df['Close']).astype(int)
price_df = price_df.dropna()

# === 2. Dane fundamentalne (quarterly) ===
financials = ticker.quarterly_financials.T
balance_sheet = ticker.quarterly_balance_sheet.T
info = ticker.fast_info

# Spróbuj wyciągnąć niektóre metryki (upewnij się, że są dostępne)
try:
    pe_ratio = info.get("trailingPE", None)
    roe = info.get("returnOnEquity", None)
    profit_margin = info.get("profitMargins", None)
except:
    pe_ratio = roe = profit_margin = None

# Wybieramy kilka fundamentalnych wskaźników z quarterly reports
fundamentals = pd.DataFrame()
fundamentals['date'] = financials.index
fundamentals['revenue'] = financials['Total Revenue']
fundamentals['net_income'] = financials['Net Income']
fundamentals['eps'] = financials['Diluted EPS']
fundamentals = fundamentals.dropna()
fundamentals['date'] = pd.to_datetime(fundamentals['date'])

# === 3. Przygotowanie danych ===
# Grupujemy ceny na kwartały i łączymy z fundamentalnymi
price_df['Date'] = pd.to_datetime(price_df['Date'])
quarterly_price = price_df[['Date', 'Close', 'target']].copy()
quarterly_price = quarterly_price.set_index('Date').resample('QE').last().dropna().reset_index()

df = pd.merge(quarterly_price, fundamentals, left_on='Date', right_on='date')
df['target'] = df['target'].astype(int)
print("Rozmiar quarterly_price:", len(quarterly_price))
print("Rozmiar fundamentals:", len(fundamentals))
print("Rozmiar po mergu:", len(df))

# === 4. Trening ===
features = ['revenue', 'net_income', 'eps']
X = df[features]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# === 5. Ewaluacja ===
y_pred = model.predict(X_test)
print("=== Raport klasyfikacji ===")
print(classification_report(y_test, y_pred))

# === 6. Wykres ===
plt.plot(y_test.values, label='Prawdziwy kierunek')
plt.plot(y_pred, label='Przewidziany kierunek', alpha=0.7)
plt.title("Predykcja zmiany ceny akcji TSLA (na podstawie fundamentów)")
plt.legend()
plt.grid()
plt.show()

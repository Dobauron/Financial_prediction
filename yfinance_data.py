import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# 1. Pobieranie danych z S&P 500
print("Pobieranie danych...")
data = yf.download("^GSPC", start="2025-01-01")

# 2. Obsługa MultiIndex (jeśli się pojawi)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.droplevel(1)

# 3. Sprawdzenie poprawności danych
if data.empty:
    raise ValueError("Dane się nie pobrały – sprawdź połączenie lub zakres dat.")

print("Dostępne kolumny:", data.columns)

required_columns = ['High', 'Low', 'Volume', 'Close']
missing = [col for col in required_columns if col not in data.columns]
if missing:
    raise ValueError(f"Brakuje kolumn: {missing}")

# 4. Przygotowanie danych
data = data.dropna(subset=required_columns)
X = data[['High', 'Low', 'Volume']]
y = data['Close']

# 5. Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 6. Trening modelu regresji liniowej
model = LinearRegression()
model.fit(X_train, y_train)

# 7. Predykcja i metryki
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n📊 Wyniki regresji:")
print("Współczynniki regresji:", model.coef_)
print("Wyraz wolny (intercept):", model.intercept_)
print(f"R² score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")


# 8. Prognoza na przyszłość z datami
def forecast_next_days(last_data, model, last_date, n_days=5):
    forecast = []
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_days, freq='B')  # B = dni robocze

    for date in future_dates:
        prediction = model.predict([last_data])[0]
        forecast.append((date, prediction))
        # Można by tu dodać logikę do generowania nowych 'last_data' (np. na podstawie średnich, szumu itp.)

    return forecast


# Ostatni wiersz danych testowych
last_data = X_test.iloc[-1].values
last_date = y_test.index[-1]

# Prognoza na 5 dni do przodu
forecast = forecast_next_days(last_data, model, last_date, n_days=5)

# Tworzymy dane do wykresu
y_test_sorted = y_test.sort_index()
dates = list(y_test_sorted.index)
values = y_test_sorted.values

forecast_dates = [f[0] for f in forecast]
forecast_values = [f[1] for f in forecast]

# 9. Wizualizacja wyników
plt.figure(figsize=(12, 6))

# Rzeczywiste dane
plt.plot(dates, values, label="Rzeczywiste dane", color="blue")

# Predykcje testowe
plt.plot(dates, y_pred[:len(dates)], label="Predykcja testowa", color="orange", linestyle="--")

# Prognoza przyszłości
plt.plot(forecast_dates, forecast_values, label="Prognoza (kolejne dni)", color="green", marker='o')

# Formatowanie wykresu
plt.xlabel("Data")
plt.ylabel("Wartość Close")
plt.title("Predykcja wartości 'Close' dla S&P 500 z prognozą na przyszłość")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


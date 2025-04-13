import yfinance as yf

# Pobieranie danych dla akcji Apple (AAPL)
data = yf.download('^GSPC')

print(data.columns)

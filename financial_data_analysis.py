import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split
import scipy.stats
import numpy as np


class FinancialAnalysis:
    def __init__(self):
        self.dataset = None
        self.model = linear_model.LinearRegression()

    def load_file(self, path):
        try:
            self.dataset = pd.read_csv(path)
            self.clean_data()
            return True
        except Exception as e:
            messagebox.showerror("Błąd", f"Nie udało się wczytać pliku: {e}")
            return False

    def clean_data(self):
        self.dataset.columns = self.dataset.columns.str.strip()
        cols = ["Profit", "Sale Price", "Units Sold", "Manufacturing Price", "Gross Sales", "COGS"]
        for col in cols:
            if col in self.dataset.columns:
                self.dataset[col] = self.dataset[col].replace(r'[\$,]', '', regex=True)
                self.dataset[col] = pd.to_numeric(self.dataset[col], errors='coerce')
                self.dataset = self.dataset[self.dataset[col].notna()]

    def show_boxplot_and_hist(self):
        self.dataset.plot(kind="box", title="Boxplot danych finansowych")
        plt.show()
        self.dataset.hist(bins=100)
        plt.show()

    def show_scatter_matrix(self):
        plots = [
            ("Sale Price", "Profit", "Stosunek ceny sprzedaży do zysków"),
            ("Gross Sales", "Profit", "Sprzedaż brutto do zysków"),
            ("COGS", "Profit", "Koszta produktu do zysków"),
            ("Units Sold", "Profit", "Ilość sprzedanych produktów do zysków"),
            ("Manufacturing Price", "Profit", "Cena wyprodukowania do zysków"),
        ]
        fig, axes = plt.subplots(1, 5, figsize=(18, 6))
        for i, (x, y, title) in enumerate(plots):
            if x in self.dataset.columns and y in self.dataset.columns:
                axes[i].scatter(self.dataset[x], self.dataset[y], color='orange')
                axes[i].set_title(title)
                axes[i].set_xlabel(x)
                axes[i].set_ylabel(y)
        plt.tight_layout()
        plt.show()

    def compute_pearsonr(self):
        if "Profit" not in self.dataset.columns:
            messagebox.showwarning("Brak danych", "Kolumna 'Profit' nie istnieje w danych.")
            return

        correlations = []
        for col in self.dataset.columns:
            if col != "Profit" and pd.api.types.is_numeric_dtype(self.dataset[col]):
                corr, _ = scipy.stats.pearsonr(self.dataset["Profit"], self.dataset[col])
                correlations.append((col, corr))

        if correlations:
            result_text = "Współczynniki korelacji Pearsona względem 'Profit':\n\n"
            result_text += "\n".join([f"{col}: {corr:.4f}" for col, corr in correlations])
            messagebox.showinfo("Korelacja Pearsona", result_text)
        else:
            messagebox.showwarning("Brak danych", "Brak wystarczających danych liczbowych do obliczenia korelacji.")

    def train_and_evaluate_model(self):
        if 'COGS' not in self.dataset.columns or 'Profit' not in self.dataset.columns:
            messagebox.showwarning("Brak danych", "Brakuje kolumn COGS lub Profit.")
            return

        X = pd.DataFrame(self.dataset['COGS'])
        y = pd.DataFrame(self.dataset['Profit'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        a = self.model.coef_[0][0]
        b = self.model.intercept_[0]
        r2 = self.model.score(X_train, y_train)
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

        result = f"""Regresja liniowa:
Współczynnik a: {a:.4f}
Wyraz wolny b: {b:.4f}
R² score: {r2:.4f}
RMSE: {rmse:.2f}"""

        messagebox.showinfo("Wyniki modelu", result)

        plt.plot(X_test, y_pred, color='red', label='Predykcja')
        plt.scatter(X_test, y_test, color='blue', label='Dane testowe')
        plt.xlabel('COGS')
        plt.ylabel('Profit')
        plt.title('Regresja liniowa: Profit vs COGS')
        plt.legend()
        plt.show()


class FinancialApp:
    def __init__(self, root):
        self.analysis = FinancialAnalysis()
        self.root = root
        self.root.title("Analiza Finansowa - GUI")

        tk.Button(root, text="📂 Wczytaj plik CSV", command=self.load_csv).pack(pady=5)
        tk.Button(root, text="📊 Boxplot i Histogram", command=self.analysis.show_boxplot_and_hist).pack(pady=5)
        tk.Button(root, text="📉 Wykresy rozrzutu", command=self.analysis.show_scatter_matrix).pack(pady=5)
        tk.Button(root, text="📈 Korelacja Pearsona", command=self.analysis.compute_pearsonr).pack(pady=5)
        tk.Button(root, text="🧠 Trenuj i oceń model", command=self.analysis.train_and_evaluate_model).pack(pady=5)

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            if self.analysis.load_file(file_path):
                messagebox.showinfo("Sukces", f"Wczytano dane z pliku:\n{file_path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = FinancialApp(root)
    root.mainloop()

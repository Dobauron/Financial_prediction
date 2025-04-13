from ML_interface import MLModelInterface
import pandas as pd # do pracy z danymi
import matplotlib.pyplot as plt # do wizualizacji wykresów
from sklearn import linear_model # model funkcji liniowej
import scipy.stats # do operacji statystycznych
from sklearn.model_selection import train_test_split # do podziału na dane treningowe i testowe
from sklearn import metrics #metryki do oceny modelu
import numpy as np # do operacji numerycznych z algebra liniowa


class BaseMLModel(MLModelInterface):
    def __init__(self):
        self.dataset = None

    def load_file(self, path):
        try:
            self.dataset = pd.read_csv(path)
            return True
        except Exception as e:
            print(f"Error loadng file: {e}")
            return False

class HumanBodyAnalysis(BaseMLModel):
    def __init__(self):
        self.model = linear_model.LinearRegression()

    def load_data(self, path):
        return self.load_file(path)

    def clean_data(self):
        self.dataset.columns = self.dataset.columns.str.strip()

        # Upewnij się, że liczby są liczbami (czyli konwersja + usunięcie NaN)
        numeric_cols = ["Height", "Weight"]
        for col in numeric_cols:
            self.dataset[col] = pd.to_numeric(self.dataset[col], errors='coerce')
        # Usuń wiersze z brakami danych
        self.dataset.dropna(subset=numeric_cols, inplace=True)
        # Konwersja jednostek
        self.dataset["Height_cm"] = self.dataset["Height"] * 2.54  # inch → cm
        self.dataset["Weight_kg"] = self.dataset["Weight"] * 0.453592  # lbs → kg
        # Usuń nadmiarowe spacje w stringach
        if 'Gender' in self.dataset.columns:
            self.dataset['Gender'] = self.dataset['Gender'].str.strip().str.title()

    def explore_data(self):
        plt.scatter(self.dataset["Height_cm"], self.dataset["Weight_kg"], color='green', alpha=0.6)
        plt.xlabel("Wzrost (cm)")
        plt.ylabel("Waga (kg)")
        plt.title("Wzrost vs Waga")
        plt.grid(True)
        plt.show()

    def train_model(self):
        HeightDataFrameX = pd.DataFrame(self.dataset["Height"])
        WeightDataFrameY = pd.DataFrame(self.dataset["Weight"])

        x_train, x_test, y_train,y_test = train_test_split(HeightDataFrameX,WeightDataFrameY,test_size=0.25,random_state=0)


        self.model.fit(x_train,y_train)
        predWeightY = self.model.predict(x_test)

        a = self.model.coef_[0][0]
        b = self.model.intercept_[0]
        r2 = self.model.score(x_train, y_train)
        rmse = np.sqrt(metrics.mean_squared_error(y_test, predWeightY))
        result = f"""Regresja liniowa:
        Współczynnik a: {a:.4f}
        Wyraz wolny b: {b:.4f}
        R² score: {r2:.4f}
        RMSE: {rmse:.2f}"""
        print(result)
        testY = self.model.predict(np.array([[71.7309784033377]]))[0]
        print(testY)
        plt.plot(x_test, predWeightY, label='Wyniki predykcji na podstawie wyuczonego modelu', color='red')
        plt.scatter(x_test, y_test, label='Originalne dane testowe', color='blue')
        plt.legend()
        plt.show()

    def evaluate_model(self):
        pass
    def preprocess_data(self):
        pass
    def visualize_results(self):
        pass

Weight2Height = HumanBodyAnalysis()
Weight2Height.load_data("height-weight.csv")
Weight2Height.clean_data()
Weight2Height.explore_data()
Weight2Height.train_model()
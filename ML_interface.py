from abc import ABC, abstractmethod

class MLModelInterface(ABC):

    @abstractmethod
    def load_data(self, path):
        """Wczytuje dane z pliku."""
        pass

    @abstractmethod
    def preprocess_data(self):
        """Czyści/przygotowuje dane do trenowania."""
        pass

    @abstractmethod
    def explore_data(self):
        """Eksploracyjna analiza danych (EDA), np. wykresy, korelacje."""
        pass

    @abstractmethod
    def train_model(self):
        """Trenuje model ML."""
        pass

    @abstractmethod
    def evaluate_model(self):
        """Ewaluuje model (R², RMSE, itp.)."""
        pass

    @abstractmethod
    def visualize_results(self):
        """Wyświetla wykresy z wynikami."""
        pass
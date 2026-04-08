from abc import ABC, abstractmethod

class GMMBaseModel(ABC):
    @abstractmethod
    def fit(self, X):
        pass

    @abstractmethod
    def partial_fit(self, X):
        pass

    @abstractmethod
    def score_samples(self, X):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def predict_anomaly(self, X):
        pass

    @abstractmethod
    def get_state(self):
        pass


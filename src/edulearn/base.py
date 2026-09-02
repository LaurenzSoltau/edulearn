import numpy as np
from numpy.typing import NDArray

class Estimator:
    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        raise NotImplementedError


class Predictor(Estimator):
    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError

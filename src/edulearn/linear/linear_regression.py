import numpy as np
from numpy.typing import NDArray

from edulearn.base import Predictor
from edulearn.optimizer import Optimizer

class LinearRegression(Predictor):
    """
    Create LinearRegression Model

    Paramters:
        fit_b (boolean): wheter or not intercept should be fitted (false means model assumes data is centered around (0,0)).
        method ("analytical", "solver")
        solver (Solver): needed when method = "solver"
        max_epochs (int): maximum number of epochs perform in fit. (only relevant when method="solver")
        tol (np.float64): tolerance for optimizers (only relevant when method="solver")

    Attributes:
        w_ (np.ndarray()) the learned weights (none before fit)
        b_ (np.float64) learned intercept (0.0 when fit_b = false)

    """
    def __init__(self, fit_b: bool, method, solver = Optimizer(), max_epochs = 1000, tol=1e-3):
        self.fit_b = fit_b
        self.method = method
        self.solver = solver
        self.b_ =  np.float64(0)
        self.max_epochs = max_epochs
        self.tol = tol

    def solve_analytical(self, X:NDArray[np.float64], y: NDArray[np.float64]):
        if self.fit_b:
            # augment X and w_
            augmented_X = np.column_stack((np.ones(X.shape[0]), X))
            ## maybe later the library will implement this method itself.
            augmented_w, _, _, _ = np.linalg.lstsq(augmented_X, y) 
            self.w_ = augmented_w[1:]
            self.b_ = augmented_w[0]
        else:
            self.w_, _, _, _ = np.linalg.lstsq(X, y)


    def solve_with_optimizer(self, X:NDArray[np.float64], y: NDArray[np.float64]):
        self.w_ = np.ones(X.shape[1])
        self.b_ = np.float64(0.0)


        rel = 0
        epochs = 0 
        while epochs < self.max_epochs:
            w_grad, b_grad = self._gradients(X, y)

            if epochs == 0:
                rel = max(1, np.linalg.norm(w_grad, ord=np.inf))

            if np.linalg.norm(w_grad, ord=np.inf) < self.tol * rel and np.abs(b_grad) < self.tol * rel:
                break

            self.w_, self.b_ = self.solver.step(self.w_, self.b_, w_grad, b_grad) 
            epochs += 1


    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        if self.method == 'analytical':
            self.solve_analytical(X, y)
        if self.method == 'solver':
            self.solve_with_optimizer(X, y)


    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        return X @ self.w_ + self.b_

    def _gradients(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> tuple[NDArray[np.float64], np.float64]:

        residuals = (self.predict(X) - y)

        n = y.shape[0]
        w_grad = 2/n * X.T @ residuals
        b_grad = np.float64(2/n * residuals.sum()) if self.fit_b else np.float64(0.0)

        return w_grad, b_grad








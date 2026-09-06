import numpy as np
from numpy.typing import NDArray

from edulearn.base import Predictor

class LinearRegression(Predictor):
    """
    Create LinearRegression Model

    Paramters:
        fit_b (boolean): wheter or not intercept should be fitted (false means model assumes data is centered around (0,0)).
        method ('analytical', 'solver')
        solver (Solver) needed when method = 'solver'

    Attributes:
        w_ (np.ndarray()) the learned weights (none before fit)
        b_ () learned intercept (0.0 when fit_b = false)

    """
    def __init__(self, fit_b: bool, method, solver=None):
        self.fit_b = fit_b
        self.method = method
        self.solver = solver
        self.b_ =  np.float64(0)

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

        
        


    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        if self.method == 'analytical':
            self.solve_analytical(X, y)
        if self.method == 'solver':
            raise NotImplementedError("not implemented yet")


        






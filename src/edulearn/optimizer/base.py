import numpy as np
from numpy.typing import NDArray

class Optimizer():
    
    def __init__(self, lr = 0.01):
        self.lr = lr
    
    def step(
        self,
        w: NDArray[np.float64],
        b: np.float64,
        grad_w: NDArray[np.float64],
        grad_b: np.float64) -> tuple[NDArray[np.float64], np.float64]:
        raise NotImplementedError()


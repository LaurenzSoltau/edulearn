from edulearn.optimizer import Optimizer

class GradientDescent(Optimizer):
    """
    Create GradientDescent Optimizer

    Parameters:
        lr (np.float64): learning rate 
    """
    def __init__(self, lr):
        super().__init__(lr)

    def step(self, w, b, grad_w, grad_b):
        w = w - self.lr * grad_w
        b = b - self.lr * grad_b
        return w, b


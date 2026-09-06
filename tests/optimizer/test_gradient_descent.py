import pytest
import numpy as np


from edulearn.optimizer import GradientDescent




@pytest.fixture
def rng():
    return np.random.default_rng()

def test_performs_correct_update():
    w = np.array([1.0, 2.0])
    b = np.float64(3.0)

    grad_w = np.array([0.5, -1.0])
    grad_b = np.float64(2.0)

    gd = GradientDescent(lr=0.1)
    gd_w, gd_b = gd.step(w, b, grad_w, grad_b)

    w_new = [0.95, 2.1]
    b_new = 2.8

    assert(np.allclose(gd_w, w_new))
    assert(np.allclose(gd_b, b_new))

def test_converges():

    optimizer = GradientDescent(lr=0.1)

    w = np.array([-5.0, 8.0])
    b = np.float64(0)
    target = np.array([3.0, -2.0])

    for _ in range(200):
        grad_w = 2 * (w - target)
        w, b = optimizer.step(w, b, grad_w, np.float64(0))

    assert np.allclose(w, target, atol=1e-6)    




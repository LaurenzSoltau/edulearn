import sklearn
import numpy as np
import pytest

from edulearn.linear import LinearRegression
from edulearn.optimizer import GradientDescent

@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_fits_correct_weights_analytical(rng):
    X = rng.normal(size=(1000, 4))
    w = np.array([2.0, -1.0, 1.5, 2.5])
    b = 4.0
    y = X @ w + b

    model = LinearRegression(True, 'analytical')
    model.fit(X, y)
    assert(np.allclose(w, model.w_))
    assert(np.isclose(b, model.b_))

def test_linear_regression_analytical_integrated(rng):
    X = rng.normal(size=(1000, 4))
    true_w = np.array([-0.45, 2, 1.5, -3])
    true_b = 3
    y = X @ true_w + true_b

    reference_model = sklearn.linear_model.LinearRegression()
    model = LinearRegression(fit_b=True, method="analytical")

    reference_model.fit(X, y)
    model.fit(X, y)

    assert(np.allclose(reference_model.coef_, model.w_))
    assert(np.isclose(reference_model.intercept_, model.b_))

def test_linear_regression_gd_integrated(rng):
    X = rng.normal(size=(1000, 4))
    true_w = np.array([-0.45, 2, 1.5, -3])
    true_b = 3
    y = X @ true_w + true_b

    reference_model = sklearn.linear_model.LinearRegression()
    model = LinearRegression(fit_b=True, method="solver", solver=GradientDescent(lr=0.01), tol=1e-4)

    reference_model.fit(X, y)
    model.fit(X, y)

    assert(np.allclose(reference_model.coef_, model.w_, atol=8e-4))
    assert(np.isclose(reference_model.intercept_, model.b_, atol=8e-4))






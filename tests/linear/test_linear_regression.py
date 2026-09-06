import numpy as np

from edulearn.linear import LinearRegression


def test_fits_correct_weights():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1000, 4))
    w = np.array([2.0, -1.0, 1.5, 2.5])
    b = 4.0
    y = X @ w + b

    model = LinearRegression(True, 'analytical')
    model.fit(X, y)
    assert(np.allclose(w, model.w_))
    assert(np.isclose(b, model.b_))
        
    





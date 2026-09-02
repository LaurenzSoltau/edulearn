# Linear Regression

## 1. What Problem does it solve?

Linnear regression is used to predict a real Number based on Input Data (typically a vector of real numbers). For example predicted housing prices based on features like square footage or number of bedrooms.

## 2. Mathematical model
When using linear regression, its convinient to use the functional interpretation, which means, we make the assumption, that the mechanism where we picked our samples from (($\mathb{x}, \mathb{y})$) behaves linearly, meaning we assume there exists a function f going from D -> B with f_w(x) = w0 + w1*x1 + ... + wk*xk = y. Our goal with linear regression is, to find w based on the samples from the dataset. We do this, by finding the w, such that f_w would predict the y from out dataset the best.

## 3. Loss function
we messure the goodness of a prediction, by using a Loss function. A loss function tells how, how good a single prediction is. (y_i - f(y_i)). Because we are not interested in the direction of the error we use the square of this (absolute is worse to compute derivitaves). so (y_i - f(x_i))^2. By summing over these and averging we get the MSE. If we take the root, to cancel the square we get the RMSE Root Mean Square Error. This is called the expected loss
Because the Loss function is dependet on the function f, it also depends on the paramters w. So our goal is to find the parameters B, which minimize the loss function. (argmin w (expected loss function_w))
There are many ways how to to this. We will cover a couple

4. Derivation / Solution
Using the Normal Equation.
TODO

Using Pseudo inverse for more Robustness
TODO

Using Gradient Descent
TODO

(More Advanced Methods)
TODO (LATER)

6. Implementation Info
TODO




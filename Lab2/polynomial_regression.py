import numpy as np
import pylab as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error


def make_regression_dataset():
    x = np.random.uniform(low=1, high=10, size=(100, 2))
    y = (3.14 * x.T[0] ** 1.1 - 2.7 * np.log(x.T[1]) ** 3 + 2.9
         + np.random.normal(0, 1, size=100))
    return x, y


def polinomial_regression(x, y, n_degree):
    x = PolynomialFeatures(n_degree).fit_transform(x)
    xTx = np.dot(x.T, x)
    xy = np.dot(x.T, y)
    w = np.dot(np.linalg.inv(xTx), xy)
    y_pred = np.dot(x, w)
    print(f'Degree: {n_degree}; MSE: {mean_squared_error(y, y_pred)}')


if __name__ == '__main__':
    x, y = make_regression_dataset()
    for i in range(15):
        polinomial_regression(x, y, i)

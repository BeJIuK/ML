import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels

class SupportVectorMachine:

    def __init__(self, C=1.0, max_iters=10000, n_passes=10, eps=1e-4):
        self.C = C
        self.max_iters = max_iters
        self.n_passes = n_passes
        self.eps = eps

    def fit(self, X, y, verbose=True):
        K = pairwise_kernels(X)
        yK = y * K
        dual_coef = np.zeros(X.shape[0])

        n_samples = K.shape[0]

        it = 0
        passes = 0
        alphas_changed = 0
        b = 0.0
        C = self.C
        while passes < self.n_passes and it < self.max_iters:
            alphas_changed = 0
            for i in range(n_samples):
                Ei = b + np.dot(dual_coef, yK[i]) - y[i]
                yEi = y[i] * Ei
                if ((yEi < -self.eps and dual_coef[i] < C) or (yEi > self.eps and dual_coef[i] > 0)):
                    j = i
                    while i == j:
                        j = np.random.randint(n_samples)
                    Ej = b + np.dot(dual_coef, yK[j]) - y[j]

                    ai = dual_coef[i]
                    aj = dual_coef[j]

                    if y[i] == y[j]:
                        L = max(0, ai + aj - C)
                        H = min(C, ai + aj)
                    else:
                        L = max(0, aj - ai)
                        H = min(C, aj - ai + C)

                    if abs(L - H) < self.eps:
                        continue

                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue

                    newaj = aj - y[j] * (Ei - Ej) / eta
                    newaj = min(newaj, H)
                    newaj = max(newaj, L)
                    if abs(aj - newaj) < 1e-4:
                        continue
                    dual_coef[j] = newaj
                    newai = ai + y[i] * y[j] * (aj - newaj)
                    dual_coef[i] = newai

                    b1 = (b - Ei - y[i] * (newai - ai) * K[i, i] - y[j] * (newaj - aj) * K[i, j])
                    b2 = (b - Ej - y[i] * (newai - ai) * K[i, j] - y[j] * (newaj - aj) * K[j, j])
                    b = 0.5 * (b1 + b2)

                    alphas_changed += 1

            it += 1
            if alphas_changed == 0:
                passes += 1
            else:
                passes = 0

        self.w = np.dot(dual_coef * y, X)
        self.b = b

        if verbose:
            print(f"Training Complete: Iterations = {it} W = {self.w} b = {self.b}")

        return self

    def predict(self, X):
        result = np.dot(self.w, X.T) + self.b
        result[result < 0] = -1
        result[result >= 0] = 1
        return result


if __name__ == '__main__':
    import pandas as pd
    import pylab as plt
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    df = pd.read_csv('train.csv', index_col=0)

    df = df[(df['type'] == 'Ghoul') | (df['type'] == 'Goblin')]
    df = pd.get_dummies(df, columns=['color'])
    df['type'] = df['type'].map({'Ghoul':-1, 'Goblin':+1})

    print(df[:10])

    features = list(df.columns)
    features.remove('type')
    X = df[features].values
    y = df['type'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = SupportVectorMachine(C=1,n_passes=100).fit(X_train, y_train)
    print(accuracy_score(model.predict(X_test), y_test))

    from sklearn.svm import SVC

    model = SVC().fit(X, y)
    print(accuracy_score(model.predict(X_test), y_test))

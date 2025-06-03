import numpy as np
import pandas as pd
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import seaborn as sns
from itertools import product
import sys as sys
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR as skSVR


class Polynomial:
    def __init__(self, M):
        self.M = M
    

    def get_name(self):
        return 'Polynomial'

    def get_param(self):
        return self.M

    def __call__(self, x1, x2):
        return (1 + np.dot(x1, x2.T))**self.M

class RBF:
    def __init__(self, sigma):
        self.sigma = sigma
    
    def get_name(self):
        return 'RBF'

    def get_param(self):
        return self.sigma

    def __call__(self, x1, x2):
        if np.ndim(x2) == 1:
            x1, x2 = x2, x1

        x1_2d = np.atleast_2d(x1)
        x2_2d = np.atleast_2d(x2)

        dist = np.sum(x1_2d**2, axis=1)[:, np.newaxis] - 2 * np.dot(x1_2d, x2_2d.T) + np.sum(x2_2d**2, axis=1)

        kernel_matrix = np.exp(-dist / (2 * self.sigma**2))
        
        if len(kernel_matrix) == 1:
            if len(kernel_matrix[0]) == 1:
                return kernel_matrix[0][0]

            return kernel_matrix[0]
        
        return kernel_matrix

class KernelizedRidgeRegression:
    def __init__(self, kernel, lambda_) -> None:
        self.kernel = kernel
        self.lambda_ = lambda_
    
    def get_name(self):
        return 'Ridge'

    def set_params(self, **params):
        if 'lambda_' in params:
            self.lambda_ = params['lambda_']
        
        if 'kernel' in params:
            self.kernel = params['kernel']
            if self.kernel.get_param() == 5:
                print()

    def fit(self, X, y):
        K = self.kernel(X, X)
        K_reg = K + self.lambda_ * np.eye(len(K))
        alpha = np.linalg.solve(K_reg, y)

        return KernelizedRidgeRegressionModel(alpha, self.kernel, X)


class KernelizedRidgeRegressionModel:
    def __init__(self, alpha, kern, X) -> None:
        self.X = X
        self.alpha = alpha
        self.kern = kern
    
    def predict(self, x):
        K = self.kern(x, self.X)
        return np.dot(K, self.alpha)

class SVR:
    def __init__(self, kernel, lambda_, epsilon):
        self.kernel = kernel
        self.C = 1/lambda_
        self.epsilon = epsilon
    

    def get_name(self):
        return 'SVR'

    def set_params(self, **params):
        if 'lambda_' in params:
            self.C = 1 / params['lambda_']
        
        if 'epsilon' in params:
            self.epsilon = params['epsilon']
        
        if 'kernel' in params:
            self.kernel = params['kernel']
    
    def getQ(self, y):
        noStarCoef = self.epsilon - y
        starCoef = self.epsilon + y

        lenY = len(y)
        q = []
        for i in range(lenY):
            q.append(noStarCoef[i])
            q.append(starCoef[i])
        
        return matrix(q)
    
    def getA(self, y):
        A = []
        lenY = len(y)

        for _ in range(lenY):
            A.append(1.0)
            A.append(-1.0)
        
        return matrix(A, (1, 2*lenY))

    def getGandH(self, y):
        lenY = len(y)

        I = np.eye(2*lenY) * -1
        I1 = np.eye(2*lenY)

        G = np.vstack((I, I1))

        h = np.zeros(4*lenY)
        h[2*lenY:] = self.C

        return matrix(G), matrix(h)
    
    def fit(self, X, y):
        q = self.getQ(y)
        A = self.getA(y)
        b = matrix(0.0)
        G, h = self.getGandH(y)
        outerProduct = self.kernel(X, X)

        kronMat = [[1, -1], [-1, 1]]
        P = np.kron(outerProduct, kronMat)
        P = matrix(P)

        sol = solvers.qp(P, q, G, h, A, b)

        alpha_optim = sol['x']
        b = sol['y'][0]

        return SVRModel(alpha_optim, X, b, self.kernel)


class SVRModel:
    def __init__(self, alpha, X, b, kern):

        self.w = np.zeros(len(X))

        for i in range(0, 2 * len(X), 2):
            idx = i // 2
            self.w[idx] = alpha[i] - alpha[i + 1]

        self.alpha = np.array(alpha).reshape(-1, 2)
        self.X = X
        self.b = b
        self.kernel = kern
    
    def get_alpha(self):
        return self.alpha
    
    def get_b(self):
        return self.b

    def predict(self, X):
        res = self.kernel(self.X, X)
        res = np.dot(self.w, res) + self.b
        return res


def train_test_split(X, y, shuffle=False):
    if shuffle:
        indices = np.random.permutation(len(X))

        X = X[indices]
        y = y[indices]

    split_index = int(len(X) * 0.8)  # 80% of the data for training
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    return X_train, X_test, y_train, y_test

def RMSE(y_pred, y_test):
    return np.sqrt(np.mean((y_pred - y_test)**2))

def MSE(y_pred, y_test):
    return np.mean((y_pred - y_test)**2)

def plot_hyperplane(X_train, y_train, X_test, y_test, svrPolynomial = None, kRRPolynomial = None, svrRBF = None, kRRrbf = None):
    # Plot the hyperplane
    plt.figure(figsize=(11, 6))
    plt.scatter(X_train, y_train, color='b', label='Training data')
    plt.scatter(X_test, y_test, color='m', label='Test data')

    X_plot = np.linspace(min(X_train), max(X_train), 1000)[:]

    for colLabel, model in  zip(['r SVR(Polynomial)', 'g SVR(RBF)'], [svrPolynomial, svrRBF]):
        if model is None:
            continue

        col = colLabel.split(' ')[0]
        label = colLabel.split(' ')[1]

        if col == 'r':
            old_support_vector_indices = np.abs(model.w) > 1e-6
        support_vector_indices = np.abs(model.w) > 1e-6
        support_vectors = X_train[support_vector_indices]
        y_values = y_train[support_vector_indices]
        plt.scatter(support_vectors, y_values, color=col, label='Support Vectors ' + label)

        if col == 'g':
            support_vector_indices = old_support_vector_indices & support_vector_indices
            support_vectors = X_train[support_vector_indices]
            y_values = y_train[support_vector_indices]
            plt.scatter(support_vectors, y_values, color='black', label='Support Vectors for both SVRs')


    for colLabel, model in zip(['r SVR(Polynomial)', 'y Ridge(Polynomial)', 'g SVR(RBF)', 'c Ridge(RBF)'],
                           [svrPolynomial, kRRPolynomial, svrRBF, kRRrbf]):
        if model is None:
            continue

        col = colLabel.split(' ')[0]
        label = colLabel.split(' ')[1]
        y_plot = model.predict(X_plot)
        plt.plot(X_plot, y_plot, color=col, label=label + ' Hyperplane')

    plt.legend(fontsize='small')
    #plt.savefig('sine.pdf')
    plt.show()


def sine():
    df = pd.read_csv("sine.csv")
    X = df['x'].values
    y = df['y'].values


    if len(X.shape) == 1:
        X = X[:, None]

    svrRBF = SVR(RBF(0.4), 1e-4, 0.7)
    kRRrbf = KernelizedRidgeRegression(RBF(1.5), 1e-8)
    svr = SVR(Polynomial(9), 1e-4, 0.7)
    kRR = KernelizedRidgeRegression(Polynomial(9), 1e-5)

    while True:
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)
        mean = np.mean(X_train)
        std = np.std(X_train)
        X_train = (X_train - mean) / std

        svrModel = svr.fit(X_train, y_train)
        svrRBFModel = svrRBF.fit(X_train, y_train)

        X_test = (X_test - mean) / std
        y_pred = svrModel.predict(X_test)

        if np.sum(svrRBFModel.w > 1e-6) > 7 and np.sum(svrRBFModel.w > 1e-6) < 12 and np.sum(svrModel.w > 1e-6) < 12:
            break

    
    mKRR = kRR.fit(X_train, y_train)
    kRRrbf = kRRrbf.fit(X_train, y_train)
    print(np.sum(np.abs(svrRBFModel.w) > 1e-6))
    print(np.sum(np.abs(svrModel.w) > 1e-6))

    error = RMSE(y_pred, y_test)
    print(f"Root Mean Squared Error: {error}")
    
    error = RMSE(mKRR.predict(X_test), y_test)
    print(f"Root Mean Squared Error: {error}")

    plot_hyperplane(X_train, y_train, X_test, y_test, svrModel, mKRR, svrRBFModel, kRRrbf)


def GridSearchCV(model, param_grid, X, y, cv=5):
    best_mse = float('inf')
    best_params = None

    all_params = [dict(zip(param_grid.keys(), values)) for values in product(*param_grid.values())]

    kf = KFold(n_splits=cv)

    for params in all_params:
        model.set_params(**params)

        mse_values = []
        for train_index, val_index in kf.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            m = model.fit(X_train, y_train)
            y_pred = m.predict(X_val)
            mse = MSE(y_pred, y_val)
            mse_values.append(mse)

        avg_mse = np.mean(mse_values)

        if avg_mse < best_mse:
            best_mse = avg_mse
            best_params = params

    return best_params, best_mse


def plot_mse_vs_param(X_train, y_train, X_test, y_test):
    kernel_range = range(1, 11)
    param_grid = {'lambda_': [0.001, 0.01, 0.1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 5e5, 7e5]}


    upper_limit = 350
    zero_threshhold = 1e-6
    plt.figure(figsize=(7, 6))
    polynomial = [SVR(Polynomial(1), 1, 11), KernelizedRidgeRegression(Polynomial(9), 1)]
    rbf = [SVR(RBF(1), 1, 8), KernelizedRidgeRegression(RBF(1), 1)]
    for model in polynomial: 
        if isinstance(model.kernel, Polynomial):
            kernel_range = range(1, 11)
            kernel_range = list(map(lambda x: Polynomial(x), kernel_range))
        else:
            kernel_range = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9, 10]
            kernel_range = list(map(lambda x: RBF(x), kernel_range))

        for gridSearch in [False, True]:
            mse_values = []
            sv_counts = []
            label = '_grid_Search' if gridSearch else ''
            label = model.get_name() + '_' + kernel_range[0].get_name() + label
            for kernel in kernel_range:
                model.set_params(kernel=kernel)
                if gridSearch:
                    best_params, best_mse = GridSearchCV(model, param_grid, X_train, y_train)
                    print(best_params, "M", kernel.get_param(), "MSE", best_mse)
                    model.set_params(**best_params)

                m = model.fit(X_train, y_train)
                y_pred = m.predict(X_test)
                mse = MSE(y_pred, y_test)
                print("MSE", mse)

                if isinstance(model, SVR):
                    support_vector_count = np.sum(m.w > zero_threshhold)

                mse_values.append(mse)
                sv_counts.append(support_vector_count)

            params = list(map(lambda x: x.get_param(), kernel_range))
            data = pd.DataFrame({'params': params, 'mse_values': mse_values, 'sv_counts': sv_counts})
            plot = sns.lineplot(x='params', y='mse_values', data=data, marker='o', label=label)

            plot.set(xlabel='Kernel Parameter Sigma', ylabel='Mean Squared Error')
            
            if isinstance(model, SVR):
                for i in range(len(params)):
                    offset = 0
                    if not gridSearch and (i > -1 and i < 11):
                        offset = 0.1

                    y_coord = np.min((mse_values[i] + mse_values[i] * offset, upper_limit - 100))
                    plt.text(params[i], y_coord, f'{sv_counts[i]}', ha='center', va='bottom', 
                            fontweight='bold')


    plt.ylim(0, upper_limit)  # Set the y-limits
    plt.legend()
    plt.savefig('housing2r_Poli2.pdf')
    plt.show()

class Linear:
    """An example of a kernel."""

    def __init__(self):
        # here a kernel could set its parameters
        pass

    def __call__(self, A, B):
        """Can be called with vectors or matrices, see the
        comment for test_kernel"""
        return A.dot(B.T)

def compareSKLearn():

    df = pd.read_csv("sine.csv")
    X = df['x'].values
    y = df['y'].values


    if len(X.shape) == 1:
        X = X[:, None]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)

    my_krr = KernelizedRidgeRegression(kernel=Linear(), lambda_=1.0)
    my_krr = my_krr.fit(X_train, y_train)

    my_svr = SVR(Linear(), lambda_=1.0, epsilon=0.1)
    my_svr = my_svr.fit(X_train, y_train)

    lib_krr = KernelRidge(kernel='linear', alpha=1.0)
    lib_krr.fit(X_train, y_train)

    lib_svr = skSVR(kernel='linear', C=1.0, epsilon=0.1)
    lib_svr.fit(X_train, y_train)

    my_krr_preds = my_krr.predict(X_test)
    lib_krr_preds = lib_krr.predict(X_test)

    my_svr_preds = my_svr.predict(X_test)
    lib_svr_preds = lib_svr.predict(X_test)

    print("KRR MSE:", MSE(my_krr_preds, lib_krr_preds))
    print("SVR MSE:", MSE(my_svr_preds, lib_svr_preds))

if __name__ == "__main__":
    df = pd.read_csv("housing2r.csv")
    X = df.drop('y', axis=1).values
    y = df['y'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    means = np.mean(X_train, axis=0)
    stds = np.std(X_train, axis=0)
    X_train = (X_train - means) / stds
    X_test = (X_test - means) / stds

    #plot_mse_vs_param(X_train, y_train, X_test, y_test)
    sine()
    #compareSKLearn()
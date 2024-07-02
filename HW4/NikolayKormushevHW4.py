import numpy as np
import csv
from scipy import stats
from sklearn.linear_model import Ridge
import time
import pandas as pd
from scipy.optimize import fmin_l_bfgs_b
from sklearn.model_selection import KFold
from tqdm import tqdm
from itertools import product
from matplotlib import pyplot as plt

import unittest
from scipy.optimize import check_grad
import numpy as np

def xavier_initialization(size_in, size_out):
    std_dev = np.sqrt(2.0 / (size_in + size_out))
    return np.random.normal(0, std_dev, (size_in, size_out)), np.zeros(size_out)

def he_initialization(size_in, size_out):
    std_dev = np.sqrt(2.0 / size_in)
    return np.random.normal(0, std_dev, (size_in, size_out)), np.zeros(size_out)

# This is xavier but the bias is put to 1/n to make sure that
# each class in the last layer has an equal chance of being selected
def last_layer_initialization_classification(size_in, size_out):
    std_dev = np.sqrt(2.0 / (size_in + size_out))
    return np.random.normal(0, std_dev, (size_in, size_out)), np.ones(size_out) * 1/size_out


def random_initialization(size_in, size_out):
    return np.random.randn(size_in, size_out), np.zeros(size_out) + 0.01

def random_initialization_small(size_in, size_out):
    return np.random.randn(size_in, size_out) * 0.1, np.zeros(size_out) + 0.01


def encode_y(Y, y_map):
    category_count = len(y_map)
    encoded_y = np.zeros((len(Y), category_count))
    for idx, y in enumerate(Y):
        encoded_y[idx] = y_map[y]
    
    return encoded_y

def one_hot_encode(Y):
    unique_ys = np.unique(Y)
    category_count = len(unique_ys)

    y_map = {}
    for idx, uy in enumerate(unique_ys):
        y_map[uy] = np.zeros(category_count)
        y_map[uy][idx] = 1


    encoded_y = encode_y(Y, y_map)

    return encoded_y, y_map

class Softmax:
    def __init__(self) -> None:
        pass

    def forward(self, x):
        self.x = x
        norm = x - np.max(x, axis=1, keepdims=True)
        exps = np.exp(norm)
        return exps / np.sum(exps, axis=1, keepdims=True)

def l2_regularization(w, _lambda):
    return _lambda * np.sum(w ** 2)

class LogLoss:
    def __init__(self, _lambda = 0, reg = l2_regularization) -> None:
        self._lambda = _lambda
        self.softmax = Softmax()
        self.regularization = reg
    
    def logLoss(self, y, pred, w):
        # Offset to avoid log(0)
        offset = 1e-10
        pred = (pred + offset)/np.sum(pred + offset, axis=1, keepdims=True)

        self.w = w
        return -np.sum(y * np.log(pred)) + self.regularization(w, self._lambda)
    
    def loss(self, y, x, w):
        pred = self.softmax.forward(x)

        return self.logLoss(y, pred, w)

    
    def backward(self, y, x):
        pred = self.softmax.forward(x)
        return pred - y

class ReLU:
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, dout):
        return np.multiply(dout, (self.x >= 0)), []

class MSE:
    def __init__(self, _lambda = 0, reg = l2_regularization) -> None:
        self._lambda = _lambda
        self.reg = reg

    def loss(self, y, pred, w):
        return np.mean((y - pred) ** 2) + self.reg(w, self._lambda)

    def backward(self, y, pred):
        return 2 * (pred - y) / pred.shape[0]
    

def l2_derivative(_lambda, A):
    return 2 * _lambda * A
    
class FullyConnectedLayer:
    def __init__(self, n, m, _lambda = 0, initialization = he_initialization, dreg = l2_derivative):
        self.n = n
        self.m = m
        self._lambda = _lambda
        self.A, self.b = initialization(n, m)
        self.dreg = dreg

    def updateValuesFromFlatArray(self, parameters, startIndex):
        matrixEnd = startIndex + self.n * self.m
        A = parameters[startIndex:matrixEnd].reshape(self.n, self.m)
        b = parameters[matrixEnd:matrixEnd + self.m]
        self.A = A
        self.b = b
        return matrixEnd + self.m, parameters[startIndex:matrixEnd]

    def getInitialValues(self):
        return np.concatenate((self.A.flatten(), self.b))

    def forward(self, x):
        self.x = x
        return x @ self.A + self.b 

    def backward(self, dout):
        db = np.sum(dout, axis=0)
        dx = dout @ self.A.T
        dA = self.x.T @ dout + self.dreg(self._lambda, self.A)

        return dx, np.concatenate((dA.flatten(), db))

def optimizer(parameters, *args):
    X = args[0]
    y = args[1]
    layers = args[2]
    loss = args[3]

    x = X
    paramCounter = 0
    w = []
    for layer in layers:
        if isinstance(layer, FullyConnectedLayer):
            paramCounter, w_curr = layer.updateValuesFromFlatArray(parameters, paramCounter)
            w = np.concatenate((w, w_curr))
        x = layer.forward(x)

    lossVal = loss.loss(y, x, w)
    dout = loss.backward(y, x)

    grads = []
    for layer in reversed(layers):
        dout, grad = layer.backward(dout)
        grads = np.concatenate((grad, grads))
    
    return lossVal, grads

class ANN:
    def __init__(self, units, lambda_):
        self.units = units
        self.lambda_ = lambda_

        self.layers = []

    def getInitialValues(self):
        return np.concatenate([layer.getInitialValues() for layer in self.layers 
                               if isinstance(layer, FullyConnectedLayer)])
    
    def initLayers(self, input_size, output_size = 1, last_layer_init = xavier_initialization):
        if len(self.units) != 0:

            self.layers = np.concatenate(([FullyConnectedLayer(input_size, self.units[0], self.lambda_, initialization=random_initialization_small),],
                                         [ReLU()]))

            for i in range(len(self.units) - 1):
                self.layers = np.append(self.layers, FullyConnectedLayer(self.units[i], self.units[i+1], self.lambda_))
                self.layers = np.append(self.layers, ReLU())
            
            self.layers = np.append(self.layers, [FullyConnectedLayer(self.units[-1], output_size, self.lambda_, last_layer_init)])
        else:
            self.layers = [FullyConnectedLayer(input_size, output_size, self.lambda_, initialization=last_layer_init)]
    

class ANNRegression(ANN):
    def __init__(self, units, lambda_):
        super().__init__(units, lambda_)
        self.loss = MSE()

    def fit(self, X, y):
        input_size = X.shape[1]
        self.initLayers(input_size)

        y = y.reshape(-1, 1)
        init = self.getInitialValues()
        opt, _, _ = fmin_l_bfgs_b(optimizer, x0=init, args=(X, y, self.layers, self.loss), maxiter=100)

        return ANNModel(self.layers, opt)

class ANNClassification(ANN):
    def __init__(self, units, lambda_):
        super().__init__(units, lambda_)

        self.loss = LogLoss(self.lambda_)
        self.category_count = 0
    
    def get_params(self, deep=True):
        return {"units": self.units, "lambda_": self.lambda_}
    
    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)
        return self

    def score(self, X, y):
        y_pred = self.child_model.predict(X)

        y = encode_y(y, self.encoding_y)
        accuracy = LogLoss().logLoss(y, y_pred, 0) / len(y)

        return -accuracy
    
    def fit(self, X, y):
        self.category_count = len(np.unique(y))
        encoded_y, self.encoding_y = one_hot_encode(y)
        input_size = X.shape[1]

        self.initLayers(input_size, self.category_count, last_layer_init=last_layer_initialization_classification)

        init = self.getInitialValues()
        opt, _, _ = fmin_l_bfgs_b(optimizer, x0=init, args=(X, encoded_y, self.layers, self.loss), pgtol=0.001, maxiter=100)

        # I add the softmax layer at the end cause initially it was part of the 
        # loss function. Better solution is to separate them or to rework this
        self.child_model = ANNModel(np.concatenate([self.layers, [Softmax()]]), opt)
        self.child_model.encoding_y = self.encoding_y
        return self.child_model

    
# Class added so we can not call fit more than once
# because that would break the model because we do not know
# input and output dimensions
class ANNModel:
    def __init__(self, layers, opt) -> None:
        paramCounter = 0
        for layer in layers:
            if isinstance(layer, FullyConnectedLayer):
                paramCounter, _ = layer.updateValuesFromFlatArray(opt, paramCounter)

        self.layers = layers

    def weights(self):
        weights = []
        for layer in self.layers:
            if isinstance(layer, FullyConnectedLayer):
                 weights.append(np.vstack((layer.A, layer.b)))

        return weights

    def predict(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X.squeeze()


def trainTestSplit(X, y):
    indices = np.random.permutation(X.shape[0])

    train_size = int(0.8 * X.shape[0])  
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test

def mean_cross_entropy(y_pred, y):
    y_pred = y_pred + 1e-10
    y_pred /= np.sum(y_pred, axis=1, keepdims=True)
    return -np.sum(y * np.log(y_pred)) / len(y)

def standardize(X):
    means = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X = (X - means) / std
    return X, means, std

def kFoldSplit(X, y, trainer, encode = False, splits_count = 5, toStandardize = True):
    kfoldSplits = KFold(n_splits=splits_count, shuffle=True, random_state=42)

    errors = []
    for train_index, test_index in tqdm(kfoldSplits.split(X)):
        X_train, X_val = X[train_index], X[test_index]
        y_train, y_val = y[train_index], y[test_index]

        if toStandardize:
            X_train, mean, std = standardize(X_train)
            X_val = (X_val - mean) / std

        model = trainer.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        if encode:
            # Use old log_loss cause it does one_hot_encoding of the test set
            #y_val = encode_y(y_val, model.encoding_y)
            y_val, _ = one_hot_encode(y_val)
            error = mean_cross_entropy(y_pred, y_val) #loss.logLoss(y_val, y_pred, 0) / len(y_val)
        else:
            error = MSE().loss(y_val, y_pred, 0)

        errors.append(error)
    
    return errors


def GridSearchCV_kFoldSplit(X, y, model, param_grid, k):
    # Create a list of all combinations of parameters
    all_params = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]

    best_mean_err = float('inf')
    best_params = None

    with open('best_params.txt', 'a') as f:
        # Iterate over all combinations of parameters
        for params in tqdm(all_params):
            # Set the parameters of the model
            model.set_params(**params)

            # Perform k-fold cross-validation
            errs = kFoldSplit(X, y, model, encode=True, splits_count=k, toStandardize=False)

            # Compute the mean error
            mean_err = np.mean(errs)

            res = f'Params: {params}, {mean_err}\n'
            f.write(res)
            print(res)

            # If this is the best mean error so far, update the best mean error and best parameters
            if mean_err < best_mean_err:
                best_mean_err = mean_err
                best_params = params

    return best_params, best_mean_err

def measure_performance():
    df = pd.read_csv('train.csv')
    X = df.drop(columns=['target', 'id']).values
    y = df['target'].values

    X, means, std = standardize(X)
    y = pd.factorize(y)[0]
    trainer = ANNClassification([160], lambda_=18)


    with open('execution_times.txt', 'a') as f:
        for i in tqdm(range(10)):
            start_time = time.time()
            trainer.fit(X, y)
            end_time = time.time()
            execution_time = end_time - start_time
            f.write(f"{execution_time} \n")



def create_final_predictions():
    df = pd.read_csv('train.csv')
    X = df.drop(columns=['target', 'id']).values
    y = df['target'].values

    X, means, std = standardize(X)
    y = pd.factorize(y)[0]
    trainer = ANNClassification([160], lambda_=18)

    best_params, best_err = GridSearchCV_kFoldSplit(X, y, trainer,
                                 {'units': [[160]],
                                 'lambda_': [18]}, 5)

    # Print best parameters
    print("Best parameters: ", best_params)
    print("Best score: ", best_err)


    logisticFitter = MultinomialLogReg()

    logErrs = kFoldSplit(X, y, logisticFitter, encode=True, splits_count=7, toStandardize=False)
    print("LOG: Mean error: ", np.mean(logErrs), "Std error: ", np.std(logErrs)/np.sqrt(len(logErrs)))

    model = trainer.fit(X, y)
    test = pd.read_csv('test.csv')
    X_test = test.drop(columns='id').values
    X_test = (X_test - means) / std
    preds = model.predict(X_test)

    with open("final.txt", "w") as f:
        f.write('id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9\n')

        for idx, pred in enumerate(preds):
            pred = [str(p) for p in pred]
            f.write(f'{idx + 1},{",".join(pred)}\n')


def create_scatter_plot(file_path):
    params = []
    scores = []

    # Read data from file
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('}')
            tmp = parts[0] + '}'
            tmp = tmp[8:]
            params.append(eval(tmp.strip()))
            scores.append(float(parts[1].split(',')[1]))

    # Prepare data for plotting
    x_values = [param['lambda_'] for param in params]
    y_values = [param['units'][0] for param in params]
    marker_sizes = [score * 100 for score in scores]  # Adjust size based on score

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x_values, y_values, s=marker_sizes, c=scores, cmap='viridis', alpha=0.7)
    plt.colorbar(label='Scores')
    plt.xlabel('Lambda')
    plt.ylabel('Units')
    plt.title('Scores for Different Parameter Configurations')
    plt.grid(True)
    plt.tight_layout()

    plt.savefig('scatter_plot.pdf', format='pdf')

    # Show plot
    plt.show()


def encode_y(Y, y_map):
    category_count = len(y_map)
    encoded_y = np.zeros((len(Y), category_count))
    for idx, y in enumerate(Y):
        encoded_y[idx] = y_map[y]
    
    return encoded_y

def one_hot_encodeNew(Y, category_count = None):
    unique_ys = np.unique(Y)
    if category_count is None:
        category_count = len(unique_ys)

    y_map = {}
    for idx, uy in enumerate(unique_ys):
        y_map[uy] = np.zeros(category_count)
        y_map[uy][idx] = 1

    encoded_y = encode_y(Y, y_map)

    return encoded_y, y_map

def softmax(x):
    # avoid nan
    x = x - np.max(x, axis = 1, keepdims=True)
    rows_sum = np.sum(np.exp(x), axis=1, keepdims=True)
    return np.exp(x) / rows_sum

def multinomial_predictor(b_shape, y):
    def predictor(parameters, x):
        b = parameters.reshape(b_shape)
        model = MultinomialLogModel(b)

        one_extend = np.ones(x.shape[0])
        x_ext = np.column_stack((x, one_extend.reshape(-1, 1)))
        y_pred = model.predict(x)
        y_act, _ = one_hot_encodeNew(y)
        grad = np.dot(x_ext.T, (y_pred[:, :-1] - y_act[:,:-1]))/len(y)

        return y_pred, grad.T.flatten()
    return predictor

def optimizer_mult(parameters, *args):
    x = args[0]
    y = args[1]
    predictor = args[2]

    y_model, grad = predictor(parameters, x)

    error = log_loss(y_model, y)

    if grad is not None:
        return error, grad

    return error

def log_loss(y_pred, y, cats = None):
    if cats is None:
        cats = len(np.unique(y))
    encoded_y, _ = one_hot_encodeNew(y, cats)
    # avoid division by zero so offset and normalize
    offset = 10e-10
    y_pred = (y_pred + offset)/np.sum(y_pred + offset, axis=1, keepdims=True)
    return -np.sum(encoded_y * np.log(y_pred)) / len(y)

class MultinomialLogReg:
    def fit(self, X, y):
        cats = np.unique(y)
        cat_count = len(cats)
        
        _, endocing_y =  one_hot_encodeNew(y)

        # + 1 for the intercept. -1 cause of the reference class
        b_shape = (cat_count - 1, X.shape[1] + 1)
        initial_values = np.ones(b_shape).flatten()
        predictor = multinomial_predictor(b_shape, y)

        b_opt, _, _ = fmin_l_bfgs_b(optimizer_mult, x0=initial_values, args=(X, y, predictor), approx_grad=False)
        
        model = MultinomialLogModel(b_opt.reshape(b_shape), cats)
        model.encoding_y = endocing_y 
        return model

class MultinomialLogModel:
    def __init__(self, b, decode_map = None):
        self.b = b
        self.decode_map = decode_map

    def predict(self, x):
        one_extend = np.ones(x.shape[0])
        x_ext = np.column_stack((x, one_extend.reshape(-1, 1)))
        linearPreds = np.matmul(x_ext, self.b.T)
        reference_class = np.zeros(x.shape[0])

        lin_preds_ext = np.column_stack((linearPreds, reference_class.reshape(-1, 1)))
        y_pred = softmax(lin_preds_ext)

        return y_pred
    
    def decode(self, y):
        if self.decode_map is None:
            return y

        return [self.decode_map[np.argmax(yi)] for yi in y]


class TestMSE(unittest.TestCase):
    def test_backward(self):
        mse = MSE()

        y = np.random.rand(10, 1)
        pred = np.random.rand(10, 1)

        def func(pred):
            pred = pred.reshape(10, 1)
            return mse.loss(y, pred, np.array([0]))

        def grad(pred):
            pred = pred.reshape(10, 1)
            return mse.backward(y, pred).flatten()

        error = check_grad(func, grad, pred.flatten())
        self.assertAlmostEqual(error, 0, places=5)


class TestReLU(unittest.TestCase):
    def test_backward(self):
        relu = ReLU()

        x = np.random.rand(10, 5)

        def func(x):
            x = x.reshape(10, 5)
            return np.sum(relu.forward(x))

        def grad(x):
            x = x.reshape(10, 5)
            relu.x = x
            dout = np.ones_like(x)
            return relu.backward(dout)[0].flatten()

        error = check_grad(func, grad, x.flatten())
        self.assertAlmostEqual(error, 0, places=5)

class TestLogLoss(unittest.TestCase):
    def test_backward(self):
        logloss = LogLoss(_lambda=0)

        x = np.random.rand(5, 5)
        y = np.eye(5, 5)
        w = np.random.rand(5, 5)
        xShape = x.shape

        def func(x):
            x = x.reshape(xShape)
            return logloss.loss(y, x, w)

        def grad(x):
            x = x.reshape(xShape)
            return logloss.backward(y, x).flatten()

        error = check_grad(func, grad, x.flatten())
        self.assertAlmostEqual(error, 0, places=5)

class TestFullyConnectedLayer(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[0, 0],
                           [0, 1],
                           [1, 0],
                           [1, 1]])
        self.y = np.array([0, 1, 2, 3])
        self.hard_y = np.array([0, 1, 1, 0])

    def test_gradients_input(self):
        fcl = FullyConnectedLayer(n=self.X.shape[1], m=1)
        out = fcl.forward(self.X)
        dout = np.ones(out.shape)

        def f(params):
            params = params.reshape(self.X.shape)
            res = fcl.forward(params)
            # I sum to return only one variable as required by check_grad
            return res.sum()

        def grad(params):
            fcl.x = params.reshape(self.X.shape)
            grads = fcl.backward(dout)[0].flatten()
            return grads

        error = check_grad(f, grad, self.X.flatten())

        self.assertLess(error, 1e-5)

    def test_gradients_weights_and_bias(self):
        fcl = FullyConnectedLayer(n=self.X.shape[1], m=1)
        out = fcl.forward(self.X)
        dout = np.ones(out.shape)

        def f(params):
            fcl.updateValuesFromFlatArray(params, 0)
            res = fcl.forward(self.X)
            # I sum to return only one variable as required by check_grad
            return res.sum()

        def grad(params):
            fcl.updateValuesFromFlatArray(params, 0)
            grads = fcl.backward(dout)[1]
            return grads

        params_initial = fcl.getInitialValues()
        error = check_grad(f, grad, params_initial)

        self.assertLess(error, 1e-5)


class TestOptimizer(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[0, 0],
                           [0, 1],
                           [1, 0],
                           [1, 1]])
        self.y = np.array([0, 1, 2, 3])
        self.hard_y = np.array([0, 1, 1, 0])

        _lambda = 0  
        self._lambda = _lambda
        self.layers = [FullyConnectedLayer(n=self.X.shape[1], m=5, _lambda=_lambda, initialization=random_initialization),
                  ReLU(),
                  FullyConnectedLayer(n=5, m=6, _lambda=_lambda, initialization=he_initialization),
                  ReLU(),
                  FullyConnectedLayer(n=6, m=6, _lambda=_lambda, initialization=he_initialization),
                  ReLU()]
    
    def backpropagation_helper(self, target, layers, loss_func):
        def f(params):
            loss, _ = optimizer(params, self.X, target, layers, loss_func(self._lambda))
            return loss

        def grad(params):
            _, grads = optimizer(params, self.X, target, layers, loss_func(self._lambda))
            return grads

        params_initial = np.concatenate([l.getInitialValues() for l in layers if isinstance(l, FullyConnectedLayer)])
        error = check_grad(f, grad, params_initial)

        self.assertLess(error, 1e-5)

    def test_backpropagation_classification(self):
        layers = np.append(self.layers,
                FullyConnectedLayer(n=6, m=4, _lambda=self._lambda,
                                 initialization=last_layer_initialization_classification))

        enc_y, _ = one_hot_encode(self.y)
        self.backpropagation_helper(enc_y, layers, LogLoss)

    def test_backpropagation_regression(self):

        layers = np.append(self.layers, FullyConnectedLayer(n=6, m=1, _lambda=self._lambda))
        transposed_y = self.y.reshape(-1, 1)
        
        self.backpropagation_helper(transposed_y, layers, MSE)

class HW6Tests(unittest.TestCase):

    def setUp(self):
        self.X = np.array([[0, 0],
                           [0, 1],
                           [1, 0],
                           [1, 1]])
        self.y = np.array([0, 1, 2, 3])
        self.hard_y = np.array([0, 1, 1, 0])

    def test_ann_classification_no_hidden_layer(self):
        fitter = ANNClassification(units=[], lambda_=0.0001)
        m = fitter.fit(self.X, self.y)
        pred = m.predict(self.X)
        self.assertEqual(pred.shape, (4, 4))
        np.testing.assert_allclose(pred, np.identity(4), atol=0.01)

    def test_ann_classification_no_hidden_layer_hard(self):
        # aiming to solve a non linear problem without hidden layers
        fitter = ANNClassification(units=[], lambda_=0.0001)
        m = fitter.fit(self.X, self.hard_y)
        pred = m.predict(self.X)
        self.assertEqual(pred.shape, (4, 2))
        np.testing.assert_allclose(pred, 0.5, atol=0.01)

    def test_ann_classification_hidden_layer_hard(self):
        # with hidden layers we can solve an non-linear problem
        fitter = ANNClassification(units=[10], lambda_=0.0001)
        m = fitter.fit(self.X, self.hard_y)
        pred = m.predict(self.X)
        self.assertEqual(pred.shape, (4, 2))
        np.testing.assert_allclose(pred, [[1, 0], [0, 1], [0, 1], [1, 0]], atol=0.01)

    def test_ann_classification_hidden_layers_hard(self):
        # two hidden layers
        fitter = ANNClassification(units=[10, 20], lambda_=0.0001)
        m = fitter.fit(self.X, self.hard_y)
        pred = m.predict(self.X)
        self.assertEqual(pred.shape, (4, 2))
        np.testing.assert_allclose(pred, [[1, 0], [0, 1], [0, 1], [1, 0]], atol=0.01)

    def test_ann_regression_no_hidden_layer(self):
        fitter = ANNRegression(units=[], lambda_=0.0001)
        m = fitter.fit(self.X, self.y)
        pred = m.predict(self.X)
        self.assertEqual(pred.shape, (4,))
        np.testing.assert_allclose(pred, self.y, atol=0.01)

    def test_ann_regression_no_hidden_layer_hard(self):
        # aiming to solve a non linear problem without hidden layers
        fitter = ANNRegression(units=[], lambda_=0.0001)
        m = fitter.fit(self.X, self.hard_y)
        pred = m.predict(self.X)
        self.assertEqual(pred.shape, (4,))
        np.testing.assert_allclose(pred, 0.5, atol=0.01)

    def test_ann_regression_hidden_layer_hard(self):
        # one hidden layer
        fitter = ANNRegression(units=[10], lambda_=0.0001)
        m = fitter.fit(self.X, self.hard_y)
        pred = m.predict(self.X)
        self.assertEqual(pred.shape, (4,))
        np.testing.assert_allclose(pred, self.hard_y, atol=0.01)

    def test_ann_regression_hidden_layer_hard(self):
        # two hidden layers
        fitter = ANNRegression(units=[13, 6], lambda_=0.0001)
        m = fitter.fit(self.X, self.hard_y)
        pred = m.predict(self.X)
        self.assertEqual(pred.shape, (4,))
        np.testing.assert_allclose(pred, self.hard_y, atol=0.01)

    def test_predictor_get_info(self):
        fitter = ANNRegression(units=[10, 5], lambda_=0.0001)
        m = fitter.fit(self.X, self.y)
        lw = m.weights()  # a list of weight matrices that include intercept biases

        self.assertEqual(len(lw), 3)  # two hidden layer == three weight matrices

        self.assertEqual(lw[0].shape, (3, 10))
        self.assertEqual(lw[1].shape, (11, 5))
        self.assertEqual(lw[2].shape, (6, 1))



class HousingTests(unittest.TestCase):
    def testRegression(self):
        df = pd.read_csv('housing2r.csv')
        df = df.sample(frac=1).reset_index(drop=True)
        X = df.iloc[:, :-1].values
        y = df['y'].values


        ammAvgErrs = []
        stdErr = []
        ridgeAvgErrs = []
        ridgeStdErr = []
        for _ in range(100):
            RidgeFitter = Ridge(alpha=0.001)
            ridgeErrs = kFoldSplit(X, y, RidgeFitter)
            ridgeAvgErrs.append(np.mean(ridgeErrs))
            ridgeStdErr.append(np.std(ridgeErrs)/np.sqrt(len(ridgeErrs)))

            annFitter = ANNRegression(units=[5], lambda_=0.1)

            ammErrs = kFoldSplit(X, y, annFitter)
            ammAvgErrs.append(np.mean(ammErrs))
            stdErr.append(np.std(ammErrs)/np.sqrt(len(ammErrs)))

        print ("\nRidge error: ", np.mean(ridgeAvgErrs), np.mean(ridgeStdErr))
        print ("\nRidge stderr: ", np.std(ridgeAvgErrs)/np.sqrt(len(ridgeAvgErrs)))
        print ("ANN error: ", np.mean(ammAvgErrs), np.mean(stdErr))
        print ("ANN stderr: ", np.std(ammAvgErrs)/np.sqrt(len(ammAvgErrs)))


        t_stat, p_val = stats.ttest_ind(ridgeErrs, ammAvgErrs)
        print("T-test for regression: t = {}, p = {}".format(t_stat, p_val))
        self.assertGreater(np.mean(ridgeErrs), np.mean(ammAvgErrs))


    def testClassification(self):
        df = pd.read_csv('housing3.csv')
        df = df.sample(frac=1).reset_index(drop=True)
        X = df.iloc[:, :-1].values
        y = df['Class'].values

        y = pd.factorize(y)[0]

        LogisticFitter = MultinomialLogReg()

        logErrs = kFoldSplit(X, y, LogisticFitter, encode=True)

        ammAvgErrs = []
        stdErr = []
        for _ in range(100):
            annFitter = ANNClassification(units=[5], lambda_=3)

            ammErrs = kFoldSplit(X, y, annFitter, encode=True)
            ammAvgErrs.append(np.mean(ammErrs))
            stdErr.append(np.std(ammErrs)/np.sqrt(len(ammErrs)))

        print ("\nlog error: ", np.mean(logErrs), np.std(logErrs)/np.sqrt(len(logErrs)))
        print ("ANN error: ", np.mean(ammErrs), np.std(ammErrs)/np.sqrt(len(ammErrs)))
        print ("ANN stderr: ", np.std(ammAvgErrs)/np.sqrt(len(ammAvgErrs)))


        t_stat, p_val = stats.ttest_ind(logErrs, ammErrs)

        print("T-test for regression: t = {}, p = {}".format(t_stat, p_val))
        self.assertGreater(np.mean(logErrs), np.mean(ammErrs))
    


class TestFinalSubmissions(unittest.TestCase):

    def test_format(self):
        """ Tests format of your final predictions. """

        with open("final.txt", "rt") as f:
            content = list(csv.reader(f))
            content = [l for l in content if l]

            ids = []

            for i, l in enumerate(content):
                # each line contains 10 columns
                self.assertEqual(len(l), 10)

                # first line is just a description line
                if i == 0:
                    self.assertEqual(l, ['id', 'Class_1', 'Class_2', 'Class_3', 'Class_4',
                                         'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9'])
                else:
                    ids.append(int(l[0]))  # first element is an id
                    probs = np.array([float(f) for f in l[1:]])  # the rest are probabilities
                    self.assertLessEqual(np.max(probs), 1)
                    self.assertGreaterEqual(np.min(probs), 0)

            # ids covered the whole range
            self.assertEqual(set(ids), set(range(1, 11878+1)))

    def test_function(self):
        try:
            from nn import create_final_predictions
            #create_final_predictions()
        except ImportError:
            self.fail("Function create_final_predictions does not exists.")


if __name__ == "__main__":
    unittest.main()

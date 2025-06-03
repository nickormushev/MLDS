import unittest
import numpy as np
import csv
from nn import ANNClassification, ANNRegression, FullyConnectedLayer, ReLU, LogLoss, MSE, kFoldSplit, optimizer, one_hot_encode
from nn import random_initialization, last_layer_initialization_classification, he_initialization, xavier_initialization

from nn import MultinomialLogReg

from scipy import stats
from sklearn.linear_model import Ridge

import pandas as pd

import unittest
from scipy.optimize import check_grad
import numpy as np

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

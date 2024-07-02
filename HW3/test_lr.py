import random
import unittest
from sklearn.linear_model import LogisticRegression as LR
from statsmodels.miscmodels.ordinal_model import OrderedModel
import statsmodels.api as sm
import pandas as pd

import numpy as np

from solution import MultinomialLogReg, OrdinalLogReg, MultinomialLogModel, one_hot_encode, \
    softmax, log_loss, sigmoid, OrdinalLogModel, load_basketball_data, \
    multinomial_bad_ordinal_good, MBOG_TRAIN

class TestMultinomialLogModel(unittest.TestCase):
    def test_when_given_high_bias_for_first_class_it_should_always_predict_first_class(self):
        # Added a huge bias to the first class so we predict it always
        model = MultinomialLogModel(np.array([[1, 0, 0, 1000], [0, 0, 1, 0]]))
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        y_pred = model.predict(X)
        for i in range(len(y_pred)):
            self.assertTrue(np.argmax(y_pred[i]) == 0)

        t = y_pred.sum(axis=1)
        np.testing.assert_almost_equal(y_pred.sum(axis=1), 1)

class HW2Tests(unittest.TestCase):

    def setUp(self):
        self.X = np.array([[0, 0],
                           [0, 1],
                           [1, 0],
                           [1, 1],
                           [1, 1]])
        self.y = np.array([0, 0, 1, 1, 2])
        self.train = self.X[::2], self.y[::2]
        self.test = self.X[1::2], self.y[1::2]

    def test_multinomial(self):
        l = MultinomialLogReg()
        c = l.build(self.X, self.y)
        prob = c.predict(self.test[0])
        self.assertEqual(prob.shape, (2, 3))
        self.assertTrue((prob <= 1).all())
        self.assertTrue((prob >= 0).all())
        np.testing.assert_almost_equal(prob.sum(axis=1), 1)

    def test_ordinal(self):
        l = OrdinalLogReg()
        c = l.build(self.X, self.y)
        prob = c.predict(self.test[0])
        self.assertEqual(prob.shape, (2, 3))
        self.assertTrue((prob <= 1).all())
        self.assertTrue((prob >= 0).all())
        np.testing.assert_almost_equal(prob.sum(axis=1), 1)

    def test_multinomial_bad_ordinal_good(self):
        rand = random.Random(0)
        X, y = multinomial_bad_ordinal_good(100, rand)
        self.assertEqual(len(X), 100)
        self.assertEqual(y.shape, (100,))
        rand = random.Random(1)
        X1, y1 = multinomial_bad_ordinal_good(100, rand)
        self.assertTrue((X != X1).any())
        trainX, trainY = multinomial_bad_ordinal_good(MBOG_TRAIN, random.Random(42))
        self.assertEqual(len(trainX), MBOG_TRAIN)

class TestOrdinalLogModel(unittest.TestCase):
    def test_when_predict_called_probabilities_should_sum_up_to_one_and_accurately_predict(self):
        # Given an OrdinalLogModel with a bias and a single delta
        model = OrdinalLogModel(np.array([1, 0, 0, 0]), np.array([1]))
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        probs = model.predict(X)

        for prob in probs:
            self.assertAlmostEqual(np.sum(prob), 1)

        # The highest probability class
        hpc = np.argmax([sigmoid(x) for x in [1, 4, 7]])

        np.testing.assert_array_equal(np.argmax(probs, axis=1), [hpc] * 3)

class MiscelaneousMethods(unittest.TestCase):
    def test_when_one_hot_encode_is_called_it_should_return_encoded_values(self):
        y = [1,2,3]
        res = one_hot_encode(y, len(np.unique(y)))
        np.testing.assert_array_equal(len(res), len(y))
        np.testing.assert_array_equal(res[0], [1, 0, 0])
        np.testing.assert_array_equal(res[1], [0, 1, 0])
        np.testing.assert_array_equal(res[2], [0, 0, 1])
    
    def test_when_given_large_numbers_softmax_should_not_return_nan(self):
        x = np.array([[1000000, 30, 2]])
        result = softmax(x)
        self.assertFalse(np.isnan(result).any())

    def test_when_given_multiple_samples_softmax_should_return_values_for_all(self):
        x = np.array([[1000000, 30, 2], [1000000, 30, 2]])
        result = softmax(x)
        for i in range(len(result)):
            self.assertTrue(result[i, 0] > result[i, 1] and result[i, 0] > result[i, 2])
            self.assertAlmostEqual(result[i].sum(), 1)

    def test_when_each_class_is_true_once_log_loss_should_return_sum_of_negative_logs_of_predictions(self):
        y_pred = np.array([[0.2, 0.3, 0.5], [0.2, 0.3, 0.5], [0.2, 0.3, 0.5]] )
        y = np.array([0, 1, 2])
        result = log_loss(y_pred, y)
        expected = -np.log(0.2) -np.log(0.3) - np.log(0.5)
        self.assertAlmostEqual(result, expected / len(y))

def get_split(X, y, rand_seed = 0):
    np.random.seed(rand_seed)
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    X_train = X[:int(0.8 * len(X))]
    y_train = y[:int(0.8 * len(y))]

    X_test = X[int(0.8 * len(X)):]
    y_test = y[int(0.8 * len(y)):]

    return X_train, y_train, X_test, y_test

class TestOrdinalLogModel(unittest.TestCase):
    # Very scientific checks
    def test_ordinal_log_model_vs_stats(self):
        X, y = multinomial_bad_ordinal_good(1000, random.Random(42))

        X_train, y_train, X_test, y_test = get_split(X, y)

        olr = OrdinalLogReg()
        model = olr.build(X_train, y_train)
        my_predictions = model.predict(X_test)

        X_train = pd.DataFrame(X_train, columns = ['x1', 'x2', 'x3'])
        X_test = pd.DataFrame(X_test, columns = ['x1', 'x2', 'x3'])

        y_train = pd.Categorical(y_train, categories=np.unique(y_train).sort(), ordered=True)
        ordinal_model = OrderedModel(y_train,
                        X_train,
                        method='lbfgs',
                        distr='logit')
        res = ordinal_model.fit(method='lbfgs', disp=False)
        stats_predictions = res.predict(X_test)
        cats = stats_predictions.shape[1]

        my_loss = log_loss(my_predictions, y_test, cats)
        stats_Loss = log_loss(stats_predictions.values, y_test, cats)

        self.assertAlmostEqual(stats_Loss, my_loss, 3)

    def test_multinomial_vs_sklearn(self):
        X, y= load_basketball_data("./dataset.csv")
        X = X.values

        X_train, y_train, X_test, y_test = get_split(X, y)

        olr = MultinomialLogReg()
        model = olr.build(X_train, y_train)
        my_predictions = model.predict(X_test)

        sklearn_model = LR(multi_class='multinomial', solver='lbfgs')
        sklearn_model.fit(X_train, y_train)
        sklearn_predictions = sklearn_model.predict_proba(X_test)

        my_loss = log_loss(my_predictions, y_test)
        sklearnLoss = log_loss(sklearn_predictions, y_test)

        self.assertAlmostEqual(sklearnLoss, my_loss, 3)



if __name__ == "__main__":
    unittest.main()

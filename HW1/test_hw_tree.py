import unittest
from unittest.mock import Mock, patch


import numpy as np
import random

from hw_tree import Tree, TreeNode, RandomForest, RFModel, mode, calculateGini, hw_tree_full, hw_randomforests


def random_feature(X, rand):
    return [rand.choice(list(range(X.shape[1])))]

class TestRFModel(unittest.TestCase):
    def setUp(self):
        self.rfmodel = RFModel()
        self.rfmodel.trees = np.array([Mock() for _ in range(3)]) 
        self.rfmodel.oob = [set([0]), set([1]), set([0])]
        self.rfmodel.X = np.array([[1, 2], [1, 5], [7, 2]])
        self.rfmodel.y = np.array([1, 2])

    @patch.object(RFModel, 'getSubtreeErrors')
    def test_importance(self, mock_getSubtreeErrors):
        return_values = [0.2, 0.3, 0.4, 0.5]
        mock_getSubtreeErrors.side_effect = return_values

        imps = self.rfmodel.importance()

        self.assertGreaterEqual(len(imps), 2)

        self.assertEqual(mock_getSubtreeErrors.call_count, 3)

        normalErr = return_values[0]
        for i in range(0, len(imps)):
            err = return_values[i + 1]
            eps = 0.00001 # to not divide by 0
            expected_imps = (err - normalErr)/(normalErr + eps) * 100
            self.assertEqual(imps[i], expected_imps)

    def test_getSubtreeErrors(self):
        # The first and last return values are correct. Second one is wrong
        predict_return_values = [np.array([1]), np.array([0]), np.array([1])]

        for i, tree in enumerate(self.rfmodel.trees):
            tree.predict.return_value = predict_return_values[i]

        result = self.rfmodel.getSubtreeErrors(self.rfmodel.X)

        # Check if the trees are called with the correct OOB samples
        for i, tree in enumerate(self.rfmodel.trees):
            np.testing.assert_array_equal(tree.predict.call_args[0][0],
                                self.rfmodel.X[list(self.rfmodel.oob[i]), :])

        # Only 1 of 3 is wrong. So the error is 1/3
        expectedErr = 1/3
        self.assertAlmostEqual(result, expectedErr)


    def test_predictWithTrees(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        for tree in self.rfmodel.trees:
            tree.predict.return_value = np.random.choice([0, 1], size=X.shape[0])

        predictions = self.rfmodel.predictWithTrees(X, self.rfmodel.trees)

        # Check that tree.predict was called with the correct arguments
        for tree in self.rfmodel.trees:
            tree.predict.assert_called_once_with(X)

        # Check that the mode was correctly calculated for each column
        # We go through the columns and then we create an array of the return value[colIdx] and take the mode
        expected_predictions = [mode([tree.predict.return_value[colIdx] for tree in self.rfmodel.trees])
                                 for colIdx in range(X.shape[0])]
        for i, pred in enumerate(predictions):
            self.assertEqual(pred, expected_predictions[i])

class TestRandomForest(unittest.TestCase):
    def setUp(self):
        self.rf = RandomForest(n=50)
        self.rf.rftree.build = Mock()

    def test_build(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([1, 1, 0, 0])
        self.rf.rftree.build.side_effect = (lambda X, y: TreeNode())

        self.rf.build(X, y)

        self.assertEqual(self.rf.rftree.build.call_count, 50)

        # Check that rftree.build was called with valid samples
        # In other words samples of size len(X)
        # and alle elements the tree builds are called with are from X
        for call in self.rf.rftree.build.call_args_list:
            args, kwargs = call
            bootstrap_X, bootstrap_y = args
            self.assertEqual(len(bootstrap_X), len(X))
            self.assertTrue(all(elem in X for elem in bootstrap_X))
            self.assertEqual(len(bootstrap_y), len(y))


class TestTree(unittest.TestCase):

    def setUp(self):
        self.X = np.array([[0, 0],
                      [0, 1],
                      [1, 0],
                      [1, 1]])
        self.y = np.array([0, 0, 1, 1])
        self.train = self.X[:3], self.y[:3]
        self.test = self.X[3:], self.y[3:]


    def test_tree_build(self):
        t = Tree(rand=random.Random(1),
                 get_candidate_columns=random_feature,
                 min_samples=2)
        
        xLeft = self.X[:2]
        yLeft = self.y[:2]
        xRight = self.X[2:]
        yRight = self.y[2:]
        col = 0
        val = 0

        m = Mock()

        def split(X, y):
            return xLeft, yLeft, xRight, yRight, col, val

        m.side_effect = split
        t.split = m

        p = t.build(self.X, self.y)

        np.testing.assert_equal(p.left.X, xLeft)
        np.testing.assert_equal(p.right.X, xRight)

        m.assert_called()
    
    def test_when_split_called_with_valid_input_and_valid_split_exists_it_should_return_it(self):
        getCandidateColumnsMock = Mock(return_value=[0, 1])
        t = Tree(rand=random.Random(1),
                 get_candidate_columns=getCandidateColumnsMock,
                 min_samples=2)

        X = np.array([[1, 2], [8, 2], [5, 5], [7, 5]])
        y = np.array([1, 1, 0, 0])
        splitXLeft, splitYLeft, splitXRight, splitYRight, splitColumn, splitValue = t.split(X, y)

        expectedXLeft = np.array([[1, 2], [8, 2]])
        expectedYLeft = np.array([1, 1])
        expectedXRight = np.array([[5, 5], [7, 5]])
        expectedYRight = np.array([0, 0])
        expectedColumn = 1
        # 3.5 because I take the mean of the expectedColumn and the next value for a better split
        expectedValue = 3.5

        np.testing.assert_array_equal(splitXLeft, expectedXLeft)
        np.testing.assert_array_equal(splitYLeft, expectedYLeft)
        np.testing.assert_array_equal(splitXRight, expectedXRight)
        np.testing.assert_array_equal(splitYRight, expectedYRight)
        np.testing.assert_equal(splitColumn, expectedColumn)
        np.testing.assert_equal(splitValue, expectedValue)

        getCandidateColumnsMock.assert_called()

    def test_when_split_doesnt_exist_split_should_return_None(self):
        getCandidateColumnsMock = Mock(return_value=[0, 1])
        t = Tree(rand=random.Random(1),
                 get_candidate_columns=getCandidateColumnsMock,
                 min_samples=2)

        X = np.array([[1, 2]])
        y = np.array([1])
        splitXLeft, splitYLeft, splitXRight, splitYRight, splitColumn, splitValue = t.split(X, y)

        expectedXLeft = None
        expectedYLeft = None
        expectedXRight = None
        expectedYRight = None
        expectedColumn = None
        expectedValue = None

        np.testing.assert_array_equal(splitXLeft, expectedXLeft)
        np.testing.assert_array_equal(splitYLeft, expectedYLeft)
        np.testing.assert_array_equal(splitXRight, expectedXRight)
        np.testing.assert_array_equal(splitYRight, expectedYRight)
        np.testing.assert_equal(splitColumn, expectedColumn)
        np.testing.assert_equal(splitValue, expectedValue)

        getCandidateColumnsMock.assert_called()

    def test_calculate_gini(self):
        gini = calculateGini(self.y)
        self.assertEqual(gini, 0.5)
        gini = calculateGini(self.y[:2])
        self.assertEqual(gini, 0)
    

# Integration tests
class HWTreeTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[0, 0],
                      [0, 1],
                      [1, 0],
                      [1, 1]])
        self.y = np.array([0, 0, 1, 1])
        self.train = self.X[:3], self.y[:3]
        self.test = self.X[3:], self.y[3:]

    def test_call_tree(self):
        t = Tree(rand=random.Random(1),
                 get_candidate_columns=random_feature,
                 min_samples=2)
        p = t.build(self.X, self.y)
        pred = p.predict(self.X)
        np.testing.assert_equal(pred, self.y)

    def test_call_randomforest(self):
        rf = RandomForest(rand=random.Random(0),
                          n=20)
        p = rf.build(self.X, self.y)
        pred = p.predict(self.X)
        np.testing.assert_equal(pred, self.y)

    def test_call_importance(self):
        rf = RandomForest(rand=random.Random(0),
                          n=20)
        p = rf.build(self.X, self.y)
        imp = p.importance()
        self.assertTrue(len(imp), self.X.shape[1])

    def test_signature_hw_tree_full(self):
        (train, train_un), (test, test_un) = hw_tree_full(self.train, self.test)
        self.assertIsInstance(train, float)
        self.assertIsInstance(test, float)
        self.assertIsInstance(train_un, float)
        self.assertIsInstance(test_un, float)

    def test_signature_hw_randomforests(self):
        (train, train_un), (test, test_un) = hw_randomforests(self.train, self.test)
        self.assertIsInstance(train, float)
        self.assertIsInstance(test, float)
        self.assertIsInstance(train_un, float)
        self.assertIsInstance(test_un, float)


if __name__ == "__main__":
    unittest.main()
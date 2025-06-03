import numpy as np
import pandas as pd
import random, time
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from tqdm import tqdm
from plotly.subplots import make_subplots
import plotly.io as pio   

import unittest
from unittest.mock import Mock, patch

# Disables a message about mathjax added to the saved plots
pio.kaleido.scope.mathjax = None


def all_columns(X, rand):
    return range(X.shape[1])

def random_sqrt_columns(X, rand):
    c = rand.sample(range(X.shape[1]), int(np.sqrt(X.shape[1])))
    return c

def tki():
    df = pd.read_csv("./tki-resistance.csv")
    learnData = df.iloc[:130].to_numpy()
    testData = df.iloc[130:].to_numpy()

    learn = (learnData[:, :-1], learnData[:, -1])
    test = (testData[:, :-1], testData[:, -1])

    legend = df.columns.to_numpy()
    return (learn, test, legend)

def calculateGini(y):
    gini = 0

    for label in np.unique(y):
        p = np.sum(y == label) / len(y)
        gini += p * (1 - p)
    
    return gini

def mode(arr):
    els, counts = np.unique(arr, return_counts=True)
    return els[np.argmax(counts)]

class Tree:
    
    def __init__(self, rand=None,
                 get_candidate_columns=all_columns,
                 min_samples=2):
        self.rand = rand
        self.get_candidate_columns = get_candidate_columns
        self.min_samples = min_samples


    def split(self, X, y):
        columns = self.get_candidate_columns(X, self.rand)
        splitXLeft, splitXRight, splitYLeft, splitYRight, splitValue, splitColumn = None, None, None, None, None, None
        bestGini = -1

        lenX = len(X)

        for col in columns:
            values = np.unique(X[:, col])
            values.sort()
            for idx, value in enumerate(values):
                # If we are not at the last value we take the average of the current and the next value
                # This assures a better split since elements close to our value will be classified the same 
                # as it and those farther are classified as the other class
                if idx != len(values) - 1:
                    value = (value + values[idx + 1])/2

                mask = X[:, col] <= value
                yLeft, yRight = y[mask], y[~mask]

                lenLeft, lenRight = len(yLeft), len(yRight)
                if lenLeft == 0 or lenRight == 0:
                    continue

                currentGini = lenLeft/lenX * calculateGini(yLeft) + lenRight/lenX * calculateGini(yRight)

                if currentGini < bestGini or bestGini == -1:
                    splitYLeft, splitYRight = yLeft, yRight
                    bestGini = currentGini
                    splitColumn = col
                    splitValue = value

                if bestGini == 0:
                    break
        
        if splitValue is None:
            return None, None, None, None, None, None

        splitMask = X[:, splitColumn] <= splitValue
        splitXLeft = X[splitMask]
        splitXRight = X[~splitMask]
        
        return splitXLeft, splitYLeft, splitXRight, splitYRight, splitColumn, splitValue

    def build(self, X, y):
        if len(X) < self.min_samples or len(np.unique(y)) == 1:
            return TreeNode(X, y)

        node = TreeNode()
        
        xLeft, yLeft, xRight, yRight, col, val = self.split(X, y)

        # It is possible we can not split the data based on the subset 
        # of columns such that xLeft and xRight != None
        # If that happens I just return a leaf node
        if xLeft is None or xRight is None:
            return TreeNode(X, y) 

        node.left = self.build(xLeft, yLeft)
        node.right = self.build(xRight, yRight)
        node.splitColIdx = col
        node.splitValue = val

        return node


class TreeNode:

    def __init__(self, X = None, y = None):
        self.X = X
        self.y = y
        self.splitColIdx = None
        self.splitValue = None
        self.left = None
        self.right = None

    def predict(self, X):
        lenX = len(X)
        if self.splitColIdx == None:
            prediction = mode(self.y)
            return np.array(lenX * [prediction])

        prediction = np.empty(lenX, dtype=object)  

        leftIdx = X[:, self.splitColIdx] <= self.splitValue

        xLeft = X[leftIdx]
        xRight = X[~leftIdx]

        prediction[leftIdx] = self.left.predict(xLeft)
        prediction[~leftIdx] = self.right.predict(xRight)
        return prediction


class RandomForest:

    def __init__(self, rand=random.Random(0), n=50):
        self.n = n
        self.rand = rand
        self.rftree = Tree(rand, random_sqrt_columns)

    def build(self, X, y):
        sampleCount = len(X)
        idxs = range(sampleCount)
        idxs_set = set(idxs)
        bootstraps = [self.rand.choices(idxs, k=sampleCount) for _ in range(self.n)]
        oob = list(map(lambda x: idxs_set.difference(x), bootstraps))
        
        rf = RFModel()
        trees = []
        for bootstrap in bootstraps:
            tree = self.rftree.build(X[bootstrap], y[bootstrap])
            trees.append(tree)
        
        rf.trees = np.array(trees)
        # Maybe do a test oob is used correctly
        rf.oob = np.array(oob)
        rf.X = X
        rf.y = y
        rf.rand = self.rand
            
        return rf


class RFModel:

    def __init__(self):
        self.trees = None
        self.oob = None
        self.X = []
        self.y = []
        self.rand = random.Random(0)
        print("RFModel created")

    def predictWithTrees(self, X, trees):
        if trees is None or trees.size == 0:
            return []

        finalPredictions = []
        predictions = np.array([tree.predict(X) for tree in trees])

        for colIdx in range(predictions[0].shape[0]):
            col = predictions[:, colIdx]
            finalPredictions.append(mode(col))

        return finalPredictions

    def predict(self, X):
        return self.predictWithTrees(X, self.trees)
    
    def getSubtreeErrors(self, X):
        errs = []
        for idx, tree in enumerate(self.trees):
            oob = [*self.oob[idx]]
            pred = tree.predict(X[oob])
            errs.append(np.mean(pred != self.y[oob]))

        return np.mean(errs)

    def importance(self):
        imps = np.zeros(self.X.shape[1])
        predErr = self.getSubtreeErrors(self.X)

        for m in range(imps.shape[0]):
            perm_X = self.X.copy()
            self.rand.shuffle(perm_X[:, m])
            noisyErr = self.getSubtreeErrors(perm_X)
            eps = 0.00001 # to not divide by 0
            imps[m] = np.mean((noisyErr - predErr)/(predErr + eps) * 100)

        return imps

def save_to_file(name, avg_build_time, std_build_time, avg_errorRateTrain, avg_stdErrorTrain, avg_errorRateTest, avg_stdErrorTest):
    # Save the average times to a text file
    with open(f'average_times_{name}.txt', 'w') as file:
        file.write(f'Function: {name}\n')
        file.write(f'Average build time: {avg_build_time}\n')
        file.write(f'Std build time: {std_build_time}\n')
        file.write(f'Average train error rate: {avg_errorRateTrain}\n')
        file.write(f'Average train standard error: {avg_stdErrorTrain}\n')
        file.write(f'Average test error rate: {avg_errorRateTest}\n')
        file.write(f'Average test standard error: {avg_stdErrorTest}\n')

def measure(f, runTimes=100, learn=None, test=None, fname="F"):
    errorRateTest, errorRateTrain = 0, 0
    stdErrorTest, stdErrorTrain = 0, 0
    build_time = 0

    build_times = []
    for i in tqdm(range(runTimes)):
        res, build_time_cur = measure_time(f, fname)
        build_times.append(build_time_cur)

        if learn is not None: 
            errorRateTrainCur, stdErrorTrainCur  = get_error_metrics(res, learn)
            errorRateTestCur, stdErrorTestCur  = get_error_metrics(res, test)
            errorRateTrain += errorRateTrainCur
            stdErrorTrain += stdErrorTrainCur
            errorRateTest += errorRateTestCur
            stdErrorTest += stdErrorTestCur

    avg_build_time = np.mean(build_times)
    std_build_time = np.std(build_times)
    avg_errorRateTrain = errorRateTrain/runTimes
    avg_stdErrorTrain = stdErrorTrain/runTimes
    avg_errorRateTest = errorRateTest/runTimes
    avg_stdErrorTest = stdErrorTest/runTimes

    save_to_file(fname, avg_build_time, std_build_time, avg_errorRateTrain, avg_stdErrorTrain, avg_errorRateTest, avg_stdErrorTest)

    return avg_build_time, std_build_time, avg_errorRateTrain, avg_stdErrorTrain, avg_errorRateTest, avg_stdErrorTest

def get_error_metrics(model, data):
    X = data[0]
    y = data[1]

    res = model.predict(X)
    errs = y != res
    errRate = np.sum(errs)/len(res)
    stdErr = np.std(errs)/np.sqrt(len(res)) 
    return errRate, stdErr

def measure_time(f, fname):
    start_time = time.time()
    res = f()
    end_time = time.time()
    build_time = end_time - start_time
    print(f"Time taken to execute {fname}: {build_time}")
    
    return res, build_time

def hw_tree_full(learn, test):
    t = Tree(rand=random.Random(0), get_candidate_columns=all_columns, min_samples=2)
    xTrain = learn[0]
    yTrain = learn[1]
    
    pred, _ = measure_time(lambda : t.build(xTrain, yTrain), "build tree")

    errorRateTrain, stdErrorTrain = get_error_metrics(pred, learn)
    errorRateTest, stdErrorTest = get_error_metrics(pred, test)
    
    return (errorRateTrain, stdErrorTrain), (errorRateTest, stdErrorTest)


def plot_tree_based_on_n(resTrain, resTest):
    resTrain = np.array(resTrain)
    resTest = np.array(resTest)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=resTrain[:, 0], y=resTrain[:, 1], mode='lines', name='Train'))
    fig.add_trace(go.Scatter(x=resTest[:, 0], y=resTest[:, 1], mode='lines', name='Test'))

    fig.update_layout(
        title='Error Rate vs Number of Trees',
        xaxis_title='Number of Trees',
        yaxis_title='Error Rate'
    )

    fig.write_image('error_rate_vs_number_of_trees.pdf')

def hw_tree_n(learn, test):
    xTrain = learn[0]
    yTrain = learn[1]

    resTrain = []
    resTest = []

    for treeCount in tqdm(range(1,101, 1)):
        rf = RandomForest(rand=random.Random(0), n=treeCount)
        model, _ = measure_time(lambda : rf.build(xTrain, yTrain), "build forest")

        errorRateTrain, _  = get_error_metrics(model, learn)
        errorRateTest, _  = get_error_metrics(model, test)

        resTrain.append((treeCount, errorRateTrain))
        resTest.append((treeCount, errorRateTest))

    plot_tree_based_on_n(resTrain, resTest)
    return resTrain, resTest


def hw_importance(learn, test):
    xTrain = learn[0]
    yTrain = learn[1]

    xTest = test[0]
    yTest = test[1]

    X = np.concatenate((xTrain, xTest), axis=0)
    y = np.concatenate((yTrain, yTest), axis=0)

    rf = RandomForest(rand=random.Random(0), n=100)

    model_imps, _ = measure_time(lambda : rf.build(X, y), "build forest")
    imp, _ = measure_time(lambda : model_imps.importance(), "measure importance")

    np.save("importance.npy", imp)
    return imp

def hw_randomforests(learn, test):
    xTrain = learn[0]
    yTrain = learn[1]

    rf = RandomForest(rand=random.Random(0), n=100)

    model, _ = measure_time(lambda : rf.build(xTrain, yTrain), "build forest")

    errorRateTrain, stdErrorTrain  = get_error_metrics(model, learn)
    errorRateTest, stdErrorTest  = get_error_metrics(model, test)

    return (errorRateTrain, stdErrorTrain), (errorRateTest, stdErrorTest)

def double_plot(legend, root, imp):
    pairs = list(zip(legend, imp, root))
    pairs.sort(key=lambda pair: pair[1], reverse=True)
    legend, imp, root = zip(*pairs)
    legend, imp, root = legend[:30], imp[:30], root[:30]

    # Create subplot with 2 rows
    fig = make_subplots(rows=2, cols=1)

    fig.add_trace(go.Bar(x=legend, y=imp, name='Feature Importances'), row=1, col=1)
    fig.add_trace(go.Bar(x=legend, y=root, name='Root Frequency'), row=2, col=1)

    fig.update_xaxes(title_text="Feature", row=1, col=1)
    fig.update_xaxes(title_text="Feature", row=2, col=1)
    fig.update_yaxes(title_text="Importance", row=1, col=1)
    fig.update_yaxes(title_text="Root Frequency", row=2, col=1)

    fig.update_layout(height=800, width=1000)
    fig.write_image('double_plot.pdf')

def plot(legend, val, title, xaxis_title, yaxis_title, filename):
    fig = go.Figure(data=go.Bar(x=legend, y=val, width=0.8))

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        width=1000,
        height = 600
    )

    fig.write_image(filename)


def plot_root(legend, root):
    plot(
        legend=legend,
        val=root,
        title='Most common chosen feature as root',
        xaxis_title='Features',
        yaxis_title='Chosen percentage',
        filename='chosen_root.svg'
    )


def plot_imp(legend, imp):
    plot(
        legend=legend,
        val=imp,
        title='Feature Importances',
        xaxis_title='Features',
        yaxis_title='Importance',
        filename='feature_importances.svg'
    )


def get_roots(learn, test):
    n = 100
    treeBuilder = Tree(rand=random.Random(0))

    xTrain, xTest = learn[0], test[0]
    yTrain, yTest = learn[1], test[1]

    X = np.concatenate((xTrain, xTest), axis=0)
    y = np.concatenate((yTrain, yTest), axis=0)

    idxs = range(X.shape[0])
    sampleCount = len(idxs)

    root = np.zeros(X.shape[1])
    for _ in range(0,n):
        bootstrap = treeBuilder.rand.choices(idxs, k=sampleCount) 
        model = treeBuilder.build(X[bootstrap], y[bootstrap])

        root[model.splitColIdx] += 1
    
    np.save("root.npy", root)
    return root

def plot_roots(legend, roots):
    fig = go.Figure(data=go.Bar(x=legend, y=roots))
    fig.update_layout(title_text='Percentage Chance to Choose Feature for First split', xaxis_title='Feature', yaxis_title='Percentage')
    fig.write_image('roots_chosen.svg')

def plot_roots_and_importance(legend, roots, importance):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=legend, y=roots, name='Roots'))
    fig.add_trace(go.Bar(x=legend, y=importance, name='Importance'))
    fig.update_layout(title_text='Roots and Importance', xaxis_title='Feature', yaxis_title='Percentage', barmode='group')
    fig.write_image('roots_and_importance.svg')


# Unit tests
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
    learn, test, legend = tki()
    print("full", hw_tree_full(learn, test))
    print("random forests", hw_randomforests(learn, test))
    imp = hw_importance(learn, test)
    hw_tree_n(learn, test)
    root = get_roots(learn, test)
    plot_roots_and_importance(legend, root, imp)

    xTrain = learn[0]
    yTrain = learn[1]
    
    # Measure RandomForest performance
    rf = RandomForest(rand=random.Random(0), n=100)
    measure(lambda : rf.build(xTrain, yTrain), 100, learn, test, fname="build_forest")

    #Measure tree performance
    t = Tree(rand=random.Random(0))
    measure(lambda : t.build(xTrain, yTrain), 100, learn, test, fname="build_tree")
    

    #Measure importance
    rf = RandomForest(rand=random.Random(0), n=100)
    model = rf.build(xTrain, yTrain)
    measure(lambda : model.importance(), 100, fname="importance")
    
    # plot root and importance
    double_plot(legend, root, imp)

    unittest.main()
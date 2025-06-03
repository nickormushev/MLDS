import random
import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression as LR
from statsmodels.miscmodels.ordinal_model import OrderedModel
from scipy.optimize import fmin_l_bfgs_b
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

MBOG_TRAIN = 100

def one_hot_encode(Y, category_count = None):
    unique_ys = np.unique(Y)
    if category_count is None:
        category_count = len(unique_ys)

    y_map = {}
    for idx, uy in enumerate(unique_ys):
        y_map[uy] = idx

    encoded_y = np.zeros((len(Y), category_count))
    for idx, y in enumerate(Y):
        encoded_y[idx, y_map[y]] = 1
    
    return encoded_y

def sigmoid(x):
    x_clip = np.clip(x, -500, 500)
    res = 1 / (1 + np.exp(-x_clip))
    return res

def softmax(x):
    # avoid nan
    x = x - np.max(x, axis = 1, keepdims=True)
    rows_sum = np.sum(np.exp(x), axis=1, keepdims=True)
    return np.exp(x) / rows_sum

def sum_feats(x):
    return x[0] + np.log(x[1]) + 5 * x[2]

def multinomial_bad_ordinal_good(n, rand):
    cats = 10
    cols = 3
    max_val = 10000
    overlap = 2/3

    # max sum of the features
    # log added so it is not possible to estimate perfectly
    max_sum = max_val * np.log(max_val) + 5 * max_val

    # feature intervals used as classes that overlap
    intervals = np.linspace(0, max_sum, cats + 1)

    x = np.zeros((n, cols))
    y = np.zeros(n).astype(int)

    # Making sure each class has at least one element
    # to avoid differences in train and test set with few elements
    for i in range(1, cats + 1):
        y[i - 1] = i - 1
        x[i] = [rand.randint(1, max_val) for _ in range(cols)]
        while sum_feats(x[i]) > intervals[i]:
            x[i] = [rand.randint(1, max_val) for _ in range(cols)]
        
    for i in range(cats, n - 1):
        x[i] = [rand.randint(1, max_val) for _ in range(cols)]
        feat_sum = sum_feats(x[i])
        for j in range(1, cats):
            if intervals[j] > feat_sum:
                # If features sum is in the last 3/4 of the interval it can be either 
                # part of the lower class or the higher. I do this so the classes 
                # overlap
                if intervals[j] * overlap < feat_sum:
                    chance = rand.choices([0,1], [0.7, 0.3], k=1)[0]
                    y[i] = chance * (j - 1) + (1 - chance) * j
                else:
                    y[i] = j - 1
                break
    
    # I standardize the features so the OrdinalModel doesn't clip the values
    # when calculating the sigmoid
    x = (x - np.mean(x, axis=0))/np.std(x, axis=0)

    return x, y

def multinomial_predictor(b_shape, y):
    def predictor(parameters, x):
        b = parameters.reshape(b_shape)
        model = MultinomialLogModel(b)

        one_extend = np.ones(x.shape[0])
        x_ext = np.column_stack((x, one_extend.reshape(-1, 1)))
        y_pred = model.predict(x)
        y_act = one_hot_encode(y)
        grad = np.dot(x_ext.T, (y_pred[:, :-1] - y_act[:,:-1]))/len(y)

        return y_pred, grad.T.flatten()
    return predictor

def optimizer(parameters, *args):
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
    encoded_y = one_hot_encode(y, cats)
    # avoid division by zero so offset and normalize
    offset = 10e-10
    y_pred = (y_pred + offset)/np.sum(y_pred + offset, axis=1, keepdims=True)
    return -np.sum(encoded_y * np.log(y_pred)) / len(y)

class MultinomialLogReg:
    def build(self, X, y):
        cats = np.unique(y)
        cat_count = len(cats)
        
        # + 1 for the intercept. -1 cause of the reference class
        b_shape = (cat_count - 1, X.shape[1] + 1)
        initial_values = np.ones(b_shape).flatten()
        predictor = multinomial_predictor(b_shape, y)

        b_opt, _, _ = fmin_l_bfgs_b(optimizer, x0=initial_values, args=(X, y, predictor), approx_grad=False)
        
        return MultinomialLogModel(b_opt.reshape(b_shape), cats)

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

def ordinal_predictor(delta_count):
    def predictor(parameters, x):
        deltas, b = parameters[:delta_count],parameters[delta_count:]
        model = OrdinalLogModel(b, deltas)
        return model.predict(x), None

    return predictor


def get_intervals(x, q):
    percentiles = [quantile * 100 for quantile in q]
    quantiles = np.percentile(x, percentiles)
    quantiles[0] = 0
    intervals = np.diff(quantiles)
    return np.log(intervals)

class OrdinalLogReg:
    def build(self, X, y):
        cats = np.unique(y)
        # +1 for the intercept
        bs_count = X.shape[1] + 1
        # -2 cause we have 2 less deltas than classes
        deltas_count = len(np.unique(y)) - 2


        bins = np.linspace(0.0, 1.0, len(cats) - 1)

        freq = np.bincount(y) / len(y)
        deltas = get_intervals(np.clip(freq.cumsum(), 0, 1), bins)


        initial_values = 0.1 * np.ones(deltas_count + bs_count)
        predictor = ordinal_predictor(deltas_count)
        opt, _, _ = fmin_l_bfgs_b(optimizer, x0=initial_values, args=(X, y, predictor),
                                    bounds=[(10**(-6), None)] * deltas_count + [(None, None)] * bs_count,
                                    approx_grad=True)

        return OrdinalLogModel(opt[deltas_count:], opt[:deltas_count], cats)

class OrdinalLogModel:
    def __init__(self, b, deltas, decode_map = None):
        self.decode_map = decode_map
        self.b = b
        # +2 for the -inf to first border and last border to +inf
        self.class_count = len(deltas) + 2

        # borders are class count - 1 
        # +2 we get for -inf and +inf
        self.brd = np.zeros(self.class_count + 1)
        self.brd[0] = -np.inf
        self.brd[-1] = np.inf

        np.cumsum(deltas, out=self.brd[2:-1])

    def predict(self, x):
        one_extend = np.ones(x.shape[0])
        x_ext = np.column_stack((x, one_extend.reshape(-1, 1)))
        linearPred = np.matmul(x_ext, self.b.T)
        
        probs = np.zeros((x.shape[0], self.class_count))

        for i in range(1, len(self.brd)):
            upper = sigmoid(self.brd[i] - linearPred)
            lower = sigmoid(self.brd[i - 1] - linearPred)
            probs[:, i - 1] = upper - lower 
        
        return probs
    
    def decode(self, y):
        if self.decode_map is None:
            return y

        return [self.decode_map[np.argmax(yi)] for yi in y]

def load_basketball_data(file_name):
    df = pd.read_csv(file_name, delimiter=';')
    y = df['ShotType'].values
    X = df.drop('ShotType', axis=1)

    # Standardize the numerical columns so the weights are more interpretable
    for col in X.columns:
        if X[col].dtype == 'float64':
            X[col] = (X[col] - X[col].mean()) / X[col].std() 

    # The first is correlated to the others which means it can make the model unstable
    # As a result we would prefer to drop it to make the weights more interpretable
    X = pd.get_dummies(X, drop_first=True, dtype=float)
    return X, y


def heatmap(mean, std_err):
    lower_bound = mean - 1.96 * std_err
    upper_bound = mean + 1.96 * std_err
    bounds = list(zip(lower_bound.values.flatten(), upper_bound.values.flatten()))

    confidence_intervals = np.array([f'[{l:.1f}, {u:.1f}]' for l, u in bounds]).reshape(mean.shape)

    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    fig, ax = plt.subplots(figsize=(mean.shape[1] + 0.2, mean.shape[0] + 0.3))
    fig.tight_layout()
    fig.subplots_adjust(top=0.95, right=1.1, left=-0.1, bottom=0.25)
    # Draw the heatmap with the mask and correct aspect ratio
    heatmapFig = sns.heatmap(mean, cmap=cmap, center=0, ax = ax, annot_kws={"size": 7},
                square=True, linewidths=.5, annot=confidence_intervals, fmt='s')


    ax.set_xlabel('Feature')
    ax.set_ylabel('Class')
    plt.yticks(rotation=0)
    plt.xticks(rotation=45)

    fig = heatmapFig.get_figure()
    fig.savefig("heatmap.pdf", dpi=fig.dpi) 

    plt.show()


def heatmapCatReg(X, y):
    bs = []
    shape = (0,0)
    reps = 100
    for i in range(reps):
        print(i)
        np.random.seed(i)
        indices = np.random.choice(len(X), len(X), replace=True)
        bX = X.loc[indices]
        by = y[indices]

        reg = MultinomialLogReg()
        model = reg.build(bX, by)

        shape = model.b.shape
        bs.append(model.b.flatten()) 

    std_err = np.std(bs, axis=0) / np.sqrt(reps)
    mean = np.mean(bs, axis=0)
    mean = mean.reshape(shape)
    std_err = std_err.reshape(shape)

    cols = np.concatenate((X.columns.values, ['Intercept']))
    cats = np.unique(y)
    means = pd.DataFrame(mean, columns = cols , index = cats[:-1])

    np.save("means_non_drop_no_norm.npy", mean)
    np.save("std_errs_non_drop_no_norm.npy", std_err)

    heatmap(means, std_err)

def plot_density(diff):
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))

    density = gaussian_kde(diff)
    xs = np.linspace(min(diff), max(diff), 200)
    ys = density(xs)

    max_x = xs[np.argmax(ys)]

    sns.kdeplot(diff, color="dodgerblue", shade=True, linewidth=3)

    plt.axvline(max_x, color='red', linestyle='--')

    plt.xticks(list(plt.xticks()[0]) + [max_x])

    plt.title('Loss difference density 100/1000 train test split', fontsize=20)
    plt.xlabel('Ordinal - Multinomial loss', fontsize=16)
    plt.ylabel('Density', fontsize=16)
    sns.despine()

    plt.show()

def ordinal_vs_multinomial():
    count = 0
    ordBuilder = OrdinalLogReg()
    mulBuilder = MultinomialLogReg()
    avg_diff = []
    for train_size in [50, 100, 300, 500, 700, 900]:
        diff = []
        print("Running for size:", train_size)
        for i in range(100):
            x_train, y_train = multinomial_bad_ordinal_good(train_size, random.Random(i * 2))
            x_test, y_test = multinomial_bad_ordinal_good(1000, random.Random(i * 2 + 1))

            ordModel = ordBuilder.build(x_train, y_train)
            mulModel = mulBuilder.build(x_train, y_train)

            mulModelPred = mulModel.predict(x_test)
            ordModelPred = ordModel.predict(x_test)

            mulLoss = log_loss(mulModelPred, y_test)
            ordLoss = log_loss(ordModelPred, y_test)

            diff.append(mulLoss - ordLoss)
            count += diff[-1] > 0
            print("Diff:", diff[-1])
        
        avg_diff.append(np.mean(diff))

    
    np.save("diff.npy", diff)

    print("Average difference: {:.2f}".format(np.mean(diff)))
    print("StdErr: {:.2f}".format(np.std(diff) / np.sqrt(len(diff))))
    print("Times multinomial loss was worse than ordinal loss:", count)
    

def plot_avg_diff(X, avg_diff):
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    plt.plot(X, avg_diff, color="dodgerblue", marker='o', linestyle='-')

    plt.title('Multinomial log loss - Ordinal log loss', fontsize=20)
    plt.xlabel('Train set size', fontsize=16)
    plt.ylabel('Average Difference', fontsize=16)

    plt.xticks(X)

    fig.savefig("avg_diff.pdf", dpi=fig.dpi)

    fig = sns.despine()

    plt.show()


# Tests

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
    avg = np.load("avg_diff_array.npy")
    X = [50, 100, 300, 500, 700, 900]
    plot_avg_diff(X, avg)
    #unittest.main()
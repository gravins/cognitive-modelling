from functools import reduce
from operator import mul
from evaluation_func import CONJ_DICT, validate_conjunction
import pandas as pd


class ProbabilisticModel:
    """
    Probabilistic model based on Naive Bayes
    """

    def __init__(self, conjunction_eval=CONJ_DICT["prod"]):
        self.freq = {}
        self.conjunction_eval = validate_conjunction(conjunction_eval).f

    def fit(self, x_train, y_train, laplace=False):
        """
        Fit the model. Compute and store all the prior and likelihood probabilities.
        :param x_train: training values, type: pandas DataFrame
        :param y_train: target values, type: pandas Series
        :param laplace: Laplace smoothing
        """
        if not isinstance(x_train, pd.DataFrame):
            raise TypeError("x_train is not an instance of " + pd.DataFrame.__name__)
        if not isinstance(y_train, pd.Series):
            raise TypeError("y_train is not an instance of " + pd.Series.__name__)
        if len(x_train) != len(y_train):
            raise ValueError("x_train and y_train with different size")

        n_classes = y_train.unique()
        for c in n_classes:
            index = y_train.index[y_train.str.contains(c)].tolist()
            attr = x_train.iloc[index]

            # Compute prior probability
            self.freq[c] = (len(index) / len(y_train), dict.fromkeys(x_train.columns.tolist()))

            # Compute likelihood probability
            for col_x in x_train.columns:
                self.freq[c][1][col_x] = {}
                unique_value = x_train[col_x].unique()
                for val in unique_value:
                    n = len(attr.index[attr[col_x] == val].tolist())
                    self.freq[c][1][col_x][val] = (n + 1) / (len(index) + len(unique_value)) if laplace else n / len(index)

    def predict(self, x_test, queries):
        """
        Compute probability that each sample in x_test is an instance of the corresponding class in queries. If the
        query is stored as a class with which the model was trained the result is computed by the posterior probability,
        if the query is a conjunction the result is computed by the conjunction function, otherwise an error occur.
        :param x_test: test values, type: pandas DataFrame
        :param queries: list of query for the samples in x_test, type: pandas Series
        """
        if not len(self.freq.keys()) > 0:
            raise NotFittedError("This instance of " + self.__class__.__name__ + " is not fitted yet")
        if not isinstance(x_test, pd.DataFrame):
            raise TypeError("x_test is not an instance of " + pd.DataFrame.__name__)
        if not isinstance(queries, pd.Series):
            raise TypeError("queries is not an instance of " + pd.Series.__name__)
        if len(x_test) != len(queries):
            raise ValueError("x_test and queries with different size")

        res = []
        for i, x in x_test.iterrows():
            query = queries[i]

            # Compute marginal probability
            marginal_prob = 0
            for c in self.freq.keys():
                v = self.freq[c][1]
                marginal_prob += self.freq[c][0] * reduce(mul, [v[k][x[k]] for k in v.keys()])

            if query in self.freq.keys():
                # Compute prior * likelihood
                values = self.freq[query][1]
                above = self.freq[query][0] * reduce(mul, [values[k][x[k]] for k in values.keys()])

                # Compute the posterior probability
                res.append(above/marginal_prob)

            elif "and" in query:
                classes = query.replace(" ", "").split("and")
                conj = []
                for c in classes:
                    # Compute prior * likelihood
                    values = self.freq[c][1]
                    above = self.freq[c][0] * reduce(mul, [values[k][x[k]] for k in values.keys()])
                    # Compute the posterior probability
                    conj.append(above / marginal_prob)
                # Compute the conjunction
                res.append(self.conjunction_eval(conj))

            else:
                raise ValueError(query + " is not a good value for query")

        return res


class DistanceModel:
    """
    Distance based model inspired by K Nearest Neighbours
    """
    def __init__(self, conjunction_eval=CONJ_DICT["prod"]):
        self.data = None
        self.target = None
        self.conjunction_eval = validate_conjunction(conjunction_eval).f

    def fit(self, x_train, y_train):
        """
        Fit the model. Store all the data used by the model.
        :param x_train: training values, type: pandas DataFrame
        :param y_train: target values, type: pandas Series
        """
        if not isinstance(x_train, pd.DataFrame):
            raise TypeError("x_train is not an instance of " + pd.DataFrame.__name__)
        if not isinstance(y_train, pd.Series):
            raise TypeError("y_train is not an instance of " + pd.Series.__name__)
        if len(x_train) != len(y_train):
            raise ValueError("x_train and y_train with different size")
        self.data = x_train
        self.target = y_train

    def predict(self, x_test, queries):
        """
        Compute probability that each sample in x_test is an instance of the corresponding class in queries.
        :param x_test: test values, type: pandas DataFrame
        :param queries: list of query for the samples in x_test, type: pandas Series
        """
        if self.data is None or self.target is None:
            raise NotFittedError("This instance of " + self.__class__.__name__ + " is not fitted yet")
        if not isinstance(x_test, pd.DataFrame):
            raise TypeError("x_test is not an instance of " + pd.DataFrame.__name__)
        if not isinstance(queries, pd.Series):
            raise TypeError("queries is not an instance of " + pd.Series.__name__)
        if len(x_test) != len(queries):
            raise ValueError("x_test and queries with different size")

        # Distance is computed as number of mismatch
        distance_func = lambda row1, row2: sum([1 for el1, el2 in zip(row1, row2) if el1 == el2])
        res = []
        for j, x in x_test.iterrows():
            query = queries[j]
            indexes = self.target.index[self.target.str.contains(query)].tolist()
            if len(indexes) > 0:
                res.append(max([distance_func(x, self.data.loc[i]) for i in indexes]) / x_test.shape[1])

            elif "and" in query:
                classes = query.replace(" ", "").split("and")
                conj = []
                for c in classes:
                    indexes = self.target.index[self.target == c].tolist()
                    conj.append(max([distance_func(x, self.data.loc[i]) for i in indexes]) / x_test.shape[1])
                res.append(self.conjunction_eval(conj))

            else:
                raise ValueError(query + " is not a good value for query")

        return res


class NotFittedError(Exception):
    def __init__(self, message):
        super().__init__(message)

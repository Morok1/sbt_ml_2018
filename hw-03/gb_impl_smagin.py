#coding=utf-8

from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeRegressor
import numpy as np


# Параметрами с которыми вы хотите обучать деревья
TREE_PARAMS_DICT = {'max_depth': 5, 'random_state': 21}
# Параметр tau (learning_rate) для вашего GB
TAU = 0.05

def grad_comp(y_data, curr_pred):
    return -2 * y_data / (1 + np.exp(2 * y_data * curr_pred))

class SimpleGB(BaseEstimator):
    def __init__(self, tree_params_dict, iters, tau):
        self.tree_params_dict = tree_params_dict
        self.iters = iters
        self.tau = tau

    def fit(self, X_data, y_data):
        y_data = y_data * 2 - 1

        self.base_algo = DecisionTreeRegressor(**self.tree_params_dict).fit(X_data, y_data)
        self.estimators = []
        curr_pred = self.base_algo.predict(X_data)

        for _ in range(self.iters):
            grad = grad_comp(y_data, curr_pred)
            algo = DecisionTreeRegressor(**self.tree_params_dict).fit(X_data, -grad)
            self.estimators.append(algo)
            curr_pred += self.tau * algo.predict(X_data)
        return self

    def predict(self, X_data):
        res = self.base_algo.predict(X_data)
        for estimator in self.estimators:
            res += self.tau * estimator.predict(X_data)
        return res > -0.05
    


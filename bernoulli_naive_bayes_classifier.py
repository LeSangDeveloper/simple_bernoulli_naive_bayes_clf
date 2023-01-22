import numpy as np

class BernoulliNB():
    def __init__(self, alpha=1.0):
        print("Call init")
        self.alpha = alpha

    def __call__(self, X):
        print("Call predict")

    def fit(self, X, y):
        print("Call fit the model")
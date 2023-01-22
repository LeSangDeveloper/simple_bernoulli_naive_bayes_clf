import numpy as np

class BernoulliNB():
    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, X):
        result = dict()
        for class_name, _ in self.cls_number.items():
            result[class_name] = self.prior[class_name]
            # improve later
            for i in range(len(X[0])):
                if X[0][i] == 1:
                    result[class_name] *= self.params[class_name][i]
                else:
                    result[class_name] *= (1 - self.params[class_name][i])
            i += 1
        total = 0

        for _, value in result.items():
            total += value
        for key, value in result.items():
            result[key] = value / total

        return result

    def fit(self, X, y):
        # identify classes
        classes = np.unique(y)
        self.cls_number = dict()
        
        i = 0
        for class_name in classes:
            self.cls_number[class_name] = i
            i = i + 1

        # count each class
        self.doc_in_class = dict()
        self.prior = dict()
        self.params = dict()
        for class_name, _ in self.cls_number.items():
            self.doc_in_class[class_name] = np.count_nonzero(y == class_name)
            self.prior[class_name] = self.doc_in_class[class_name] / len(y)
            for value, label in zip(X, y):
                if label == class_name:
                    if label not in self.params.keys():
                        self.params[label] = value
                    else:
                        self.params[label] = self.params[label] + value
                    self.params[label] = (self.params[label] + self.alpha) / (np.count_nonzero(y == class_name) + self.alpha + self.beta)
        
        return self
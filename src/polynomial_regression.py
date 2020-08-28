import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

class PolynomialRegression():
    def __init__(self, degree):
        self.degree = degree
        self.model = None
        self.train = None

    def fit(self, features, targets):
        train = np.ones((np.shape(features)[0], self.degree + 1))
        for i in range(len(features)):
            for degree in range(self.degree):
                train[i, degree+1] = features[i] ** (degree+1)

        self.train = train

        self.model = np.dot(np.linalg.inv((np.dot(np.transpose(train), train))), np.dot(np.transpose(train), targets))

    def predict(self, features):
        prediction = np.empty([])
        train = np.ones((np.shape(features)[0], self.degree + 1))

        for i in range(len(features)):
            for degree in range(self.degree):
                train[i, degree + 1] = features[i] ** (degree + 1)

        for i in range(len(train)):
            prediction=np.append(prediction, np.dot(train[i], self.model))

        return np.delete(prediction,0)

    def visualize(self, features, targets):
        print("gg")

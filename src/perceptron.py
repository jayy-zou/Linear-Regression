import numpy as np
import math as math
try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt



def transform_data(features):
    my_features = features.copy()

    for i in range(len(features)):
        my_features[i][0] = np.sqrt(np.square(features[i,0]) + np.square(features[i,1]))
        my_features[i][1] = math.atan(features[i][1] / features[i][0])

    return my_features


class Perceptron():
    def __init__(self, max_iterations=200):
        self.max_iterations = max_iterations
        self.model = None
        self.train = None

    def conv(self, targets, train):
        ans= np.empty([])

        for i in train:
            exp = np.dot(np.transpose(self.model), i)
            if exp>0:
                ans=np.append(ans, 1)
            else:
                ans=np.append(ans, -1)

        return np.allclose(targets, np.delete(ans,0))

    def mis(self, targets, train):
        if np.dot(np.transpose(self.model), train)[0]>0:
            ans=1
        else:
            ans=-1

        return not (ans==targets)

    def fit(self, features, targets):
        self.model = np.random.uniform(-1, 1, (3, 1))

        self.train=np.insert(features, 0, 1, axis=1)

        count = 0
        while (count < self.max_iterations and not self.conv(targets, self.train)):
            for i in range(len(self.train)):
                if (self.mis(targets[i], self.train[i])):
                    self.model = np.transpose(np.transpose(self.model) + (self.train[i] * targets[i]))
            count += 1

    def predict(self, features):
        train=np.insert(features, 0, 1, axis=1)

        predictions = np.empty([])
        for i in range(len(train)):
            if np.dot(np.transpose(self.model), train[i])[0] > 0:
                predictions=np.append(predictions, 1)
            else:
                predictions = np.append(predictions, -1)

        return np.delete(predictions,0)



    def visualize(self, features, targets):
        print("gg")


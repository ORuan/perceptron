class Perceptron(object):
    def __init__(self, learning_rate=0.01, epochs=50):
        self.learning_rate = learning_rate
        self.epochs = epochs
    def train(self, X, y):
        self._weights = np.zeros(1 + X.shape[1])
        self._errors = []
        for _ in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y):
                error = (target - self.predict(xi))
                errors += int(error != 0.0)
                update = self.learning_rate * error
                self._weights[1:] += update * xi
                self._weights[0] += update
            self._errors.append(errors)
        return self
    def net_input(self, X):
        w_bias = self._weights[0]
        return np.dot(X, self._weights[1:]) + w_bias
    def activation(self, X):
        return self.net_input(X)
    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)
algorithm = Perceptron()
training = np.array([2,9,3,1,9,0,2])
algorithm.predict(X=training)
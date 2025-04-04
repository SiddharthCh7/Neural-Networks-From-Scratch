import numpy as np

class BinaryClassifier:
    def __init__(self):
        pass

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def loss_fn(self, y_true, y_pred):
        return np.mean(y_true - y_pred ** 2)
    
    def initialize_weights(self, input_size, hidden_size, output_size):
        np.random.seed(42)
        # Using Xavier initialization
        weights1 = np.random.randn(input_size, hidden_size) * np.sqrt(1. / input_size)
        bias1 = np.zeros((1, hidden_size))

        weights2 = np.random.randn(hidden_size, output_size) * np.sqrt(1. / hidden_size)
        bias2 = np.zeros((1, output_size))

        return weights1, bias1, weights2, bias2

    def forward_pass(self, x):
        weights1, bias1, weights2, bias2 = self.initialize_weights(2, 2, 1)

        # Layer1
        l1 = np.dot(x, weights1) + bias1
        la1 = self.sigmoid(l1)

        # Layer2
        l2 = np.dot(la1, weights2) + bias2
        la2 = self.sigmoid(l2)

        # Final output
        return la2
    
# Data
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Training
net = BinaryClassifier()
y_pred = net.forward_pass(X)
print(net.loss_fn(y_true=y, y_pred=y_pred))




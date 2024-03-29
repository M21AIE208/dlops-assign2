from utils import sigmoid, relu, leaky_relu, tanh
random_values = [-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6, 7.1]

for item in random_values:
    print("Sigmoid:", sigmoid(item))
    print("relu:", relu(item))
    print("leaky-relu:", leaky_relu(item))
    print("tanh:", tanh(item))
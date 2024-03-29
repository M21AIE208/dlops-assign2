import numpy as np
import matplotlib.pyplot as plt
from utils import sigmoid,relu,leaky_relu,tanh
# Generate x values
x = np.linspace(-5, 5, 100)

# Compute y values for each activation function
y_sigmoid = sigmoid(x)
y_relu = relu(x)
y_leaky_relu = leaky_relu(x)
y_tanh = tanh(x)

# Plotting
plt.figure(figsize=(10, 6))

# Sigmoid plot
plt.subplot(2, 2, 1)
plt.plot(x, y_sigmoid)
plt.title('Sigmoid Activation Function')

# ReLU plot
plt.subplot(2, 2, 2)
plt.plot(x, y_relu)
plt.title('ReLU Activation Function')

# Leaky ReLU plot
plt.subplot(2, 2, 3)
plt.plot(x, y_leaky_relu)
plt.title('Leaky ReLU Activation Function')

# Tanh plot
plt.subplot(2, 2, 4)
plt.plot(x, y_tanh)
plt.title('Tanh Activation Function')

# Adjust layout and display the plots
plt.tight_layout()
plt.show()
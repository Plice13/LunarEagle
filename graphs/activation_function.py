import numpy as np
import matplotlib.pyplot as plt

# Define activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

def leaky_relu(x, alpha=0.1):
    return np.where(x > 0, x, x * alpha)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0)

def swish(x, beta=1.0):
    return x * sigmoid(beta * x)

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

def gaussian(x, mu=0, sigma=1):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.)))

# Generate data
x = np.linspace(-2, 2, 100)

# Calculate activation outputs
y_sigmoid = sigmoid(x)
y_relu = relu(x)
y_tanh = tanh(x)
y_leaky_relu = leaky_relu(x)
y_softmax = softmax(x)
y_swish = swish(x)
y_gelu = gelu(x)
y_gaussian = gaussian(x)

# Plotting
plt.figure(figsize=(7*0.7, 6*0.7))

plt.plot(x, y_sigmoid, label='Sigmoid', color='#1f77b4')   # blue
plt.plot(x, y_tanh, label='Tanh', color='#2ca02c')          # green
plt.plot(x, y_leaky_relu, label='Leaky ReLU', color='#ff7f0e')  # red
plt.plot(x, y_swish, label='Swish', color='#9467bd')        # purple
plt.plot(x, y_gelu, label='GELU', color='#8c564b')          # brown
plt.plot(x, y_relu, label='ReLU', color='#d62728')          # orange
plt.subplots_adjust(left=0.1, right=0.92, top=0.92, bottom=0.08)

plt.title('Activation Functions')
plt.xlabel('Input')
plt.ylabel('Output')
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
plt.legend()
plt.show()

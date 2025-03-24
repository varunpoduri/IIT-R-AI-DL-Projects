import numpy as np
import matplotlib.pyplot as plt

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

# Training data
x1, x2 = 3, 2
y_actual = 17

# Initialize random weights
np.random.seed(42)
w1, w2 = np.random.uniform(1, 10, 2)
learning_rate = 0.01
epochs = 20

# Lists to track progress
errors = []
predictions = []

# Training loop
for epoch in range(epochs):
    # Feed Forward
    y_pred = x1 * w1 + x2 * w2
    
    # Compute error
    error = (y_actual - y_pred) ** 2
    errors.append(error)
    predictions.append(y_pred)
    
    # Compute gradients
    grad_w1 = -2 * (y_actual - y_pred) * x1
    grad_w2 = -2 * (y_actual - y_pred) * x2
    
    # Update weights
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
    
    print(f"Epoch {epoch+1}: y_pred={y_pred:.4f}, error={error:.4f}")

# Visualization
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(epochs), errors, marker='o', linestyle='-')
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.title("Error Reduction Over Time")

plt.subplot(1, 2, 2)
plt.plot(range(epochs), predictions, marker='o', linestyle='-')
plt.axhline(y=y_actual, color='r', linestyle='--', label="Actual y")
plt.xlabel("Epochs")
plt.ylabel("Predicted y")
plt.title("Prediction Progress Over Time")
plt.legend()

plt.show()

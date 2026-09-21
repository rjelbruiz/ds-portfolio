# September 21, 2026
# Ross Jervin Lorenz B. Ruiz
# 2023305556
# BS Data Science - DS4A
# DS414 Elective 4 (Deep Learning)
# Laboratory Task 3

import numpy as np

def main():
    x = np.array([1.0, 0.0, 1.0])
    y = np.array([1.0])
    lr = 0.001

    W1 = np.array([
        [ 0.2, -0.3],
        [ 0.4,  0.1],
        [-0.5,  0.2]
    ])
    b1 = np.array([-0.4, 0.2])

    W2 = np.array([
        [-0.3],
        [-0.2]
    ])
    b2 = np.array([0.1])

    def relu(z):
        return np.maximum(0, z)

    def relu_derivative(z):
        return (z > 0).astype(float)

    # Forward Pass
    Z1_Z2 = np.dot(x, W1) + b1
    H = relu(Z1_Z2)

    Z3 = np.dot(H, W2) + b2
    y_hat = relu(Z3)

    error = y_hat - y
    loss = 0.5 * (error ** 2)

    print("Results:")
    print(f"Z1, Z2           : {Z1_Z2}")
    print(f"H (H1, H2)       : {H}")
    print(f"Z3               : {Z3[0]:.4f}")
    print(f"y_hat            : {y_hat[0]:.4f}")
    print(f"Error (y_hat - y): {error[0]:.4f}")
    print(f"Loss             : {loss[0]:.4f}\n")

    # Backward Pass
    dZ3 = error * relu_derivative(Z3)
    dW2 = np.outer(H, dZ3)
    db2 = dZ3

    dH = np.dot(dZ3, W2.T)
    dZ_hidden = dH * relu_derivative(Z1_Z2)

    dW1 = np.outer(x, dZ_hidden)
    db1 = dZ_hidden.reshape(-1)

    print("Computed Gradients:")
    print(f"dL/dW2:\n{dW2}")
    print(f"dL/db2: {db2}")
    print(f"dL/dW1:\n{dW1}")
    print(f"dL/db1: {db1}\n")

    # Gradient Descent Update
    W1_updated = W1 - lr * dW1
    b1_updated = b1 - lr * db1
    W2_updated = W2 - lr * dW2
    b2_updated = b2 - lr * db2

    print("Updated Weights and Biases:")
    print(f"Updated W1:\n{W1_updated}")
    print(f"Updated b1: {b1_updated}")
    print(f"Updated W2:\n{W2_updated}")
    print(f"Updated b2: {b2_updated}")

if __name__ == "__main__":
    main()
# September 21, 2026
# Ross Jervin Lorenz B. Ruiz
# 2023305556
# BS Data Science - DS4A
# DS414 Elective 4 (Deep Learning)
# Laboratory Task 2

import torch
import torch.nn as nn

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    x = torch.tensor([[1.0], 
                      [0.0], 
                      [1.0]], dtype=torch.float32, device=device)

    y = torch.tensor([[1.0]], dtype=torch.float32, device=device)

    W1 = torch.tensor([[ 0.2, -0.3],
                        [ 0.4,  0.1],
                        [-0.5,  0.2]], dtype=torch.float32, requires_grad=True, device=device)

    W2 = torch.tensor([[-0.3],
                        [-0.2]], dtype=torch.float32, requires_grad=True, device=device)

    theta_hidden = torch.tensor([[-0.4, 0.2]], dtype=torch.float32, requires_grad=True, device=device)
    theta_out = torch.tensor([[0.1]], dtype=torch.float32, requires_grad=True, device=device)

    relu = nn.ReLU()

    # Forward Pass
    Z_hidden = torch.matmul(x.T, W1) + theta_hidden
    Z1, Z2 = Z_hidden[0, 0], Z_hidden[0, 1]

    H = relu(Z_hidden)
    H1, H2 = H[0, 0], H[0, 1]

    Z3 = torch.matmul(H, W2) + theta_out
    y_hat = relu(Z3)

    # Loss Calculation
    error = y_hat - y
    mse_loss = 0.5 * (y_hat - y) ** 2

    print("Results:")
    print(f"Z1              : {Z1.item():.4f}")
    print(f"Z2              : {Z2.item():.4f}")
    print(f"H1 (ReLU(Z1))   : {H1.item():.4f}")
    print(f"H2 (ReLU(Z2))   : {H2.item():.4f}")
    print(f"Z3              : {Z3.item():.4f}")
    print(f"Predicted y_hat : {y_hat.item():.4f}")
    print(f"Output Error    : {error.item():.4f}")
    print(f"Squared Error   : {mse_loss.item():.4f}")

if __name__ == "__main__":
    main()
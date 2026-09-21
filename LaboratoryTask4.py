# September 21, 2026
# Ross Jervin Lorenz B. Ruiz
# 2023305556
# BS Data Science - DS4A
# DS414 Elective 4 (Deep Learning)
# Laboratory Task 4

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    num_samples = 160
    num_features = 5

    X_data = np.random.randn(num_samples, num_features).astype(np.float32)
    true_weights = np.array([[1.5], [-2.0], [3.0], [0.5], [-1.0]], dtype=np.float32)
    noise = np.random.randn(num_samples, 1).astype(np.float32) * 0.1
    y_data = np.dot(X_data, true_weights) + noise

    X_tensor = torch.from_numpy(X_data)
    y_tensor = torch.from_numpy(y_data)

    dataset = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(dataset=dataset, batch_size=8, shuffle=True)

    class TwoLayerLinearRegression(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            return self.fc2(self.relu(self.fc1(x)))

    model = TwoLayerLinearRegression(input_dim=num_features, hidden_dim=16, output_dim=1).to(device)

    epochs = 1000
    learning_rate = 0.01
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    print("Training Starts:")
    loss_history = []

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_X.size(0)

        epoch_loss /= len(train_loader.dataset)
        loss_history.append(epoch_loss)

        if epoch % 100 == 0 or epoch == 1:
            print(f"Epoch [{epoch:4d}/{epochs}] - Loss (MSE): {epoch_loss:.6f}")

    print("Training Ends.")
    print(f"Final Training MSE Loss: {loss_history[-1]:.6f}")

if __name__ == "__main__":
    main()
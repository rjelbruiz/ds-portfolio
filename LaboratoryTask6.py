# September 21, 2026
# Ross Jervin Lorenz B. Ruiz
# 2023305556
# BS Data Science - DS4A
# DS414 Elective 4 (Deep Learning)
# Laboratory Task 6

import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.dropout = nn.Dropout(p=0.2)
        
        self.fcn1 = nn.Linear(in_features=256 * 7 * 7, out_features=1000)
        self.fcn2 = nn.Linear(in_features=1000, out_features=500)
        self.fcn3 = nn.Linear(in_features=500, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        
        x = self.dropout(x)
        x = torch.flatten(x, start_dim=1)
        
        x = F.relu(self.fcn1(x))
        x = F.relu(self.fcn2(x))
        x = F.softmax(self.fcn3(x), dim=1)
        
        return x

def main():
    model = CustomCNN(num_classes=10)
    sample_input = torch.randn(32, 1, 28, 28)
    output = model(sample_input)
    
    print("Model Architecture Loaded Successfully!")
    print(f"Input Shape : {sample_input.shape}")
    print(f"Output Shape: {output.shape}")

if __name__ == "__main__":
    main()
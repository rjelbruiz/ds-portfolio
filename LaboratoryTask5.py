# September 21, 2026
# Ross Jervin Lorenz B. Ruiz
# 2023305556
# BS Data Science - DS4A
# DS414 Elective 4 (Deep Learning)
# Laboratory Task 5

import torch
import numpy as np

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    set_seed(42)
    arr = np.random.randint(0, 5, size=6)
    print("Array:")
    print(arr)
    print("-" * 50)

    x = torch.from_numpy(arr)
    print("Tensor x:")
    print(x)
    print("-" * 50)

    x = x.type(torch.int64)
    print("x with updated dtype (int64):")
    print(x)
    print("x.dtype:", x.dtype)
    print("-" * 50)

    x = x.reshape(3, 2)
    print("Reshaped 3x2 Tensor x:")
    print(x)
    print("-" * 50)

    right_column = x[:, 1]
    print("Right-hand column of x:")
    print(right_column)
    print("-" * 50)

    squared_x = x.pow(2)
    print("Square values of x:")
    print(squared_x)
    print("-" * 50)

    set_seed(42)
    y = torch.randint(0, 5, (2, 3), dtype=torch.int64)
    print("Tensor y [2x3]:")
    print(y)
    print("-" * 50)

    matrix_product = torch.mm(x, y)
    print("Matrix Product of x and y [3x3]:")
    print(matrix_product)

if __name__ == "__main__":
    main()
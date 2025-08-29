import numpy as np

A = np.array([
    [0.8, 0.3, 0.1],
    [0.2, 0.7, 0.4],
    [0.1, 0.3, 0.9],
    [0.5, 0.2, 0.3]
])

b = np.array([1.0, 0.8, 0.6, 0.9])  


x = np.ones(A.shape[1])

for k in range(10):  
    grad = 2 * A.T @ (A @ x - b)
    H = 2 * A.T @ A
    delta = np.linalg.solve(H, grad) 
    x = x - delta
    print(f"Iter {k+1}: x = {x}, error = {np.linalg.norm(A@x - b)}")

print("Optimized dwell times:", x)
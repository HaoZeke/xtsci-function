import numpy as np
from cmcrameri import cm
import matplotlib.pyplot as plt

grid = np.load("grid.npz")
X = grid.get("x")
Y = grid.get("y")
Z = grid.get("z")

min_val = np.min(Z)
minima_coords = np.where(Z == min_val)

plt.figure(figsize=(12, 9))
plt.contourf(X, Y, Z, 50, cmap=cm.batlow, alpha=0.8)
plt.colorbar()

# Minima
plt.scatter(
    X[minima_coords], Y[minima_coords], marker="*", color="black", s=200, label="Minima"
)
plt.legend()

plt.show()

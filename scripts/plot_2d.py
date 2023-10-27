import numpy as np
from cmcrameri import cm
import matplotlib.pyplot as plt
import argparse

# Set up command-line argument parsing
parser = argparse.ArgumentParser(description="Load and plot grid data from an NPZ file.")
parser.add_argument("filename", type=str, help="The path to the NPZ file containing grid data.")
args = parser.parse_args()

# Load data from the specified file
grid = np.load(args.filename)
X = grid.get("x")
Y = grid.get("y")
Z = grid.get("z")

# Identify the minima
min_val = np.min(Z)
minima_coords = np.where(Z == min_val)

# Plotting
plt.figure(figsize=(12, 9))
plt.contourf(X, Y, Z, 50, cmap=cm.batlow, alpha=0.8)
plt.colorbar()

# Minima
plt.scatter(
    X[minima_coords], Y[minima_coords], marker="*", color="black", s=200, label="Minima"
)
plt.legend()

plt.show()

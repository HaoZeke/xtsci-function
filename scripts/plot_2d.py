import numpy as np
from cmcrameri import cm
import matplotlib.pyplot as plt
import argparse
from adjustText import adjust_text

# Set up command-line argument parsing
parser = argparse.ArgumentParser(description="Load and plot grid data from an NPZ file.")
parser.add_argument("filename", type=str, help="The path to the NPZ file containing grid data.")
parser.add_argument("--num_minima", type=int, default=1, help="The number of minima to identify and plot.")
parser.add_argument("--exclusion_radius", type=float, default=0.1, help="The exclusion radius around each minimum on function values.")
args = parser.parse_args()

# Load data from the specified file
grid = np.load(args.filename)
X = grid.get("x")
Y = grid.get("y")
Z = grid.get("z")

# Identify the minima with exclusion zones
minima_coords = []
z_copy = np.copy(Z)

for _ in range(args.num_minima):
    min_val_index = np.argmin(z_copy)
    min_coord = np.unravel_index(min_val_index, z_copy.shape)
    minima_coords.append(min_coord)

    # Mask the area around the minima
    exclusion_zone = np.where(np.abs(Z - Z[min_coord]) < args.exclusion_radius)
    z_copy[exclusion_zone] = np.inf

# Plotting
plt.figure(figsize=(12, 9))
plt.contourf(X, Y, Z, 50, cmap=cm.batlow, alpha=0.8)
plt.colorbar()

texts = []
# Plot the identified minima and add text labels
for i, coords in enumerate(minima_coords):
    x_coord = X[coords]
    y_coord = Y[coords]
    z_val = Z[coords]
    plt.scatter(x_coord, y_coord, marker="*", color="black", s=200, label="Minima" if i == 0 else "")
    texts.append(plt.text(x_coord, y_coord, f"M{i} ({x_coord:.2f}, {y_coord:.2f}, {z_val:.2f})", fontsize=9, color="white"))

# Repel the labels to avoid overlaps
adjust_text(texts, arrowprops=dict(arrowstyle="-", color="k"))

# If multiple minima, only label the first one to prevent legend duplicates
plt.legend()

plt.show()

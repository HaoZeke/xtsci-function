import numpy as np
from cmcrameri import cm
import matplotlib.pyplot as plt
import argparse
from adjustText import adjust_text

# Set up command-line argument parsing
parser = argparse.ArgumentParser(
    description="Load and plot grid data from an NPZ file."
)
parser.add_argument(
    "filename", type=str, help="The path to the NPZ file containing grid data."
)
parser.add_argument(
    "--num_minima",
    type=int,
    default=1,
    help="The number of minima to identify and plot.",
)
args = parser.parse_args()

# Load data from the specified file
grid = np.load(args.filename)
X = grid.get("x")
Y = grid.get("y")
Z = grid.get("z")

# Identify the minima
sorted_indices = np.argsort(Z, axis=None)[: args.num_minima]
top_minima_coords = [np.unravel_index(index, Z.shape) for index in sorted_indices]

# Plotting
plt.figure(figsize=(12, 9))
plt.contourf(X, Y, Z, 50, cmap=cm.batlow, alpha=0.8)
plt.colorbar()

texts = []
# Plot the identified minima and add text labels
for i, coords in enumerate(top_minima_coords):
    x_coord = X[coords]
    y_coord = Y[coords]
    z_val = Z[coords]
    plt.scatter(
        x_coord,
        y_coord,
        marker="*",
        color="black",
        s=200,
        label="Minima" if i == 0 else "",
    )
    texts.append(
        plt.text(
            x_coord,
            y_coord,
            f"M{i} ({x_coord:.2f}, {y_coord:.2f}, {z_val:.2f})",
            fontsize=9,
            color="white",
        )
    )

# Repel the labels to avoid overlaps
adjust_text(texts, arrowprops=dict(arrowstyle="-", color="k"))

# If multiple minima, only label the first one to prevent legend duplicates
plt.legend()

plt.show()

# About `xtsci-function`

<img src="./logo_xts-min.png" alt="xtsi logo" width="200" height="200"/>

`xtsci-function` is a C++ library offering a function base class designed for ease of exploration, incorporating optimizers and saddle search methods. It provides a suite of standard test functions from statistics and chemistry, making it ideal for scientific computing and optimization tasks.

## Trial Functions Implemented

- [X] Rosenbrock function
- [X] Himmelblau's function
- [X] Eggholder's function
- [X] Muller-Brown function
- [X] Branin function

## Features

- Allows masks for fixing degrees of freedom, enabling more controlled optimization processes.

## Usage

For visualization and trial runs, `tiny_cli.cpp` can be modified and executed along with Python scripts.

### Build and Run

```bash
pixi shell # Or manage dependencies manually
meson setup bbdir
meson compile -C bbdir
./bbdir/CppCore/tiny_cli
```

## Example Plots

The trial functions can be plotted with the provided scripts:

```bash
python scripts/plot_2d.py "rosen.npz"
python scripts/plot_2d.py "himmelblau.npz" --num_minima 4 --exclusion_radius 0.03
python scripts/plot_2d.py "mullerbrown.npz" --num_minima 3 --exclusion_radius 0.8
python scripts/plot_2d.py "eggholder.npz" --num_minima 5 --exclusion_radius 100
python scripts/plot_2d.py "branin.npz" --num_minima 4
```

## Tests

For the most part these are computed using reference implementations in `R`, or
via `sympy`. These &ldquo;generating&rdquo; scripts are also in the `scripts` folder.

## Provenance

References are provided, but these were originally conceived for use with
[xtsci-optimize](https://github.com/HaoZeke/xtsci-optimize). Most of these are derived from the work of [Surjanovic and
Bingham](https://www.sfu.ca/~ssurjano/index.html) (Virtual Library of Simulation Experiments).

## License

MIT.

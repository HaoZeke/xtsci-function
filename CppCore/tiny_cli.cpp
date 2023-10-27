// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include <fmt/format.h>
#include <fmt/ostream.h>

#include <cmath>
#include <cstdlib>

#include <iostream>
#include <random>

#include "xtensor-fmt/misc.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xvectorize.hpp"

#include "xtsci/func/plot_aid.hpp"
#include "xtsci/func/trial/D2/rosenbrock.hpp"
#include "xtsci/func/trial/D2/himmelblau.hpp"
#include "xtsci/func/trial/D2/mullerbrown.hpp"

int main(int argc, char *argv[]) {
  // Eat warnings, also safer
  static_cast<void>(argc);
  static_cast<void>(argv);
  xts::func::trial::D2::Rosenbrock<double> rosen;
  xts::func::trial::D2::Himmelblau<double> himmelblau;
  // xts::func::trial::D2::QuadraticFunction<double> quadratic;
  // xts::func::trial::D2::Eggholder<double> eggholder;
  xts::func::trial::D2::MullerBrown<double> mullerbrown;

  // clang-format off
  // Rosenbrock
  // xt::xarray<double> min1 = xt::row(rosen.minima, 0); // {0, 0}
  // fmt::print("Rosenbrock Minima: {} with fval {}\n", min1, rosen(min1));
  // clang-format on

  // Grid test
  // auto z_mesh = xts::func::eval_on_grid2D<double>({-2, 2, 100}, {-2, 2, 100}
  // ,rosen); for (auto i : z_mesh) {
  //   fmt::print("{} ", i);
  // }

  // clang-format off
  // Muller Brown
  // xt::xarray<double> min1 = xt::row(mullerbrown.minima, 0); // {-0.558, 1.441}
  // xt::xarray<double> min2 = xt::row(mullerbrown.minima, 1); // {0.623, 0.028}
  // xt::xarray<double> min3 = xt::row(mullerbrown.minima, 2); // {-0.050, 0.466}
  // fmt::print("Minima: {}, {}, {}\n", min1, min2, min3);
  // clang-format on

  // clang-format off
  // Himmelblau
  // xt::xarray<double> min1 =
  // xt::row(himmelblau.minima, 0); // {3, 2};
  // xt::xarray<double> min2 =
  // xt::row(himmelblau.minima, 1); // {-2.805118, 3.131312};
  // xt::xarray<double> min3 =
  // xt::row(himmelblau.minima, 2); // {-3.779310, -3.283186};
  // xt::xarray<double> min4 =
  // xt::row(himmelblau.minima, 3); // {3.584428, -1.848126};
  // fmt::print("Himmelblau Minima:\n{}, {}\n{}, {}\n{}, {}\n{}, {}\n", min1,
  //            himmelblau(min1), min2, himmelblau(min2), min3, himmelblau(min3),
  //            min4, himmelblau(min4));
  // clang-format on

  // npz test
  xts::func::npz_on_grid2D<double>({-2, 2, 100}, {-2, 2, 100}, rosen, "rosen.npz");
  xts::func::npz_on_grid2D<double>({-5, 5, 400}, {-5, 5, 400}, himmelblau, "himmelblau.npz");
  xts::func::npz_on_grid2D<double>({-1.5, 1.2, 400}, {-0.2, 2.0, 400}, mullerbrown, "mullerbrown.npz");

  return EXIT_SUCCESS;
}

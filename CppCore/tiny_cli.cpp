// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include <fmt/format.h>
#include <fmt/ostream.h>

#include <cmath>
#include <cstdlib>

#include <iostream>
#include <random>

#include "xtensor-io/xnpz.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor-fmt/misc.hpp"

#include "xtsci/func/trial/rosenbrock.hpp"

int main(int argc, char *argv[]) {
  // Eat warnings, also safer
  static_cast<void>(argc);
  static_cast<void>(argv);
  xts::func::trial::Rosenbrock<double> rosen;
  // xts::optimize::trial_functions::Himmelblau<double> himmelblau;
  // xts::optimize::trial_functions::QuadraticFunction<double> quadratic;
  // xts::optimize::trial_functions::Eggholder<double> eggholder;
  // xts::optimize::trial_functions::MullerBrown<double> mullerbrown;

  // clang-format off
  // Rosenbrock
  xt::xarray<double> min1 = xt::row(rosen.minima, 0); // {0, 0}
  fmt::print("Rosenbrock Minima: {} with fval {}\n", min1, rosen(min1));
  // clang-format on

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
  // fmt::print("Minima: {}, {}, {}, {}\n", min1, min2, min3, min4);
  // clang-format on

  return EXIT_SUCCESS;
}

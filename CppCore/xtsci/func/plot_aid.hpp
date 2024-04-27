#pragma once
// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include <algorithm>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "xtensor-blas/xlinalg.hpp"
#include "xtensor-io/xnpz.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xvectorize.hpp"

#include "xtsci/func/helpers.hpp"

namespace xts {
namespace func {
template <typename ScalarType>
xt::xarray<ScalarType> eval_on_grid2D(
    const std::array<ScalarType, 3> &axOne,
    const std::array<ScalarType, 3> &axTwo,
    std::function<ScalarType(ScalarType x_val, ScalarType y_val)> func) {
  auto x_line = xt::linspace<ScalarType>(axOne[0], axOne[1], axOne[2]);
  auto y_line = xt::linspace<ScalarType>(axTwo[0], axTwo[1], axTwo[2]);
  auto [x_mesh, y_mesh] = xt::meshgrid(x_line, y_line);
  auto vec              = xt::vectorize(func);
  return vec(x_mesh, y_mesh);
}

template <typename ScalarType>
void npz_on_grid2D(
    const std::array<ScalarType, 3> &axOne,
    const std::array<ScalarType, 3> &axTwo,
    std::function<ScalarType(ScalarType x_val, ScalarType y_val)> func,
    std::string filename = "grid.npz") {
  auto x_line = xt::linspace<ScalarType>(axOne[0], axOne[1], axOne[2]);
  auto y_line = xt::linspace<ScalarType>(axTwo[0], axTwo[1], axTwo[2]);
  auto [x_mesh, y_mesh] = xt::meshgrid(x_line, y_line);
  auto vec              = xt::vectorize(func);
  auto z_val            = vec(x_mesh, y_mesh);
  xt::dump_npz(filename, "z", z_val, true, true);
  xt::dump_npz(filename, "x", x_mesh, true, true);
  xt::dump_npz(filename, "y", y_mesh, true, true);
}
} // namespace func
} // namespace xts

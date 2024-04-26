#pragma once
// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include <algorithm>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "xtensor-blas/xlinalg.hpp"
#include "xtensor/xarray.hpp"

namespace xts {
namespace func {
namespace helpers {
template <class E, class ScalarType = double>
void ensure_normalized(E &&vector,
                       bool is_normalized = false,
                       ScalarType tol = static_cast<ScalarType>(1e-6)) {
  if (!is_normalized) {
    auto norm = xt::linalg::norm(vector, 2);
    if (std::abs(norm - static_cast<ScalarType>(1.0)) >= tol) {
      vector /= norm;
    } else {
      throw std::runtime_error(
          "Cannot normalize a vector whose norm is smaller than tol");
    }
  }
}
} // namespace helpers
} // namespace func
} // namespace xts

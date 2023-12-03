#pragma once
// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include <numbers>

#include "xtensor-blas/xlinalg.hpp"
#include "xtsci/func/base.hpp"

namespace xts {
namespace func {
namespace trial {
namespace D1 {

template <typename ScalarType>
class QuadraticFunction : public ObjectiveFunction<ScalarType, 1> {
  // Domain is R^n
  // Global minimum is at x = 0 with f(x) = 0
  explicit QuadraticFunction(const xt::xtensor<bool, 1> &isFixed = {})
      : ObjectiveFunction<ScalarType, 1>(isFixed) {
    this->minima = {{0.0}};
  }

private:
  ScalarType compute(const xt::xarray<ScalarType> &x) const override {
    return xt::linalg::dot(x, x)(0); // x^T x
  }

  std::optional<xt::xarray<ScalarType>>
  compute_gradient(const xt::xarray<ScalarType> &x) const override {
    return 2.0 * x; // 2x
  }

  std::optional<xt::xarray<ScalarType>>
  compute_hessian(const xt::xarray<ScalarType> &x) const override {
    // For a quadratic function, the Hessian is constant: 2I where I is the
    // identity matrix.
    return 2.0 * xt::eye(x.size());
  }
};

} // namespace D1
} // namespace trial
} // namespace func
} // namespace xts

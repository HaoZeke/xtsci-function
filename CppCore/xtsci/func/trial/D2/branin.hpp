#pragma once
// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include <numbers>

#include "xtensor-blas/xlinalg.hpp"
#include "xtsci/func/base.hpp"

namespace xts {
namespace func {
namespace trial {
namespace D2 {

template <typename ScalarType = double>
class Branin : public ObjectiveFunction<ScalarType> {
  // More details: https://www.sfu.ca/~ssurjano/branin.html
  // Domain is R^2
  // Global minimum are at x = (\pi, 12.275), (\pi, 2.275), (9.42478, 2.475)
  // with f(x) = 0.397887
public:
  explicit Branin(const xt::xtensor<bool, 1> &isFixed = xt::zeros<bool>({2}))
      : ObjectiveFunction<ScalarType>(2, isFixed) {
    // Initialize the minima for the Branin function
    this->minima = {{std::numbers::pi_v<ScalarType>, 12.275},
                    {std::numbers::pi_v<ScalarType>, 2.275},
                    {9.42478, 2.475}}; // Three minima
  }

private:
  static constexpr ScalarType a = 1;
  static constexpr ScalarType b = 5.1 / (4 * std::numbers::pi_v<ScalarType> *
                                         std::numbers::pi_v<ScalarType>);
  static constexpr ScalarType c = 5 / std::numbers::pi_v<ScalarType>;
  static constexpr ScalarType r = 6;
  static constexpr ScalarType s = 10;
  static constexpr ScalarType t = 1 / (8 * std::numbers::pi_v<ScalarType>);

  ScalarType compute(const xt::xarray<ScalarType> &x) const override {
    ScalarType x1 = x(0);
    ScalarType x2 = x(1);
    return a * std::pow(x2 - b * x1 * x1 + c * x1 - r, 2) +
           s * (1 - t) * std::cos(x1) + s;
  }

  std::optional<xt::xarray<ScalarType>>
  compute_gradient(const xt::xarray<ScalarType> &x) const override {
    ScalarType x1 = x(0);
    ScalarType x2 = x(1);
    ScalarType df_dx1 =
        2 * a * (x2 - b * x1 * x1 + c * x1 - r) * (-2 * b * x1 + c) -
        s * (1 - t) * std::sin(x1);
    ScalarType df_dx2 = 2 * a * (x2 - b * x1 * x1 + c * x1 - r);
    return xt::xarray<ScalarType>{df_dx1, df_dx2};
  }

  std::optional<xt::xarray<ScalarType>>
  compute_hessian(const xt::xarray<ScalarType> &x) const override {
    ScalarType x1 = x(0);
    ScalarType d2f_dx12 = 2 * a * (-2 * b + 2 * b * c - 4 * b * b * x1) -
                          s * (1 - t) * std::cos(x1);
    ScalarType d2f_dxdy = 2 * a * (-2 * b * x1 + c);
    ScalarType d2f_dy2 = 2 * a;
    xt::xarray<ScalarType> hess = {{d2f_dx12, d2f_dxdy}, {d2f_dxdy, d2f_dy2}};
    return hess;
  }
};

} // namespace D2
} // namespace trial
} // namespace func
} // namespace xts

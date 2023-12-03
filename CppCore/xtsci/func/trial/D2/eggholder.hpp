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

#include "xtsci/func/base.hpp"

namespace xts {
namespace func {
namespace trial {
namespace D2 {

template <typename ScalarType = double>
class Eggholder : public ObjectiveFunction<ScalarType, 2> {
  // Domain is -512 to 512
  // Minimum at (512, 404.2319) with value -959.6407
public:
  Eggholder(const xt::xtensor<bool, 1> &isFixed = xt::zeros<bool>({2}))
      : ObjectiveFunction<ScalarType, 2>(isFixed) {
    this->minima = {{512, 404.2319}};
  }

private:
  ScalarType compute(const xt::xarray<ScalarType> &x) const override {
    ScalarType x_val = x(0);
    ScalarType y_val = x(1);
    return -(y_val + 47) *
               std::sin(std::sqrt(std::abs(x_val / 2 + (y_val + 47)))) -
           x_val * std::sin(std::sqrt(std::abs(x_val - (y_val + 47))));
  }
};

} // namespace D2
} // namespace trial
} // namespace func
} // namespace xts

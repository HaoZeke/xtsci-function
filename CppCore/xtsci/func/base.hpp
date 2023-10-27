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
#include "xtensor/xarray.hpp"

#include "xtsci/func/helpers.hpp"

namespace xts {
namespace func {

struct EvaluationCounter {
  size_t function_evals = 0;
  size_t gradient_evals = 0;
  size_t hessian_evals = 0;
  size_t unique_func_grad = 0;
};

template <typename ScalarType = double> class ObjectiveFunction {
public:
  ObjectiveFunction() = default;
  explicit ObjectiveFunction(const xt::xarray<ScalarType> &min) : minima(min) {}
  explicit ObjectiveFunction(const xt::xarray<ScalarType> &min,
                             const xt::xarray<ScalarType> &sad)
      : minima(min), saddles(sad) {}
  const xt::xarray<ScalarType> minima;
  const xt::xarray<ScalarType> saddles;

  virtual ~ObjectiveFunction() = default;

  ScalarType operator()(const xt::xarray<ScalarType> &x) const {
    ++m_counter.function_evals;
    ++m_counter.unique_func_grad;
    return this->compute(x);
  }

  ScalarType operator()(ScalarType x_val, ScalarType y_val) const {
    xt::xarray<ScalarType> input = {x_val, y_val};
    return this->operator()(input);
  }

  virtual std::optional<xt::xarray<ScalarType>>
  gradient(const xt::xarray<ScalarType> &x) const {
    ++m_counter.gradient_evals;
    ++m_counter.unique_func_grad;
    return this->compute_gradient(x);
  }

  virtual std::optional<xt::xarray<ScalarType>>
  hessian(const xt::xarray<ScalarType> &x) const {
    ++m_counter.hessian_evals;
    return this->compute_hessian(x);
  }

  ScalarType
  directional_derivative(const xt::xarray<ScalarType> &x,
                         const xt::xarray<ScalarType> &direction) const {
    auto grad_opt = gradient(x);
    if (!grad_opt) {
      throw std::runtime_error(
          "Gradient required for computing directional derivative.");
    }
    return xt::linalg::dot(*grad_opt, direction)();
  }

  std::pair<xt::xarray<ScalarType>, xt::xarray<ScalarType>>
  grad_components(const xt::xarray<ScalarType> &xpt,
                  xt::xarray<ScalarType> &direction,
                  bool is_normalized = false) const {
    helpers::ensure_normalized(direction, is_normalized);
    auto grad = this->gradient(xpt).value();
    auto parallel_projection = xt::linalg::dot(grad, direction) * direction;
    auto perpendicular_projection = grad - parallel_projection;
    return {parallel_projection, perpendicular_projection};
  }

  EvaluationCounter evaluation_counts() const { return m_counter; }

private:
  mutable EvaluationCounter m_counter;

  virtual ScalarType compute(const xt::xarray<ScalarType> &x) const = 0;

  virtual std::optional<xt::xarray<ScalarType>>
  compute_gradient(const xt::xarray<ScalarType> &) const {
    return std::nullopt;
  }

  virtual std::optional<xt::xarray<ScalarType>>
  compute_hessian(const xt::xarray<ScalarType> &) const {
    return std::nullopt;
  }
};

} // namespace func
} // namespace xts

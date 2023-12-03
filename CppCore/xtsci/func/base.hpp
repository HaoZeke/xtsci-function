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

#include "xtensor/xbuilder.hpp"
#include "xtsci/func/helpers.hpp"

namespace xts {
namespace func {

struct EvaluationCounter {
  size_t function_evals = 0;
  size_t gradient_evals = 0;
  size_t hessian_evals = 0;
  size_t unique_func_grad = 0;
};

template <typename ScalarType = double, size_t Dims = 2>
class ObjectiveFunction {
public: // Variables
        // TODO: Better sanity checks, make m_isFixed private and check dims on
        // setter
  xt::xtensor<ScalarType, 2> minima;
  xt::xtensor<ScalarType, 2> saddles;
  xt::xtensor<bool, 1> m_isFixed;

public: // Constructors and destructor
  virtual ~ObjectiveFunction() = default;
  // Default constructor
  ObjectiveFunction() : m_isFixed(xt::zeros<bool>({Dims})) {
    minima = xt::xtensor<ScalarType, 1>::empty();
    saddles = xt::xtensor<ScalarType, 1>::empty();
  }

  // Constructor with optional fixed mask
  explicit ObjectiveFunction(const xt::xtensor<bool, 1> &isFixed)
      : m_isFixed(isFixed) {
    if (m_isFixed.size() != Dims) {
      throw std::invalid_argument(
          "Size of isFixed mask does not match the problem dimensionality.");
    }
  }

public: // Functions and Operators
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
    auto grad = this->compute_gradient(x);
    if (!grad) {
      return std::nullopt;
    }

    // Zero out gradients for fixed degrees of freedom
    for (std::size_t idx = 0; idx < grad->size(); ++idx) {
      if (m_isFixed.at(idx)) {
        (*grad)[idx] = 0.0;
      }
    }

    return grad;
  }

  virtual std::optional<xt::xarray<ScalarType>>
  hessian(const xt::xarray<ScalarType> &x) const {
    ++m_counter.hessian_evals;
    auto hess = this->compute_hessian(x);
    if (!hess) {
      return std::nullopt;
    }

    // Zero out Hessian rows and columns for fixed degrees of freedom
    for (size_t idx = 0; idx < hess->shape()[0]; ++idx) {
      for (size_t jdx = 0; jdx < hess->shape()[1]; ++jdx) {
        if (m_isFixed.at(idx) || m_isFixed.at(jdx)) {
          (*hess)(idx, jdx) = 0;
        }
      }
    }

    return hess;
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

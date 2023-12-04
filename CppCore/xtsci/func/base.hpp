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
  size_t unique_func_grad_hess = 0;
};

template <typename ScalarType = double> class ObjectiveFunction {
private: // Variables
  size_t m_dims;
  mutable xt::xarray<ScalarType> m_lastX;
  mutable ScalarType m_lastFunctionValue;
  mutable std::optional<xt::xarray<ScalarType>> m_lastGradient;
  mutable std::optional<xt::xarray<ScalarType>> m_lastHessian;
  mutable bool m_isCacheValid = false;
  mutable EvaluationCounter m_counter;

public: // Variables
        // TODO(rg): Better sanity checks, make m_isFixed private and check
        // m_dims on setter
  xt::xtensor<ScalarType, 2> minima;
  xt::xtensor<ScalarType, 2> saddles;
  xt::xtensor<bool, 1> m_isFixed;

public: // Constructors and destructor
  virtual ~ObjectiveFunction() = default;
  // Default constructor
  explicit ObjectiveFunction(size_t dims)
      : m_dims(dims), m_isFixed(xt::zeros<bool>({m_dims})) {
    auto shape = std::vector<size_t>{
        0, m_dims}; // Create an empty tensor with 0 rows and m_dims columns
    minima = xt::empty<ScalarType>(shape);
    saddles = xt::empty<ScalarType>(shape);
  }

  // Constructor with optional fixed mask
  explicit ObjectiveFunction(size_t dims, const xt::xtensor<bool, 1> &isFixed)
      : m_dims(dims), m_isFixed(isFixed) {
    if (m_isFixed.size() != m_dims) {
      throw std::invalid_argument(
          "Size of isFixed mask does not match the problem dimensionality.");
    }
    auto shape = std::vector<size_t>{0, m_dims};
    minima = xt::empty<ScalarType>(shape);
    saddles = xt::empty<ScalarType>(shape);
  }

public: // Functions and Operators
  ScalarType operator()(const xt::xarray<ScalarType> &x) const {
    ++m_counter.function_evals;
    if (!(m_isCacheValid && xt::all(xt::equal(x, m_lastX)))) {
      m_lastX = x;
      m_lastFunctionValue = this->compute(x);
      m_isCacheValid = true;
      ++m_counter.unique_func_grad_hess;
    }
    return m_lastFunctionValue;
  }

  virtual std::optional<xt::xarray<ScalarType>>
  gradient(const xt::xarray<ScalarType> &x) const {
    ++m_counter.gradient_evals;
    if (!m_isCacheValid || !m_lastGradient.has_value() ||
        !xt::allclose(x, m_lastX)) {
      m_lastX = x;
      ++m_counter.unique_func_grad_hess;
      m_lastGradient = this->compute_gradient(x);
      m_isCacheValid = true;
      applyFixedMaskToGradient(m_lastGradient);
    }
    return m_lastGradient;
  }

  virtual std::optional<xt::xarray<ScalarType>>
  hessian(const xt::xarray<ScalarType> &x) const {
    ++m_counter.hessian_evals;
    if (!m_isCacheValid || !m_lastHessian.has_value() ||
        !xt::allclose(x, m_lastX)) {
      m_lastX = x;
      ++m_counter.unique_func_grad_hess;
      m_lastHessian = this->compute_hessian(x);
      m_isCacheValid = true;
      applyFixedMaskToHessian(m_lastHessian);
    }
    return m_lastHessian;
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
  virtual ScalarType compute(const xt::xarray<ScalarType> &x) const = 0;

  virtual std::optional<xt::xarray<ScalarType>>
  compute_gradient(const xt::xarray<ScalarType> &) const {
    return std::nullopt;
  }

  virtual std::optional<xt::xarray<ScalarType>>
  compute_hessian(const xt::xarray<ScalarType> &) const {
    return std::nullopt;
  }
  void
  applyFixedMaskToGradient(std::optional<xt::xarray<ScalarType>> &grad) const {
    if (!grad) {
      return;
    }
    for (std::size_t idx = 0; idx < grad->size(); ++idx) {
      if (m_isFixed.at(idx)) {
        (*grad)[idx] = 0.0;
      }
    }
  }

  void
  applyFixedMaskToHessian(std::optional<xt::xarray<ScalarType>> &hess) const {
    if (!hess) {
      return;
    }
    for (size_t idx = 0; idx < hess->shape()[0]; ++idx) {
      for (size_t jdx = 0; jdx < hess->shape()[1]; ++jdx) {
        if (m_isFixed.at(idx) || m_isFixed.at(jdx)) {
          (*hess)(idx, jdx) = 0;
        }
      }
    }
  }
};

} // namespace func
} // namespace xts

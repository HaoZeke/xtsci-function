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

#include "rgpot/Potential.hpp"
#include "rgpot/types/adapters/xtensor.hpp"
#include "xtsci/func/base.hpp"
#include <xtensor/xadapt.hpp>

namespace xts {
namespace pot {

template <typename ScalarType = double>
class XTPot : public func::ObjectiveFunction<ScalarType> {
public:
  XTPot(std::shared_ptr<rgpot::Potential> pot,
        const xt::xtensor<int, 1> &atomTypes,
        const xt::xtensor<double, 2> &boxMatrix,
        const xt::xtensor<bool, 1> &fixedMask = {})
      : m_pot(pot),
        m_atomTypes(rgpot::types::adapt::xtensor::convertToVector(atomTypes)),
        m_box(rgpot::types::adapt::xtensor::convertToArray3x3(boxMatrix)),
        m_fixedMask(fixedMask) {
    if (m_fixedMask.size() == 0) {
      m_fixedMask = xt::zeros<bool>({m_atomTypes.size()});
    }
  }

  virtual ~XTPot() = default;

private:
  std::shared_ptr<rgpot::Potential> m_pot;
  std::vector<int> m_atomTypes;
  std::array<std::array<double, 3>, 3> m_box;
  xt::xtensor<bool, 1> m_fixedMask;

  virtual ScalarType compute(const xt::xarray<ScalarType> &x) const override {
    xt::xtensor<double, 2> positions = this->reshape_x_to_positions(x);

    auto [energy, forces] =
        (*m_pot)(rgpot::types::adapt::xtensor::convertToAtomMatrix(positions),
                 m_atomTypes, m_box);

    return energy;
  }

  virtual std::optional<xt::xarray<ScalarType>>
  compute_gradient(const xt::xarray<ScalarType> &x) const override {
    xt::xtensor<double, 2> positions = this->reshape_x_to_positions(x);

    auto [energy, forces] =
        (*m_pot)(rgpot::types::adapt::xtensor::convertToAtomMatrix(positions),
                 m_atomTypes, m_box);

    // Zero out forces for constrained atoms
    for (size_t idx{0}; idx < forces.rows(); idx++) {
      if (m_fixedMask[idx] == true) {
        forces(idx, 0) = 0.0;
        forces(idx, 1) = 0.0;
        forces(idx, 2) = 0.0;
      }
    }

    std::array<size_t, 2> shape = {m_atomTypes.size(), 3};
    auto forces_xtensor =
        xt::adapt(forces.data(), shape, xt::layout_type::row_major);

    // Convert forces to gradient
    auto gradient = -1.0 * forces_xtensor;
    auto flattened_gradient = xt::flatten(gradient);
    return flattened_gradient;
  }

  xt::xtensor<double, 2>
  reshape_x_to_positions(const xt::xarray<ScalarType> &x) const {
    std::array<std::size_t, 2> shape = {m_atomTypes.size(), 3};
    return xt::reshape_view(x, shape);
  }
};

} // namespace pot
} // namespace xts

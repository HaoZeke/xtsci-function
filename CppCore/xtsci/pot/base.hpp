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

namespace xts {
namespace pot {

template <typename ScalarType = double>
class XTPot : public func::ObjectiveFunction<ScalarType> {
public:
  XTPot(std::shared_ptr<rgpot::Potential> pot,
        const xt::xtensor<int, 1> &atomTypes,
        const xt::xtensor<double, 2> &boxMatrix)
      : m_pot(pot),
        m_atomTypes(rgpot::types::adapt::xtensor::convertToVector(atomTypes)),
        m_box(rgpot::types::adapt::xtensor::convertToArray3x3(boxMatrix)) {}

  virtual ~XTPot() = default;

private:
  std::shared_ptr<rgpot::Potential> m_pot;
  std::vector<int> m_atomTypes;
  std::array<std::array<double, 3>, 3> m_box;

  virtual ScalarType compute(const xt::xarray<ScalarType> &x) const override {
    // Create a mutable copy.
    auto x_mutable = x;

    // Assuming x is a flattened version of positions.
    // We reshape it back into the shape of positions.
    xt::xtensor<double, 2> positions =
        x_mutable.reshape({m_atomTypes.size(), 3});

    // Use the stored Potential object to compute energy and forces.
    auto [energy, forces] =
        (*m_pot)(rgpot::types::adapt::xtensor::convertToAtomMatrix(positions),
                 m_atomTypes, m_box);

    return energy; // Assuming we only want the energy for the objective
                   // function.
  }
};

} // namespace pot
} // namespace xts

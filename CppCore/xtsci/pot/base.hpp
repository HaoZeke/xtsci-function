#pragma once
// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "rgpot/Potential.hpp"
#include "rgpot/types/adapters/xtensor.hpp"
#include "xtsci/func/base.hpp"
#include <xtensor/xadapt.hpp>
#include <xtensor/xindex_view.hpp>

namespace xts {
namespace pot {

template <typename ScalarType = double>
class XTPot : public func::ObjectiveFunction<ScalarType> {
public:
  XTPot(std::shared_ptr<rgpot::Potential> pot,
        const xt::xtensor<ScalarType, 2> &base_pos,
        const xt::xtensor<int, 1> &atomTypes,
        const xt::xtensor<double, 2> &boxMatrix,
        const xt::xtensor<bool, 1> &fixedMask = {})
      : func::ObjectiveFunction<ScalarType>(
            atomTypes.size() * 3, expandFixedMask(fixedMask, atomTypes.size())),
        m_pot(pot), m_basepos(base_pos),
        m_atomTypes(rgpot::types::adapt::xtensor::convertToVector(atomTypes)),
        m_box(rgpot::types::adapt::xtensor::convertToArray3x3(boxMatrix)) {}

  virtual ~XTPot() = default;

protected: // Useful to test the damn thing
  xt::xtensor<double, 2>
  reconstruct_full(const xt::xarray<ScalarType> &free_x) {
    std::array<std::size_t, 2> shape = {m_atomTypes.size(), 3};
    xt::xtensor<double, 2> allpos = xt::reshape_view(m_basepos, shape);
    xt::xarray<bool> free_mask = xt::reshape_view(!this->m_isFixed, shape);
    // Flatten all structures for simpler indexing
    auto flat_indices = xt::from_indices(xt::nonzero(xt::flatten(free_mask)));
    // Check the size of free_x to match the flat indices count
    if (flat_indices.size() != free_x.size()) {
      throw std::runtime_error(
          "Size mismatch between free positions and unmasked indices.");
    }
    xt::index_view(xt::flatten(allpos), flat_indices) = xt::flatten(free_x);
    return allpos;
  }

private:
  std::shared_ptr<rgpot::Potential> m_pot;
  std::vector<int> m_atomTypes;
  std::array<std::array<double, 3>, 3> m_box;

  ScalarType compute(const xt::xarray<ScalarType> &x) const override {
    xt::xtensor<double, 2> positions = this->reshape_x_to_positions(x);

    auto [energy, forces] =
        (*m_pot)(rgpot::types::adapt::xtensor::convertToAtomMatrix(positions),
                 m_atomTypes, m_box);

    return energy;
  }

  std::optional<xt::xarray<ScalarType>>
  compute_gradient(const xt::xarray<ScalarType> &x) const override {
    xt::xtensor<double, 2> positions = this->reshape_x_to_positions(x);

    auto [energy, forces] =
        (*m_pot)(rgpot::types::adapt::xtensor::convertToAtomMatrix(positions),
                 m_atomTypes, m_box);

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

  xt::xtensor<bool, 1> expandFixedMask(const xt::xtensor<bool, 1> &mask,
                                       size_t numMolecules) {
    if (mask.size() != numMolecules && mask.size() != 0) {
      throw std::runtime_error("Incorrect size of the fixed mask tensor.");
    }
    xt::xtensor<bool, 1> expandedMask = xt::zeros<bool>({numMolecules * 3});
    for (size_t i = 0; i < numMolecules; ++i) {
      bool maskValue = (mask.size() == numMolecules) ? mask(i) : false;
      std::fill_n(expandedMask.begin() + i * 3, 3, maskValue);
    }
    return expandedMask;
  }
};

} // namespace pot
} // namespace xts

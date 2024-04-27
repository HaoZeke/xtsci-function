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
#include "xtensor/xtensor_forward.hpp"
#include "xtsci/func/base.hpp"
#include <xtensor/xadapt.hpp>
#include <xtensor/xindex_view.hpp>

namespace xts {
namespace pot {

template <typename ScalarType = double>
class XTPot : public func::ObjectiveFunction<ScalarType> {
  friend class TestXTPot; // Make a test class a friend
public:
  XTPot(
      std::shared_ptr<rgpot::Potential> pot,
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

  xt::xtensor<ScalarType, 1> get_free(const xt::xarray<ScalarType> &pos) const {
    return xt::filter(xt::flatten(pos), m_free);
  }

protected: // Useful to test the damn thing
  xt::xtensor<ScalarType, 2>
  reconstruct_full(const xt::xarray<ScalarType> &free_x) const {
    std::array<std::size_t, 2> shape = {m_atomTypes.size(), 3};
    xt::xtensor<double, 2> allpos    = xt::reshape_view(m_basepos, shape);
    // Flatten all structures for simpler indexing
    auto flat_indices = xt::from_indices(xt::nonzero(m_free));
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
  const xt::xtensor<bool, 1> m_free{!this->m_isFixed};
  xt::xtensor<double, 2> m_basepos;

  ScalarType compute(const xt::xarray<ScalarType> &free_x) const override {
    xt::xtensor<double, 2> positions = this->reconstruct_full(free_x);

    auto [energy, forces] = (*m_pot)(
        rgpot::types::adapt::xtensor::convertToAtomMatrix(positions),
        m_atomTypes, m_box);

    return energy;
  }

  std::optional<xt::xarray<ScalarType>>
  compute_gradient(const xt::xarray<ScalarType> &free_x) const override {
    xt::xtensor<double, 2> positions = this->reconstruct_full(free_x);

    auto [energy, forces] = (*m_pot)(
        rgpot::types::adapt::xtensor::convertToAtomMatrix(positions),
        m_atomTypes, m_box);

    std::array<size_t, 2> shape = {m_atomTypes.size(), 3};
    auto forces_xtensor
        = xt::adapt(forces.data(), shape, xt::layout_type::row_major);

    // Convert forces to gradient
    auto gradient       = -1.0 * forces_xtensor;
    auto flat_free_grad = get_free(xt::flatten(gradient));
    return flat_free_grad;
  }

  xt::xtensor<bool, 1>
  expandFixedMask(const xt::xtensor<bool, 1> &mask, size_t numMolecules) {
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

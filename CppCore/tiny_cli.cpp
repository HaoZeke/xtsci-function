// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include <fmt/format.h>
#include <fmt/ostream.h>

#include <cmath>
#include <cstdlib>

#include <iostream>
#include <random>

#include "xtensor-fmt/misc.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xlayout.hpp"
#include "xtensor/xvectorize.hpp"

#include "xtsci/func/plot_aid.hpp"
#include "xtsci/func/trial/D2/branin.hpp"
#include "xtsci/func/trial/D2/eggholder.hpp"
#include "xtsci/func/trial/D2/himmelblau.hpp"
#include "xtsci/func/trial/D2/mullerbrown.hpp"
#include "xtsci/func/trial/D2/rosenbrock.hpp"

#include "rgpot/CuH2/CuH2Pot.hpp"
#include "xtsci/pot/base.hpp"

#include "Helpers.hpp"
#include "include/BaseTypes.hpp"
#include "include/FormatConstants.hpp"
#include "include/ReadCon.hpp"
#include "include/helpers/StringHelpers.hpp"

xt::xtensor<double, 2> extract_positions(const yodecon::types::ConFrameVec& frame) {
    size_t n_atoms = frame.x.size();
    std::array<size_t, 2> shape = {static_cast<size_t>(n_atoms), 3};

    xt::xtensor<double, 2> positions = xt::empty<double>(shape);
    for (size_t i = 0; i < n_atoms; ++i) {
        positions(i, 0) = frame.x[i];
        positions(i, 1) = frame.y[i];
        positions(i, 2) = frame.z[i];
    }

    return positions;
}

xt::xtensor<double, 1> normalize(const xt::xtensor<double, 1> &vec) {
  double norm = xt::linalg::norm(vec);
  if (norm == 0.0) {
    throw std::runtime_error("Cannot normalize a zero vector");
  }
  return vec / norm;
}

xt::xtensor<double, 2>
peturb_positions(const xt::xtensor<double, 2> &base_positions,
                 const xt::xtensor<int, 1> &atmNumVec, double hcu_dist,
                 double hh_dist) {
  xt::xtensor<double, 2> positions = base_positions;
  std::vector<size_t> hIndices, cuIndices;

  for (size_t i = 0; i < atmNumVec.size(); ++i) {
    if (atmNumVec(i) == 1) { // Hydrogen atom
      hIndices.push_back(i);
    } else if (atmNumVec(i) == 29) { // Copper atom
      cuIndices.push_back(i);
    } else {
      throw std::runtime_error("Unexpected atomic number");
    }
  }

  if (hIndices.size() != 2) {
    throw std::runtime_error("Expected exactly two hydrogen atoms");
  }

  // Compute the midpoint of the hydrogens
  auto hMidpoint =
      (xt::row(positions, hIndices[0]) + xt::row(positions, hIndices[1])) / 2;

  // TODO(rg): This is buggy in cuh2vizR!! (maybe)
  // Compute the HH direction
  xt::xtensor<double, 1> hh_direction;
  size_t h1_idx, h2_idx;
  if (positions(hIndices[0], 0) < positions(hIndices[1], 0)) {
    hh_direction = normalize(xt::row(positions, hIndices[1]) -
                             xt::row(positions, hIndices[0]));
    h1_idx = hIndices[0];
    h2_idx = hIndices[1];
  } else {
    hh_direction = normalize(xt::row(positions, hIndices[0]) -
                             xt::row(positions, hIndices[1]));
    h1_idx = hIndices[1];
    h2_idx = hIndices[0];
  }

  // Set the new position of the hydrogens using the recorded indices
  xt::row(positions, h1_idx) = hMidpoint - (0.5 * hh_dist) * hh_direction;
  xt::row(positions, h2_idx) = hMidpoint + (0.5 * hh_dist) * hh_direction;

  // Find the z-coordinate of the topmost Cu layer
  double maxCuZ = std::numeric_limits<double>::lowest();
  for (auto cuIndex : cuIndices) {
    maxCuZ = std::max(maxCuZ, positions(cuIndex, 2));
  }

  // Compute the new z-coordinate for the H atoms
  double new_z = maxCuZ + hcu_dist;

  // Update the z-coordinates of the H atoms
  for (auto hIndex : hIndices) {
    positions(hIndex, 2) = new_z;
  }

  return positions;
}

std::pair<double, double>
calculateDistances(const xt::xtensor<double, 2> &positions,
                   const xt::xtensor<int, 1> &atmNumVec) {
  std::vector<size_t> hIndices, cuIndices;
  for (size_t i = 0; i < atmNumVec.size(); ++i) {
    if (atmNumVec(i) == 1) { // Hydrogen atom
      hIndices.push_back(i);
    } else if (atmNumVec(i) == 29) { // Copper atom
      cuIndices.push_back(i);
    } else {
      throw std::runtime_error("Unexpected atomic number");
    }
  }

  if (hIndices.size() != 2) {
    throw std::runtime_error("Expected exactly two hydrogen atoms");
  }

  // Calculate the distance between Hydrogen atoms
  double hDistance =
      xt::linalg::norm(xt::view(positions, hIndices[0], xt::all()) -
                       xt::view(positions, hIndices[1], xt::all()));

  // Calculate the midpoint of Hydrogen atoms
  xt::xtensor<double, 1> hMidpoint =
      (xt::view(positions, hIndices[0], xt::all()) +
       xt::view(positions, hIndices[1], xt::all())) /
      2.0;

  // Find the z-coordinate of the topmost Cu layer
  double maxCuZ = std::numeric_limits<double>::lowest();
  for (size_t cuIndex : cuIndices) {
    maxCuZ = std::max(maxCuZ, positions(cuIndex, 2));
  }

  double cuSlabDist = positions(hIndices[0], 2) - maxCuZ;

  return std::make_pair(hDistance, cuSlabDist);
}

int main(int argc, char *argv[]) {
  // Eat warnings, also safer
  static_cast<void>(argc);
  static_cast<void>(argv);
  // xts::func::trial::D2::Rosenbrock<double> rosen;
  // xts::func::trial::D2::Himmelblau<double> himmelblau;
  // // xts::func::trial::D2::QuadraticFunction<double> quadratic;
  // xts::func::trial::D2::Eggholder<double> eggholder;
  // xts::func::trial::D2::MullerBrown<double> mullerbrown;
  // xts::func::trial::D2::Branin<double> branin;

  // clang-format off
  // Rosenbrock
  // xt::xarray<double> min1 = xt::row(rosen.minima, 0); // {0, 0}
  // fmt::print("Rosenbrock Minima: {} with fval {}\n", min1, rosen(min1));
  // clang-format on

  // Grid test
  // auto z_mesh = xts::func::eval_on_grid2D<double>({-2, 2, 100}, {-2, 2, 100}
  // ,rosen); for (auto i : z_mesh) {
  //   fmt::print("{} ", i);
  // }

  // clang-format off
  // Muller Brown
  // xt::xarray<double> min1 = xt::row(mullerbrown.minima, 0);
  // {-0.558, 1.441}
  // xt::xarray<double> min2 = xt::row(mullerbrown.minima, 1);
  // {0.623, 0.028}
  // xt::xarray<double> min3 = xt::row(mullerbrown.minima, 2);
  // {-0.050, 0.466}
  // fmt::print("Minima: {}, {}, {}\n", min1, min2, min3);
  // clang-format on

  // clang-format off
  // Himmelblau
  // xt::xarray<double> min1 =
  // xt::row(himmelblau.minima, 0); // {3, 2};
  // xt::xarray<double> min2 =
  // xt::row(himmelblau.minima, 1); // {-2.805118, 3.131312};
  // xt::xarray<double> min3 =
  // xt::row(himmelblau.minima, 2); // {-3.779310, -3.283186};
  // xt::xarray<double> min4 =
  // xt::row(himmelblau.minima, 3); // {3.584428, -1.848126};
  // fmt::print("Himmelblau Minima:\n{}, {}\n{}, {}\n{},
  // {}\n{}, {}\n", min1,
  //            himmelblau(min1), min2, himmelblau(min2),
  //            min3, himmelblau(min3),
  //            min4, himmelblau(min4));
  // clang-format on

  // clang-format off
  // npz test
  // xts::func::npz_on_grid2D<double>({-2, 2, 100}, {-2, 2, 100}, rosen,
  //                                  "rosen.npz");
  // xts::func::npz_on_grid2D<double>({-5, 5, 400}, {-5, 5, 400}, himmelblau,
  //                                  "himmelblau.npz");
  // xts::func::npz_on_grid2D<double>({-1.5, 1.2, 400}, {-0.2, 2.0, 400},
  //                                  mullerbrown, "mullerbrown.npz");
  // xts::func::npz_on_grid2D<double>({-512, 612, 400}, {-512, 613, 400},
  //                                  eggholder, "eggholder.npz");
  // xts::func::npz_on_grid2D<double>({-5, 18, 400}, {-5, 20, 400}, branin,
  //                                  "branin.npz");
  // clang-format on
  xt::xtensor<bool, 1> fixedMask = {true, false};
  xts::func::trial::D2::Branin<double> branin_fixed(fixedMask);
  xt::xarray<double> x = {1.0, 1.0};

  fmt::print(
      "Branin function at 1, 1 is {} and gradient is {} with mask ({} {})",
      branin_fixed(x), *branin_fixed.gradient(x), branin_fixed.m_isFixed[0],
      branin_fixed.m_isFixed[1]);

  auto cuh2pot = std::make_shared<rgpot::CuH2Pot>();
  // xt::xtensor<int, 1> atomTypes{{29, 29, 1, 1}};
  // xt::xtensor<double, 2> boxMatrix{
  //     {15, 0, 0},
  //     {0, 20, 0},
  //     {0, 0, 30},
  // };

  // xt::xtensor<double, 2> positions{
  //     {0.63940268750835, 0.90484742551374, 6.97516498544584}, // Cu
  //     {3.19652040936288, 0.90417430354811, 6.97547796369474}, // Cu
  //     {8.98363230369760, 9.94703496017833, 7.83556854923689}, // H
  //     {7.64080177576300, 9.94703114803832, 7.83556986121272}, // H
  // };


  std::vector<std::string> fconts =
      yodecon::helpers::file::read_con_file("cuh2.con");

  auto frame = yodecon::create_single_con<yodecon::types::ConFrameVec>(fconts);

  auto positions = extract_positions(frame);
  auto atomTypes = xt::adapt(yodecon::symbols_to_atomic_numbers(frame.symbol));
  auto boxMatrix = frame.boxl;

  xts::pot::XTPot<double> objFunc(cuh2pot, atomTypes, xt::adapt(boxMatrix));

  double energy = objFunc(xt::ravel<xt::layout_type::row_major>(positions));
  auto grad =
      objFunc.gradient(xt::ravel<xt::layout_type::row_major>(positions));
  // Reference:
  // Got energy -2.7114093289369636
  //  Forces:
  //      1.49194 -0.000392731  0.000182606
  //     -1.49194  0.000392731 -0.000182606
  //     -4.91186 -1.39442e-05    4.799e-06
  //      4.91186  1.39442e-05   -4.799e-06%
  auto [hdist, cusdist] = calculateDistances(positions, atomTypes);
  fmt::print("HH distance {}\n CuSlab distance {}\n", hdist, cusdist);

  auto new_positions = peturb_positions(positions, atomTypes, cusdist, hdist);
  fmt::print("New positions:\n{}\n", fmt::streamed(new_positions));
  fmt::print("Old positions:\n{}\n", fmt::streamed(positions));

  fmt::print("Got energy {}\n", energy);
  fmt::print("Got gradient {}\n", fmt::streamed(*grad));
  double new_energy =
      objFunc(xt::ravel<xt::layout_type::row_major>(new_positions));
  auto new_grad =
      objFunc.gradient(xt::ravel<xt::layout_type::row_major>(new_positions));
  fmt::print("Got new energy {}\n", new_energy);
  fmt::print("Got gradient {}\n", fmt::streamed(*new_grad));

  auto energyFunc = [&objFunc, &positions, &atomTypes](
                        double hh_dist, double cu_slab_dist) -> double {
    auto perturbed_positions =
        peturb_positions(positions, atomTypes, cu_slab_dist, hh_dist);
    return objFunc(xt::ravel<xt::layout_type::row_major>(perturbed_positions));
  };

  xts::func::npz_on_grid2D<double>({0.4, 3.2, 60}, {0.05, 5.1, 60}, energyFunc,
                                   "cuh2_grid.npz");

  return EXIT_SUCCESS;
}

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
#include "xtensor/xshape.hpp"
#include "xtensor/xvectorize.hpp"

#include "xtsci/func/plot_aid.hpp"
#include "xtsci/func/trial/D2/branin.hpp"
#include "xtsci/func/trial/D2/eggholder.hpp"
#include "xtsci/func/trial/D2/himmelblau.hpp"
#include "xtsci/func/trial/D2/mullerbrown.hpp"
#include "xtsci/func/trial/D2/rosenbrock.hpp"

#include "rgpot/CuH2/CuH2Pot.hpp"
#include "rgpot/CuH2/cuh2Utils.hpp"
#include "xtsci/pot/base.hpp"

#include "readCon/include/BaseTypes.hpp"
#include "readCon/include/FormatConstants.hpp"
#include "readCon/include/ReadCon.hpp"
#include "readCon/include/helpers/StringHelpers.hpp"

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

  auto positions = rgpot::cuh2::utils::xts::extract_positions(frame);
  auto atomNumbersVec = yodecon::symbols_to_atomic_numbers(frame.symbol);
  xt::xtensor<int, 1> atomTypes = xt::empty<int>({atomNumbersVec.size()});
  for (size_t i = 0; i < atomNumbersVec.size(); ++i) {
    atomTypes(i) = atomNumbersVec[i];
  }
  // std::array<size_t, 2> shape = {1, 3};
  xt::xtensor<double, 2> boxMatrix = xt::empty<double>(xt::shape({1, 3}));
  for (size_t i = 0; i < 3; ++i) {
    boxMatrix(0, i) = frame.boxl[i];
  }
  xt::xtensor<bool, 1> booltypes = xt::adapt(frame.is_fixed);

  fmt::print("\nPutting in {}\n", fmt::streamed(booltypes));
  xts::pot::XTPot<double> objFunc(cuh2pot, atomTypes, boxMatrix, booltypes);

  double energy = objFunc(positions);
  auto grad = objFunc.gradient(positions);
  // Reference:
  // Got energy -2.7114093289369636
  //  Forces:
  //      1.49194 -0.000392731  0.000182606
  //     -1.49194  0.000392731 -0.000182606
  //     -4.91186 -1.39442e-05    4.799e-06
  //      4.91186  1.39442e-05   -4.799e-06%
  auto [hdist, cusdist] = rgpot::cuh2::utils::xts::calculateDistances(positions, atomTypes);
  fmt::print("HH distance {}\n CuSlab distance {}\n", hdist, cusdist);

  // auto new_positions = rgpot::cuh2::utils::xts::perturb_positions(positions,
  // atomTypes, cusdist, hdist); fmt::print("New positions:\n{}\n",
  // fmt::streamed(new_positions)); fmt::print("Old positions:\n{}\n",
  // fmt::streamed(positions));

  fmt::print("Got energy {}\n", energy);
  fmt::print("Got gradient {}\n", fmt::streamed(*grad));
  // double new_energy =
  //     objFunc(xt::ravel<xt::layout_type::row_major>(new_positions));
  // auto new_grad =
  //     objFunc.gradient(xt::ravel<xt::layout_type::row_major>(new_positions));
  // fmt::print("Got new energy {}\n", new_energy);
  // fmt::print("Got gradient {}\n", fmt::streamed(*new_grad));

  // auto energyFunc = [&objFunc, &positions, &atomTypes](
  //                       double hh_dist, double cu_slab_dist) -> double {
  //   auto perturbed_positions =
  //       rgpot::cuh2::utils::xts::perturb_positions(positions, atomTypes,
  //       cu_slab_dist, hh_dist);
  //   return
  //   objFunc(xt::ravel<xt::layout_type::row_major>(perturbed_positions)) -
  //          (-697.311695);
  // };

  // xts::func::npz_on_grid2D<double>({0.4, 3.2, 60}, {-0.05, 3.1, 60},
  // energyFunc,
  //                                  "cuh2_grid.npz");

  return EXIT_SUCCESS;
}

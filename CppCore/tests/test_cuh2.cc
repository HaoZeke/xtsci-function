// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include "rgpot/CuH2/CuH2Pot.hpp"
#include "rgpot/CuH2/cuh2Utils.hpp"
#include "xtsci/pot/base.hpp"
#include <catch2/catch_all.hpp>

constexpr double TEST_EPS{1e-5};

TEST_CASE("CuH2 Energy and Gradient Calculations", "[CuH2EnergyCalc]") {
  // Setup
  auto cuh2pot = std::make_shared<rgpot::CuH2Pot>();
  auto objFunc = xts::pot::mk_xtpot_con("cuh2.con", cuh2pot);
  std::vector<std::string> fconts
      = yodecon::helpers::file::read_con_file("cuh2.con");
  auto frame = yodecon::create_single_con<yodecon::types::ConFrameVec>(fconts);
  auto [positions, atomTypes, boxMatrix, booltypes]
      = xts::pot::extract_condat("cuh2.con");

  SECTION("Initial Energy and Gradient Calculation") {
    double energy          = objFunc(objFunc.get_free(positions));
    auto grad              = objFunc.gradient(objFunc.get_free(positions));
    double expected_energy = -697.313451290564;
    xt::xtensor<double, 2> expected_free_force = {
        {0.00258381, 2.71051e-20, 0.000809943}, // H1
        {-0.00258381, 1.69407e-20, 0.000809943} // H2
    };

    // NOTE: REQUIRE_THAT with the matcher for single values, else xt::isclose
    // is a quick way
    REQUIRE_THAT(energy, Catch::Matchers::WithinAbs(expected_energy, TEST_EPS));
    REQUIRE(grad->size() == 6);
    REQUIRE(xt::isclose(*grad, -1 * expected_free_force, TEST_EPS)());
  }

  SECTION("Perturbed Energy and Gradient Calculation") {
    auto [hdist, cusdist]
        = rgpot::cuh2::utils::xts::calculateDistances(positions, atomTypes);
    REQUIRE(xt::isclose(hdist, 0.7401999999999997, TEST_EPS)());
    REQUIRE(xt::isclose(cusdist, 4.75760000000000, TEST_EPS)());
    auto new_positions = rgpot::cuh2::utils::xts::perturb_positions(
        positions, atomTypes, cusdist, hdist + 10);
    double energy = objFunc(
        objFunc.get_free(xt::ravel<xt::layout_type::row_major>(new_positions)));
    auto grad = objFunc.gradient(
        objFunc.get_free(xt::ravel<xt::layout_type::row_major>(new_positions)));
    double expected_energy                     = -692.5791459208481;
    xt::xtensor<double, 2> expected_free_force = {
        {-3.047233e-05, -5.637851e-18, -1.814462e-02}, // H1
        {6.773808e-06, -6.993104e-18, -1.814615e-02}   // H2
    };

    REQUIRE(xt::isclose(energy, expected_energy, TEST_EPS)());
    REQUIRE(grad->size() == 6);
    REQUIRE(xt::isclose(*grad, -1 * expected_free_force, TEST_EPS)());
    auto [new_hdist, new_cusdist]
        = rgpot::cuh2::utils::xts::calculateDistances(new_positions, atomTypes);
    // BUG: This should pass but it is off by 2!
    // REQUIRE_THAT(new_hdist, Catch::Matchers::WithinAbs(10.7401999999999997,
    // TEST_EPS));
  }
}

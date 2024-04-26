// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include "xtsci/pot/base.hpp"
#include <catch2/catch_all.hpp>

constexpr double TEST_EPS{1e-5};

class TestXTPot : public xts::pot::XTPot<double> {
public:
  using XTPot<double>::XTPot;
  xt::xtensor<double, 2>
  test_reconstruct_full(const xt::xarray<double> &free_x) {
    return this->reconstruct_full(free_x);
  }
};

TEST_CASE("Test reconstruct_full functionality", "[XTPot]") {
  // Setup
  xt::xarray<double> base_pos = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  xt::xarray<int> atomTypes   = {1, 2};
  xt::xarray<double> boxMatrix
      = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
  xt::xarray<bool> fixedMask = {false, true};

  TestXTPot pot(nullptr, base_pos, atomTypes, boxMatrix, fixedMask);
  xt::xarray<double> free_x = {0.1, 0.2, 0.3};

  // Execute
  auto result = pot.test_reconstruct_full(free_x);

  // Validate
  REQUIRE(xt::isclose(result(0, 0), 0.1, TEST_EPS)());
  REQUIRE(xt::isclose(result(0, 1), 0.2, TEST_EPS)());
  REQUIRE(xt::isclose(result(0, 2), 0.3, TEST_EPS)());
  REQUIRE(xt::isclose(result(1, 0), 4.0, TEST_EPS)());
  REQUIRE(xt::isclose(result(1, 1), 5.0, TEST_EPS)());
  REQUIRE(xt::isclose(result(1, 2), 6.0, TEST_EPS)());
}

TEST_CASE("Test reconstruct_full functionality with 6 atoms", "[XTPot]") {
  // Setup: 6 atoms, 2 free
  xt::xarray<double> base_pos = {
      {1.0, 2.0, 3.0},    // Atom 1 (fixed)
      {4.0, 5.0, 6.0},    // Atom 2 (fixed)
      {7.0, 8.0, 9.0},    // Atom 3 (fixed)
      {10.0, 11.0, 12.0}, // Atom 4 (fixed)
      {13.0, 14.0, 15.0}, // Atom 5 (free)
      {16.0, 17.0, 18.0}  // Atom 6 (free)
  };
  xt::xarray<int> atomTypes = {1, 2, 3, 4, 5, 6};
  xt::xarray<double> boxMatrix
      = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
  xt::xarray<bool> fixedMask = {true, true, true, true, false, false};

  TestXTPot pot(nullptr, base_pos, atomTypes, boxMatrix, fixedMask);
  xt::xarray<double> free_x = {
      {0.1, 0.2, 0.3},   // Updates for Atom 5
      {19.0, 20.0, 21.0} // Updates for Atom 6
  };

  // Execute
  auto result = pot.test_reconstruct_full(free_x);

  // Validate
  REQUIRE(xt::isclose(result(4, 0), 0.1, TEST_EPS)());
  REQUIRE(xt::isclose(result(4, 1), 0.2, TEST_EPS)());
  REQUIRE(xt::isclose(result(4, 2), 0.3, TEST_EPS)());
  REQUIRE(xt::isclose(result(5, 0), 19.0, TEST_EPS)());
  REQUIRE(xt::isclose(result(5, 1), 20.0, TEST_EPS)());
  REQUIRE(xt::isclose(result(5, 2), 21.0, TEST_EPS)());
}

TEST_CASE(
    "Test get_free functionality with mixed fixed and free atoms", "[XTPot]") {
  // Setup: 6 atoms, 4 fixed and 2 free
  xt::xarray<double> base_pos = {
      {1.0, 2.0, 3.0},    // Atom 1 (fixed)
      {4.0, 5.0, 6.0},    // Atom 2 (fixed)
      {7.0, 8.0, 9.0},    // Atom 3 (fixed)
      {10.0, 11.0, 12.0}, // Atom 4 (fixed)
      {13.0, 14.0, 15.0}, // Atom 5 (free)
      {16.0, 17.0, 18.0}  // Atom 6 (free)
  };
  xt::xarray<int> atomTypes = {1, 2, 3, 4, 5, 6};
  xt::xarray<double> boxMatrix
      = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
  xt::xarray<bool> fixedMask = {true, true, true, true, false, false};

  TestXTPot pot(nullptr, base_pos, atomTypes, boxMatrix, fixedMask);

  // Execute
  xt::xarray<double> pos
      = xt::flatten(base_pos); // Flatten base_pos for simplicity
  auto free_positions = pot.get_free(pos);

  // Validate
  xt::xarray<double> expected_free_positions
      = {13.0, 14.0, 15.0, 16.0, 17.0, 18.0};
  REQUIRE(
      xt::all(xt::isclose(free_positions, expected_free_positions, TEST_EPS)));
}

// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include "xtsci/func/trial/D2/himmelblau.hpp"
#include <catch2/catch_all.hpp>

TEST_CASE("Himmelblau Function properties", "[Himmelblau]") {
  using Scalar = double;
  xts::func::trial::D2::Himmelblau<Scalar> himmelblauFunc;
  xt::xarray<Scalar> x = xt::xarray<Scalar>::from_shape({2});

  SECTION("Value at first minimum") {
    x = {3.0, 2.0};
    REQUIRE_THAT(himmelblauFunc(x), Catch::Matchers::WithinAbs(0.0, 1e-4));
  }

  SECTION("Value at second minimum") {
    x = {-2.805118, 3.131312};
    REQUIRE_THAT(himmelblauFunc(x), Catch::Matchers::WithinAbs(0.0, 1e-4));
  }

  SECTION("Value at third minimum") {
    x = {-3.779310, -3.283186};
    REQUIRE_THAT(himmelblauFunc(x), Catch::Matchers::WithinAbs(0.0, 1e-4));
  }

  SECTION("Value at fourth minimum") {
    x = {3.584428, -1.848126};
    REQUIRE_THAT(himmelblauFunc(x), Catch::Matchers::WithinAbs(0.0, 1e-4));
  }

  SECTION("Gradient at first minimum is zero vector") {
    x                       = {3.0, 2.0};
    xt::xarray<Scalar> grad = himmelblauFunc.gradient(x).value();
    REQUIRE_THAT(grad(0), Catch::Matchers::WithinAbs(0.0, 1e-4));
    REQUIRE_THAT(grad(1), Catch::Matchers::WithinAbs(0.0, 1e-4));
  }

  SECTION("Hessian at first minimum is positive definite") {
    x                       = {3.0, 2.0};
    xt::xarray<Scalar> hess = himmelblauFunc.hessian(x).value();
    // For a 2x2 matrix, it's positive definite if det > 0 and top left value >
    // 0
    Scalar determinant = hess(0, 0) * hess(1, 1) - hess(0, 1) * hess(1, 0);
    REQUIRE(determinant > 0.0);
    REQUIRE(hess(0, 0) > 0.0);
  }

  SECTION("Value at arbitrary points") {
    REQUIRE_THAT(himmelblauFunc({1, 1}), Catch::Matchers::WithinAbs(106, 1e-4));
    REQUIRE_THAT(
        himmelblauFunc({0.5, 0.5}), Catch::Matchers::WithinAbs(144.125, 1e-4));
    REQUIRE_THAT(
        himmelblauFunc({0.623, 0.028}),
        Catch::Matchers::WithinAbs(152.674455823, 1e-4));
  }

  SECTION("Gradient at arbitrary point") {
    x                       = {0.623, 0.028};
    xt::xarray<Scalar> grad = himmelblauFunc.gradient(x).value();
    REQUIRE_THAT(grad(0), Catch::Matchers::WithinAbs(-39.1274386386, 1e-4));
    REQUIRE_THAT(grad(1), Catch::Matchers::WithinAbs(-21.8818528312, 1e-4));
  }

  SECTION("Hessian at an arbitrary point") {
    x                       = {0.623, 0.028};
    xt::xarray<Scalar> hess = himmelblauFunc.hessian(x).value();
    REQUIRE_THAT(hess(0, 0), Catch::Matchers::WithinAbs(-37.23044967651, 1e-4));
    REQUIRE_THAT(hess(0, 1), Catch::Matchers::WithinAbs(2.60399961472, 1e-4));
    REQUIRE_THAT(hess(1, 0), Catch::Matchers::WithinAbs(2.60399961472, 1e-4));
    REQUIRE_THAT(hess(1, 1), Catch::Matchers::WithinAbs(-23.49859046936, 1e-4));
  }
}

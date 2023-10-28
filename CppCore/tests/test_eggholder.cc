// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include "xtsci/func/trial/D2/eggholder.hpp"
#include <catch2/catch_all.hpp>

TEST_CASE("Eggholder Function properties", "[Eggholder]") {
  using Scalar = double;
  xts::func::trial::D2::Eggholder<Scalar> eggholderFunc;
  xt::xarray<Scalar> x = xt::xarray<Scalar>::from_shape({2});

  SECTION("Value at the minimum") {
    x = {512, 404.2319};
    REQUIRE_THAT(eggholderFunc(x), Catch::Matchers::WithinAbs(-959.6407, 1e-4));
  }

  // TODO(rgoswami): Fix this
  // SECTION("Gradient at the minimum is zero vector") {
  //   x = {512, 404.2319};
  //   xt::xarray<Scalar> grad = eggholderFunc.gradient(x).value();
  //   REQUIRE_THAT(grad(0), Catch::Matchers::WithinAbs(0.0, 1e-4));
  //   REQUIRE_THAT(grad(1), Catch::Matchers::WithinAbs(0.0, 1e-4));
  // }

  // SECTION("Hessian at first minimum is positive definite") {
  //   x = {512, 404.2319};
  //   xt::xarray<Scalar> hess = eggholderFunc.hessian(x).value();
  //   // For a 2x2 matrix, it's positive definite if det > 0 and top left value >
  //   // 0
  //   Scalar determinant = hess(0, 0) * hess(1, 1) - hess(0, 1) * hess(1, 0);
  //   REQUIRE(determinant > 0.0);
  //   REQUIRE(hess(0, 0) > 0.0);
  // }

  SECTION("Value at arbitrary points") {
    REQUIRE_THAT(eggholderFunc({1, 1}), Catch::Matchers::WithinAbs(-30.7614121992, 1e-4));
    REQUIRE_THAT(eggholderFunc({0.5, 0.5}),
                 Catch::Matchers::WithinAbs(-28.1381212887, 1e-4));
    REQUIRE_THAT(eggholderFunc({0.623, 0.028}),
                 Catch::Matchers::WithinAbs(-26.7590279212, 1e-4));
  }

  // TODO(rgoswami): Fix this
  // SECTION("Gradient at arbitrary point") {
  //   x = {0.623, 0.028};
  //   xt::xarray<Scalar> grad = eggholderFunc.gradient(x).value();
  //   REQUIRE_THAT(grad(0), Catch::Matchers::WithinAbs(-1.87816053485, 1e-4));
  //   REQUIRE_THAT(grad(1), Catch::Matchers::WithinAbs(-3.42783757618, 1e-4));
  // }

  // SECTION("Hessian at an arbitrary point") {
  //   x = {0.623, 0.028};
  //   xt::xarray<Scalar> hess = eggholderFunc.hessian(x).value();
  //   REQUIRE_THAT(hess(0, 0), Catch::Matchers::WithinAbs(0.1712307929993, 1e-4));
  //   REQUIRE_THAT(hess(0, 1), Catch::Matchers::WithinAbs(-0.0107824802399, 1e-4));
  //   REQUIRE_THAT(hess(1, 0), Catch::Matchers::WithinAbs(-0.0107824802399, 1e-4));
  //   REQUIRE_THAT(hess(1, 1), Catch::Matchers::WithinAbs(0.0514380931854, 1e-4));
  // }
}

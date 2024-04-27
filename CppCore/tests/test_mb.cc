// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include "xtsci/func/trial/D2/mullerbrown.hpp"
#include <catch2/catch_all.hpp>

TEST_CASE("MullerBrown Function properties", "[MullerBrown]") {
  using Scalar = double;
  xts::func::trial::D2::MullerBrown<Scalar> mullerBrown;
  xt::xarray<Scalar> x = xt::xarray<Scalar>::from_shape({2});

  SECTION("Value at first minimum") {
    x = {-0.558, 1.442};
    REQUIRE_THAT(
        mullerBrown(x), Catch::Matchers::WithinAbs(-146.69948920058778, 1e-4));
  }

  SECTION("Value at second minimum") {
    x = {0.623, 0.028};
    REQUIRE_THAT(
        mullerBrown(x), Catch::Matchers::WithinAbs(-108.16665005353302, 1e-4));
  }

  SECTION("Value at third minimum") {
    x = {-0.050, 0.466};
    REQUIRE_THAT(
        mullerBrown(x), Catch::Matchers::WithinAbs(-80.76746772526472, 1e-4));
  }

  SECTION("Value at first saddle point") {
    x = {0.212, 0.293};
    REQUIRE_THAT(
        mullerBrown(x), Catch::Matchers::WithinAbs(-72.24891965936473, 1e-4));
  }

  SECTION("Value at second saddle point") {
    x = {-0.822, 0.624};
    REQUIRE_THAT(
        mullerBrown(x), Catch::Matchers::WithinAbs(-40.66484530104902, 1e-4));
  }

  // TODO(rgoswami): Look into this, though it should be 0, doesn't seem to be
  // the case SECTION("Gradient at first minimum") {
  //     x = {-0.558, 1.442};
  //     xt::xarray<Scalar> grad = mullerBrown.gradient(x).value();
  //     REQUIRE_THAT(grad(0), Catch::Matchers::WithinAbs(0.0, 1e-4));
  //     REQUIRE_THAT(grad(1), Catch::Matchers::WithinAbs(0.0, 1e-4));
  // }

  // SECTION("Gradient at second minimum") {
  //     x = {0.623, 0.028};
  //     xt::xarray<Scalar> grad = mullerBrown.gradient(x).value();
  //     REQUIRE_THAT(grad(0), Catch::Matchers::WithinAbs(0.0, 1e-4));
  //     REQUIRE_THAT(grad(1), Catch::Matchers::WithinAbs(0.0, 1e-4));
  // }

  // SECTION("Gradient at third minimum") {
  //     x = {-0.050, 0.466};
  //     xt::xarray<Scalar> grad = mullerBrown.gradient(x).value();
  //     REQUIRE_THAT(grad(0), Catch::Matchers::WithinAbs(0.0, 0.2));
  //     REQUIRE_THAT(grad(1), Catch::Matchers::WithinAbs(0.0, 1e-4));
  // }

  SECTION("Value at arbitrary points") {
    x = {-1.050, 0.466};
    REQUIRE_THAT(
        mullerBrown(x), Catch::Matchers::WithinAbs(-26.4116181673, 1e-4));
    REQUIRE_THAT(
        mullerBrown({1.623, .38}),
        Catch::Matchers::WithinAbs(875.435521808, 1e-4));
  }

  SECTION("Gradient at arbitrary point") {
    x                       = {1.623, 0.38};
    xt::xarray<Scalar> grad = mullerBrown.gradient(x).value();
    REQUIRE_THAT(grad(0), Catch::Matchers::WithinAbs(3075.34429488, 1e-4));
    REQUIRE_THAT(grad(1), Catch::Matchers::WithinAbs(873.2579683, 1e-4));
  }

  // TODO(rgoswami): Fix this
  // SECTION("Hessian at an arbitrary point") {
  //     x = {1.623, 0.38};
  //     xt::xarray<Scalar> hess = mullerBrown.hessian(x).value();
  //     REQUIRE_THAT(hess(0, 0), Catch::Matchers::WithinAbs(11191.4224014,
  //     1e-4)); REQUIRE_THAT(hess(0, 1),
  //     Catch::Matchers::WithinAbs(2421.639873505, 1e-4)); REQUIRE_THAT(hess(1,
  //     0), Catch::Matchers::WithinAbs(2421.6398735, 1e-4));
  //     REQUIRE_THAT(hess(1, 1), Catch::Matchers::WithinAbs(613.918945312,
  //     1e-4));
  // }
}

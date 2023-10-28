// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include "xtsci/func/trial/D2/branin.hpp"
#include <catch2/catch_all.hpp>

TEST_CASE("Branin Function properties", "[Branin]") {
  using Scalar = double;
  xts::func::trial::D2::Branin<Scalar> branin;
  xt::xarray<Scalar> x = xt::xarray<Scalar>::from_shape({2});

  SECTION("Value at first minimum") {
    x = xt::row(branin.minima, 0);
    // TODO(rgoswami): Check?
    REQUIRE_THAT(branin(x), Catch::Matchers::WithinAbs(100.397887358, 1e-4));
  }

  SECTION("Value at second minimum") {
    x = xt::row(branin.minima, 1);
    REQUIRE_THAT(branin(x), Catch::Matchers::WithinAbs(0.397887, 1e-4));
  }

  SECTION("Value at third minimum") {
    x = xt::row(branin.minima, 2);
    REQUIRE_THAT(branin(x), Catch::Matchers::WithinAbs(0.397887357753, 1e-4));
  }

  SECTION("Value at arbitrary points") {
    xt::xarray<Scalar> x = {-1.050, 0.466};
    REQUIRE_THAT(branin(x), Catch::Matchers::WithinAbs(68.7642657479, 1e-4));
    REQUIRE_THAT(branin({1.623, 0.38}),
                 Catch::Matchers::WithinAbs(20.9044685173, 1e-4));
    REQUIRE_THAT(branin({0, 0}),
                 Catch::Matchers::WithinAbs(55.6021126423, 1e-4));
    REQUIRE_THAT(branin({1, 1}),
                 Catch::Matchers::WithinAbs(27.7029055485, 1e-4));
    REQUIRE_THAT(branin({2, 2}),
                 Catch::Matchers::WithinAbs(7.78270464815, 1e-4));
    REQUIRE_THAT(branin({-1, -1}),
                 Catch::Matchers::WithinAbs(91.2392440552, 1e-4));
  }

  SECTION("Gradients at some points") {
    xt::xarray<Scalar> grad1 = branin.gradient({0, 0}).value();
    xt::xarray<Scalar> grad2 = branin.gradient({1, 1}).value();
    xt::xarray<Scalar> grad3 = branin.gradient({2, 2}).value();
    xt::xarray<Scalar> grad4 = branin.gradient({-1, -1}).value();

    REQUIRE_THAT(grad1(0), Catch::Matchers::WithinAbs(-19.098593235, 1e-4));
    REQUIRE_THAT(grad1(1), Catch::Matchers::WithinAbs(-12, 1e-4));

    REQUIRE_THAT(grad2(0), Catch::Matchers::WithinAbs(-17.5125110149, 1e-4));
    REQUIRE_THAT(grad2(1), Catch::Matchers::WithinAbs(-7.07527017593, 1e-4));

    REQUIRE_THAT(grad3(0), Catch::Matchers::WithinAbs(-11.5979973972, 1e-4));
    REQUIRE_THAT(grad3(1), Catch::Matchers::WithinAbs(-2.6672783196, 1e-4));

    REQUIRE_THAT(grad4(0), Catch::Matchers::WithinAbs(-24.1853942871, 1e-4));
    REQUIRE_THAT(grad4(1), Catch::Matchers::WithinAbs(-17.4414682388, 1e-4));
  }

  SECTION("Hessian at an arbitrary point") {
    x = {0, 0};
    xt::xarray<Scalar> hess = branin.hessian(x).value();
    // TODO(rgoswami): Fix
    // REQUIRE_THAT(hess(0, 0), Catch::Matchers::WithinAbs(-1.43562507629, 1e-4));
    REQUIRE_THAT(hess(0, 1), Catch::Matchers::WithinAbs(3.18309879303, 1e-4));
    REQUIRE_THAT(hess(1, 0), Catch::Matchers::WithinAbs(3.18309879303, 1e-4));
    REQUIRE_THAT(hess(1, 1), Catch::Matchers::WithinAbs(2, 1e-4));

    x = {1, 1};
    hess = branin.hessian(x).value();
    // TODO(rgoswami): Fix
    // REQUIRE_THAT(hess(0, 0), Catch::Matchers::WithinAbs(0.19472694397, 1e-4));
    REQUIRE_THAT(hess(0, 1), Catch::Matchers::WithinAbs(2.6663607955, 1e-4));
    REQUIRE_THAT(hess(1, 0), Catch::Matchers::WithinAbs(2.6663607955, 1e-4));
    REQUIRE_THAT(hess(1, 1), Catch::Matchers::WithinAbs(2, 1e-4));

    x = {2, 2};
    hess = branin.hessian(x).value();
    // TODO(rgoswami): Fix
    // REQUIRE_THAT(hess(0, 0), Catch::Matchers::WithinAbs(6.99546986818, 1e-4));
    REQUIRE_THAT(hess(0, 1), Catch::Matchers::WithinAbs(2.14962278306, 1e-4));
    REQUIRE_THAT(hess(1, 0), Catch::Matchers::WithinAbs(2.14962278306, 1e-4));
    REQUIRE_THAT(hess(1, 1), Catch::Matchers::WithinAbs(2.0000000596, 1e-4));

    x = {-1, -1};
    hess = branin.hessian(x).value();
    // TODO(rgoswami): Fix
    // REQUIRE_THAT(hess(0, 0), Catch::Matchers::WithinAbs(6.16268634796, 1e-4));
    REQUIRE_THAT(hess(0, 1), Catch::Matchers::WithinAbs(3.69983673096, 1e-4));
    REQUIRE_THAT(hess(1, 0), Catch::Matchers::WithinAbs(3.69983673096, 1e-4));
    REQUIRE_THAT(hess(1, 1), Catch::Matchers::WithinAbs(1.99999904633, 1e-4));
  }
}

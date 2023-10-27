// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include "xtensor/xarray.hpp"

#include "xtsci/func/trial/D2/rosenbrock.hpp"

#include <catch2/catch_all.hpp>

TEST_CASE("Rosenbrock Function properties", "[Rosenbrock]") {
    using Scalar = double;
    xts::func::trial::D2::Rosenbrock<Scalar> rosenbrock;
    xt::xarray<Scalar> x = xt::xarray<Scalar>::from_shape({2});

    SECTION("Value at global minimum is zero") {
        x = {1.0, 1.0};
        REQUIRE_THAT(rosenbrock(x), Catch::Matchers::WithinAbs(0.0, 1e-4));
    }

    SECTION("Gradient at global minimum is zero vector") {
        x = {1.0, 1.0};
        xt::xarray<Scalar> grad = rosenbrock.gradient(x).value();
        REQUIRE_THAT(grad(0), Catch::Matchers::WithinAbs(0.0, 1e-4));
        REQUIRE_THAT(grad(1), Catch::Matchers::WithinAbs(0.0, 1e-4));
    }

    SECTION("Hessian at global minimum is positive definite") {
        x = {1.0, 1.0};
        xt::xarray<Scalar> hess = rosenbrock.hessian(x).value();
        // For a 2x2 matrix, it's positive definite if det > 0 and top left value > 0
        Scalar determinant = hess(0,0) * hess(1,1) - hess(0,1) * hess(1,0);
        REQUIRE(determinant > 0.0);
        REQUIRE(hess(0,0) > 0.0);
    }

    SECTION("Hessian at an arbitrary point") {
        x = {0.3, 2.0};
        xt::xarray<Scalar> hess = rosenbrock.hessian(x).value();
        REQUIRE_THAT(hess(0, 0), Catch::Matchers::WithinAbs(-689.9999999, 1e-4));
        REQUIRE_THAT(hess(0, 1), Catch::Matchers::WithinAbs(-120, 1e-4));
        REQUIRE_THAT(hess(1, 0), Catch::Matchers::WithinAbs(-120, 1e-4));
        REQUIRE_THAT(hess(1, 1), Catch::Matchers::WithinAbs(200, 1e-4));
    }

    SECTION("Value at arbitrary point") {
        x = {0.3, 4.0};
        REQUIRE_THAT(rosenbrock(x), Catch::Matchers::WithinAbs(1529.3, 1e-4));
    }

    SECTION("Gradient at arbitrary point") {
        x = {0.3, 1.0};
        xt::xarray<Scalar> grad = rosenbrock.gradient(x).value();
        REQUIRE_THAT(grad(0), Catch::Matchers::WithinAbs(-110.60, 1e-4));
        REQUIRE_THAT(grad(1), Catch::Matchers::WithinAbs(182, 1e-4));
    }
}

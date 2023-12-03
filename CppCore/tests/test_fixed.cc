// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include "xtsci/func/trial/D2/branin.hpp"
#include "xtsci/func/trial/D2/rosenbrock.hpp"
#include <catch2/catch_all.hpp>

TEST_CASE("Branin Function with Fixed Degrees of Freedom", "[BraninFixed]") {
  using Scalar = double;
  xt::xtensor<bool, 1> fixedMask = {true, false};

  xts::func::trial::D2::Branin<Scalar> branin_fixed(fixedMask);
  xts::func::trial::D2::Branin<Scalar> branin_standard;

  xt::xarray<Scalar> x = {1.0, 1.0};

  SECTION("Gradient with Fixed Degree of Freedom") {
    xt::xarray<Scalar> grad_fixed = branin_fixed.gradient(x).value();
    xt::xarray<Scalar> grad_standard = branin_standard.gradient(x).value();

    REQUIRE_THAT(grad_fixed(0),
                 Catch::Matchers::WithinAbs(
                     0.0, 1e-4)); // Fixed DOF should have zero gradient
    REQUIRE_THAT(
        grad_fixed(1),
        Catch::Matchers::WithinAbs(
            grad_standard(1), 1e-4)); // Non-fixed DOF should match standard
  }

  SECTION("Hessian with Fixed Degree of Freedom") {
    xt::xarray<Scalar> hess_fixed = branin_fixed.hessian(x).value();
    xt::xarray<Scalar> hess_standard = branin_standard.hessian(x).value();

    // Row and column corresponding to fixed DOF should be zero
    REQUIRE_THAT(hess_fixed(0, 0), Catch::Matchers::WithinAbs(0.0, 1e-4));
    REQUIRE_THAT(hess_fixed(0, 1), Catch::Matchers::WithinAbs(0.0, 1e-4));
    REQUIRE_THAT(hess_fixed(1, 0), Catch::Matchers::WithinAbs(0.0, 1e-4));

    // Other elements should match standard
    REQUIRE_THAT(hess_fixed(1, 1),
                 Catch::Matchers::WithinAbs(hess_standard(1, 1), 1e-4));
  }
}

TEST_CASE("Rosenbrock Function with Fixed Degrees of Freedom",
          "[RosenbrockFixed]") {
  using Scalar = double;
  xt::xtensor<bool, 1> fixedMask = {false, true};

  xts::func::trial::D2::Rosenbrock<Scalar> rosenbrock_fixed(fixedMask);
  xts::func::trial::D2::Rosenbrock<Scalar>
      rosenbrock_standard; // Without fixed mask

  xt::xarray<Scalar> x = {1.0, 1.0}; // Test point

  SECTION("Gradient with Fixed Degree of Freedom") {
    xt::xarray<Scalar> grad_fixed = rosenbrock_fixed.gradient(x).value();
    xt::xarray<Scalar> grad_standard = rosenbrock_standard.gradient(x).value();

    REQUIRE_THAT(grad_fixed(0),
                 Catch::Matchers::WithinAbs(grad_standard(0), 1e-4));
    REQUIRE_THAT(grad_fixed(1), Catch::Matchers::WithinAbs(0.0, 1e-4));
  }

  SECTION("Hessian with Fixed Degree of Freedom") {
    xt::xarray<Scalar> hess_fixed = rosenbrock_fixed.hessian(x).value();
    xt::xarray<Scalar> hess_standard = rosenbrock_standard.hessian(x).value();

    // Row and column corresponding to fixed DOF should be zero
    REQUIRE_THAT(hess_fixed(1, 0), Catch::Matchers::WithinAbs(0.0, 1e-4));
    REQUIRE_THAT(hess_fixed(1, 1), Catch::Matchers::WithinAbs(0.0, 1e-4));
    REQUIRE_THAT(hess_fixed(0, 1), Catch::Matchers::WithinAbs(0.0, 1e-4));

    // Other elements should match standard
    REQUIRE_THAT(hess_fixed(0, 0),
                 Catch::Matchers::WithinAbs(hess_standard(0, 0), 1e-4));
  }
}

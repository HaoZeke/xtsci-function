// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include "xtsci/func/trial/D2/rosenbrock.hpp"
#include <catch2/catch_all.hpp>

TEST_CASE("ObjectiveFunction Caching Mechanism", "[ObjectiveFunctionCache]") {
  using Scalar = double;
  xt::xarray<Scalar> x = {0.3, 2.0}; // Arbitrary test point

  SECTION("Function Caching") {
    xts::func::trial::D2::Rosenbrock<Scalar> rosenbrock;
    auto initial_evals = rosenbrock.evaluation_counts().function_evals;
    REQUIRE(initial_evals == 0);
    rosenbrock(x);
    REQUIRE(rosenbrock.evaluation_counts().function_evals == 1);
    REQUIRE(rosenbrock.evaluation_counts().unique_func_grad_hess == 1);
    rosenbrock(x);
    REQUIRE(rosenbrock.evaluation_counts().function_evals == 2);
    REQUIRE(rosenbrock.evaluation_counts().unique_func_grad_hess == 1);
  }

  SECTION("Gradient Caching") {
    xts::func::trial::D2::Rosenbrock<Scalar> rosenbrock;
    auto initial_evals = rosenbrock.evaluation_counts().gradient_evals;
    REQUIRE(initial_evals == 0);
    rosenbrock.gradient(x);
    REQUIRE(rosenbrock.evaluation_counts().gradient_evals == 1);
    REQUIRE(rosenbrock.evaluation_counts().unique_func_grad_hess == 1);
    rosenbrock.gradient(x);
    REQUIRE(rosenbrock.evaluation_counts().gradient_evals == 2);
    REQUIRE(rosenbrock.evaluation_counts().unique_func_grad_hess == 1);
  }

  SECTION("Hessian Caching") {
    xts::func::trial::D2::Rosenbrock<Scalar> rosenbrock;
    auto initial_evals = rosenbrock.evaluation_counts().hessian_evals;
    REQUIRE(initial_evals == 0);
    rosenbrock.hessian(x);
    REQUIRE(rosenbrock.evaluation_counts().hessian_evals == 1);
    REQUIRE(rosenbrock.evaluation_counts().unique_func_grad_hess == 1);
    rosenbrock.hessian(x);
    REQUIRE(rosenbrock.evaluation_counts().hessian_evals == 2);
    REQUIRE(rosenbrock.evaluation_counts().unique_func_grad_hess == 1);
  }

  SECTION("Function then Gradient Caching") {
    xts::func::trial::D2::Rosenbrock<Scalar> rosenbrock;
    rosenbrock(x);
    REQUIRE(rosenbrock.evaluation_counts().function_evals == 1);
    REQUIRE(rosenbrock.evaluation_counts().unique_func_grad_hess == 1);

    rosenbrock.gradient(x);
    REQUIRE(rosenbrock.evaluation_counts().gradient_evals == 1);
    REQUIRE(rosenbrock.evaluation_counts().unique_func_grad_hess == 2);
  }

  SECTION("Function then Hessian Caching") {
    xts::func::trial::D2::Rosenbrock<Scalar> rosenbrock;
    rosenbrock(x);
    REQUIRE(rosenbrock.evaluation_counts().function_evals == 1);
    REQUIRE(rosenbrock.evaluation_counts().unique_func_grad_hess == 1);

    rosenbrock.hessian(x);
    REQUIRE(rosenbrock.evaluation_counts().hessian_evals == 1);
    REQUIRE(rosenbrock.evaluation_counts().unique_func_grad_hess == 2);
  }

  SECTION("Gradient then Hessian Caching") {
    xts::func::trial::D2::Rosenbrock<Scalar> rosenbrock;
    rosenbrock.gradient(x);
    REQUIRE(rosenbrock.evaluation_counts().gradient_evals == 1);
    REQUIRE(rosenbrock.evaluation_counts().unique_func_grad_hess == 1);

    rosenbrock.hessian(x);
    REQUIRE(rosenbrock.evaluation_counts().hessian_evals == 1);
    REQUIRE(rosenbrock.evaluation_counts().unique_func_grad_hess == 2);
  }

  SECTION("Function, Gradient, then Hessian Caching") {
    xts::func::trial::D2::Rosenbrock<Scalar> rosenbrock;
    rosenbrock(x);
    REQUIRE(rosenbrock.evaluation_counts().function_evals == 1);
    REQUIRE(rosenbrock.evaluation_counts().unique_func_grad_hess == 1);

    rosenbrock.gradient(x);
    REQUIRE(rosenbrock.evaluation_counts().gradient_evals == 1);
    REQUIRE(rosenbrock.evaluation_counts().unique_func_grad_hess == 2);

    rosenbrock.hessian(x);
    REQUIRE(rosenbrock.evaluation_counts().hessian_evals == 1);
    REQUIRE(rosenbrock.evaluation_counts().unique_func_grad_hess == 3);
  }
}

library(R6)
library(pracma)

Himmelblau <- R6Class("Himmelblau",
  public = list(
    minima = list(c(3, 2),
                  c(-2.805118, 3.131312),
                  c(-3.779310, -3.283186),
                  c(3.584428, -1.848126)),

    initialize = function() {
      # Constructor
      print("Himmelblau function initialized")
    },

    compute = function(x) {
      x_val <- x[1]
      y_val <- x[2]
      return((x_val^2 + y_val - 11)^2 + (x_val + y_val^2 - 7)^2)
    }
  )
)

# Usage:
himmelblau_instance <- Himmelblau$new()
result <- himmelblau_instance$compute(c(3, 2))
print(result)

xx <- c(0.623, 0.028)
res <- numericDeriv(expr = quote(himmelblau_instance$compute(xx)), theta = "xx")
print(res)

result_hessian <- pracma::hessian(himmelblau_instance$compute, xx)
print(result_hessian)

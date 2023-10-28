library(R6)
library(numDeriv)
library(pracma)

Eggholder <- R6Class("Eggholder",
  public = list(
    initialize = function() {
      print("Eggholder function initialized")
    },

    compute = function(x) {
      x_val <- x[1]
      y_val <- x[2]
      return(-(y_val + 47) * sin(sqrt(abs(x_val / 2 + (y_val + 47)))) -
               x_val * sin(sqrt(abs(x_val - (y_val + 47)))))
    }
  )
)

eggholder_instance <- Eggholder$new()

# Test the function
xx <- c(0.623, 0.028)
res <- numericDeriv(expr = quote(eggholder_instance$compute(xx)), theta = "xx")
print(res)

# Compute the Hessian
result_hessian <- pracma::hessian(eggholder_instance$compute, xx)
print(result_hessian)

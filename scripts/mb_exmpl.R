library(pracma)

muller_brown <- function(x) {
  A <- c(-200, -100, -170, 15)
  a <- c(-1, -1, -6.5, 0.7)
  b <- c(0, 0, 11, 0.6)
  c_coeff <- c(-10, -10, -6.5, 0.7)
  x0 <- c(1, 0, -0.5, -1)
  y0 <- c(0, 0.5, 1.5, 1)

  result <- 0
  for (i in 1:4) {
    result <- result + A[i] * exp(a[i] * (x[1] - x0[i])^2 +
                                  b[i] * (x[1] - x0[i]) * (x[2] - y0[i]) +
                                  c_coeff[i] * (x[2] - y0[i])^2)
  }
  return(result)
}

xx <- c(-0.558, 1.442)
res <- numericDeriv(expr = quote(muller_brown(xx)), theta = "xx")
print(attr(res, "gradient"))

xx <- c(0.623, 0.028)
res <- numericDeriv(expr = quote(muller_brown(xx)), theta = "xx")
print(attr(res, "gradient"))

result_hessian <- pracma::hessian(muller_brown, c(-0.558, 1.442))
print(result_hessian)

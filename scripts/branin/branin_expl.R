library(pracma)

branin <- function(x) {
  a <- 1
  b <- 5.1 / (4 * pi^2)
  c <- 5 / pi
  r <- 6
  s <- 10
  t <- 1 / (8 * pi)

  x1 <- x[1]
  x2 <- x[2]

  term1 <- a * (x2 - b * x1^2 + c * x1 - r)^2
  term2 <- s * (1 - t) * cos(x1)

  return(term1 + term2 + s)
}

test_points <- list(c(0, 0), c(1, 1), c(2, 2), c(-1, -1))

for (point in test_points) {
  cat("For point (", point[1], ",", point[2], "):\n")
  cat("Function Value: ", branin(point), "\n")
  cat("Gradient: ", attr(numericDeriv(expr = quote(branin(point)),
                                 theta = "point"), "gradient"), "\n")
  cat("Hessian:\n", pracma::hessian(branin, point), "\n\n")
}

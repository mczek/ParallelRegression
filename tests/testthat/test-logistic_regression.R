test_that("serial implementation", {
  set.seed(1865)
  x <- matrix(rnorm(200), ncol=2)
  beta <- matrix(c(1, 2), ncol=1)
  p <- exp(x %*% beta) / (1 + exp(x %*% beta))
  y <- as.numeric(runif(100) < p)
  
  par_model <- ParallelRegression::ParLR(x, y)
  beta_hat <- par_model$beta
  niter <- par_model$niter
  
  logit <- glm(y ~ x[,1] + x[,2] - 1, family = "binomial")
  best <- as.vector(logit$coefficients)
  
  
  expect_equal(as.vector(beta_hat), best, tolerance=0.05)
  expect_equal(par_model$niter, 6, tolerance=0.05)
})


test_that("parallel with 2 cores", {
  set.seed(1865)
  x <- matrix(rnorm(200), ncol=2)
  beta <- matrix(c(1, 2), ncol=1)
  p <- exp(x %*% beta) / (1 + exp(x %*% beta))
  y <- as.numeric(runif(100) < p)
  
  par_model <- ParallelRegression::ParLR(x, y, 2)
  beta_hat <- par_model$beta
  niter <- par_model$niter
  
  logit <- glm(y ~ x[,1] + x[,2] - 1, family = "binomial")
  best <- as.vector(logit$coefficients)
  
  expect_equal(as.vector(beta_hat), best, tolerance=0.05)
  expect_equal(par_model$niter, c(6, 6), tolerance=1)
})

test_that("parallel with 2 cores, comm = 1", {
  set.seed(1865)
  x <- matrix(rnorm(200), ncol=2)
  beta <- matrix(c(1, 2), ncol=1)
  p <- exp(x %*% beta) / (1 + exp(x %*% beta))
  y <- as.numeric(runif(100) < p)
  
  par_model <- ParallelRegression::ParLR(x, y, 2, 1)
  beta_hat <- par_model$beta
  niter <- par_model$niter
  
  logit <- glm(y ~ x[,1] + x[,2] - 1, family = "binomial")
  best <- as.vector(logit$coefficients)
  
  expect_equal(as.vector(beta_hat), best, tolerance=0.05)
})

test_that("parallel with 2 cores, comm = 2", {
  set.seed(1865)
  x <- matrix(rnorm(200), ncol=2)
  beta <- matrix(c(1, 2), ncol=1)
  p <- exp(x %*% beta) / (1 + exp(x %*% beta))
  y <- as.numeric(runif(100) < p)
  
  par_model <- ParallelRegression::ParLR(x, y, 2, 2)
  beta_hat <- par_model$beta
  niter <- par_model$niter
  
  logit <- glm(y ~ x[,1] + x[,2] - 1, family = "binomial")
  best <- as.vector(logit$coefficients)
  
  expect_equal(as.vector(beta_hat), best, tolerance=0.05)
})


test_that("parallel with 2 cores, comm = 3", {
  set.seed(1865)
  x <- matrix(rnorm(200), ncol=2)
  beta <- matrix(c(1, 2), ncol=1)
  p <- exp(x %*% beta) / (1 + exp(x %*% beta))
  y <- as.numeric(runif(100) < p)
  
  par_model <- ParallelRegression::ParLR(x, y, 2, 3)
  beta_hat <- par_model$beta
  niter <- par_model$niter
  
  logit <- glm(y ~ x[,1] + x[,2] - 1, family = "binomial")
  best <- as.vector(logit$coefficients)
  
  expect_equal(as.vector(beta_hat), best, tolerance=0.05)
})


test_that("timing", {
  set.seed(1865)
  x <- matrix(rnorm(200000), ncol=2)
  beta <- matrix(c(1, 2), ncol=1)
  p <- exp(x %*% beta) / (1 + exp(x %*% beta))
  y <- as.numeric(runif(100000) < p)
  
  new_time2 <- system.time(beta_hat <- ParallelRegression::ParLR(x, y, 2))
  new_time1 <- system.time(beta_hat <- ParallelRegression::ParLR(x, y, 1))
  
  testthat::expect_lt(new_time2[3], new_time1[3])
})

##################################################################
# Bayesian Optimization in 1D based on Expected Improvement
##################################################################

library(DEoptim)
library(mvtnorm)

#Objective function that we wish to maximize
obj <- \(x) (1.4 - 3*x) * sin(18*x)

#True maximum
xTrue <- DEoptim(\(x) -obj(x), 0, 1.2, 
                 control = DEoptim.control(trace = FALSE))$optim$bestmem
yTrue <- obj(xTrue)

covMat32 <- function(s, t, alpha, rho) {
  dist <- sqrt((s-t)^2)
  alpha^2 * (1 + (sqrt(3) / rho) * dist) * exp((-sqrt(3) / rho) * dist);
}

getOptPar <- \(x, y) {
  opt <- DEoptim(\(par) {
    cMat <- outer(x, x, covMat32, par[2], par[3]) + diag(1e-12, length(x))
    -dmvnorm(y, mean = rep(par[1], length(x)), sigma = cMat, log = TRUE)
  }, c(0, 0, 0), c(10, 10, 10), control = DEoptim.control(trace = FALSE))
  parOpt <- opt$optim$bestmem
  
  list(mu = parOpt[1], alpha = parOpt[2], rho = parOpt[3])
}

getPosterior <- \(xEval, x, y, mu, alpha, rho) {
  C1 <- outer(x, x, covMat32, parOpt$alpha, parOpt$rho) + diag(1e-12, length(x))
  C2 <- outer(xEval, x, covMat32, parOpt$alpha, parOpt$rho)
  C3 <- outer(xEval, xEval, covMat32, parOpt$alpha, parOpt$rho)
  
  f_mu <- as.numeric(parOpt$mu + C2 %*% solve(C1) %*% (y - parOpt$mu))
  f_cov <- C3 - C2 %*% solve(C1, t(C2))
  f_sigma <- sqrt(diag(f_cov))

  #Calculate Expected Improvement
  #EI(xStar) = E[max(f(xStar) - max(y), 0)]
  Z <- (f_mu - max(y)) / f_sigma
  EI <- (f_mu - max(y)) * pnorm(Z) + f_sigma * dnorm(Z)
  
  list(f_mu = f_mu,
       f_cov = f_cov,
       EI = EI)
}

################### Fitting starts here ###################
#Start by filling out the search space with a few points
x <- seq(0, 1.2, length.out = 3)
y <- obj(x)

#Only used for plotting
xEval <- seq(0, 1.2, length.out = 100)

# Maximum number of iterations
iterMax <- 40

set.seed(12345)
for(iter in 1:iterMax) {
  parOpt <- getOptPar(x, y)
  posterior <- getPosterior(xEval, x, y, parOpt$mu, parOpt$alpha, parOpt$rho)
  
  #Only used for visualization - can be omitted
  f <- rmvnorm(100, posterior$f_mu, posterior$f_cov)
  
  #Find the next proposal
  xNew <- DEoptim(\(xStar) {
    -getPosterior(xStar, x, y, parOpt$mu, parOpt$alpha, parOpt$rho)$EI
  }, c(0), c(1.2), control = DEoptim.control(trace = FALSE))$optim$bestmem
  yNew <- obj(xNew)
  
  ################### Plotting stuff ###################
  cat("Iter ", iter, ": New = ", 
      xNew, " (", yNew, "), ",
      "Best = ", x[which.max(y)], " (", max(y), ")",
      "\n", sep="")
  
  par(mgp=c(2,1,0), mar=c(3,3,1,0), bty="n")
  matplot(xEval, t(f), type="l", lty=1, col="lightgray", ylim=c(-3, 3),
          xlab="x", ylab="f(x)")
  title(paste0("Iteration ", iter))
  curve(obj(x), 0, 1.2, 200, col=2, lwd=2, ylim=c(-3,3), xlab="", ylab="", add = TRUE)
  points(xTrue, yTrue, col = 2, pch = 19, cex=2)
  points(x, y, pch = 19)
  points(xNew, yNew, pch = 17, col=4, cex=2)
  abline(v = xNew, col = 4)
  abline(v = xTrue, col = 2)
  legend("topright", c("Sampled", "Truth", "Proposal"),
         col = c(1,2,4), pch = c(19, 19, 17), bty="n")
  ######################################################
  
  #Update with new observation
  x <- c(x, xNew)
  y <- c(y, yNew)
}
cat("Truth = ", xTrue, " (", yTrue, ")", "\n", sep="")

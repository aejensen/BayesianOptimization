##################################################################
# Bayesian Optimization in 2D based on Expected Improvement
##################################################################

library(DEoptim)
library(mvtnorm)
library(fields)

#Objective function that we wish to maximize
#McCormick function
obj <- \(x1, x2) -(sin(x1+x2) + (x1-x2)^2 - 1.5*x1 + 2.5*x2 + 1)

plotSurface <- \() {
  xEval1 <- seq(-1.5, 4, length.out = 500)
  xEval2 <- seq(-3, 4, length.out = 500)
  objEval <- outer(xEval1, xEval2, obj)
  par(mgp = c(2,1,0), mar = c(3,3,2,1), bty = "n", cex.lab = 1.3)
  fields::image.plot(xEval1, xEval2, objEval, nlevel = 512,
                     xlab=expression(x[1]), ylab=expression(x[2]))
}

#True maximum
xTrue <- DEoptim(\(x) -obj(x[1], x[2]), c(-1.5, -3), c(4, 4), 
                 control = DEoptim.control(trace = FALSE))$optim$bestmem
yTrue <- obj(xTrue[1], xTrue[2])

# covMat32_2D <- function(s, t, alpha, rho) {
#   dist1 <- sqrt((s[1] - t[1])^2)
#   dist2 <- sqrt((s[2] - t[2])^2)
#   
#   K1 <- (1 + (sqrt(3) / rho[1]) * dist1) * exp((-sqrt(3) / rho[1]) * dist1)
#   K2 <- (1 + (sqrt(3) / rho[2]) * dist2) * exp((-sqrt(3) / rho[2]) * dist2)
#   
#   alpha^2 * K1 * K2
# }
# 
# getCovMat <- function(s, t, alpha, rho) {
#   #Stupid implementation - very slow - fix!
#   cMat <- matrix(NA, nrow(s), nrow(t))
#   for(a in 1:nrow(s)) {
#     for(b in 1:nrow(t)) {
#       cMat[a, b] <- covMat32_2D(s[a,], t[b,], alpha, rho)
#     }
#   }
#   cMat
# }

covMat32_2D <- \(s, t, rho){
  dist <- sqrt((s - t) ^ 2)
  (1 + (sqrt(3) / rho) * dist) * exp((-sqrt(3) / rho) * dist)
}

getCovMat <- \(s, t, alpha, rho){# also holds for higher dimensions
  alpha ^ 2 * do.call("*", lapply(seq_len(ncol(s)), \(x) outer(s[,x], t[,x], covMat32_2D, rho[x])))
}

getOptPar <- \(x, y) {
  opt <- DEoptim(\(par) {
    cMat <- getCovMat(x, x, par[2], par[3:4])
    -dmvnorm(y, mean = rep(par[1], nrow(x)), sigma = cMat, log = TRUE)
  }, rep(0, 4), rep(100, 4), control = DEoptim.control(trace = FALSE))
  parOpt <- opt$optim$bestmem
  
  list(mu = parOpt[1], alpha = parOpt[2], rho = parOpt[3:4])
}

getPosterior <- \(xEval, x, y, mu, alpha, rho) {
  C1 <- getCovMat(x, x, parOpt$alpha, parOpt$rho) + diag(1e-12, nrow(x))
  C2 <- getCovMat(xEval, x, parOpt$alpha, parOpt$rho)
  C3 <- getCovMat(xEval, xEval, parOpt$alpha, parOpt$rho)
  
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


x <- as.matrix(expand.grid(x1 = seq(-1.5, 4, length.out = 3), 
                           x2 = seq(-3, 4, length.out = 3)))
y <- apply(x, 1, \(q) obj(q[1], q[2]))

iterMax <- 50
savePlot <- FALSE

if(savePlot) {
  png(paste0("img-", sprintf("%02d", 0), ".png"), 800, 600)
}
plotSurface()
title(paste0("Iteration ", 0))
points(x[,1], x[,2], pch = 19, cex = 1.2)
points(xTrue[1], xTrue[2], pch = 19, cex = 2, col = 2)
points(x[which.max(y),1], x[which.max(y),2], pch = 3, cex = 2, col = 7, lwd = 2)
if(savePlot) {
  dev.off()
}

for(iter in 1:iterMax) {
  parOpt <- getOptPar(x, y)
  xNew <- DEoptim(\(xStar) {
    -getPosterior(matrix(xStar, 1, 2), x, y, parOpt$mu, parOpt$alpha, parOpt$rho)$EI
  }, c(-1.5, -3), c(4, 4), control = DEoptim.control(trace = FALSE))$optim$bestmem
  yNew <- obj(xNew[1], xNew[2])

  if(savePlot) {
    png(paste0("img-", sprintf("%02d", iter), ".png"), 800, 600)
  }
  plotSurface()
  title(paste0("Iteration ", iter))
  points(x[,1], x[,2], pch=19, cex = 1.2)
  points(xTrue[1], xTrue[2], pch = 19, cex = 2, col = 2)
  points(xNew[1], xNew[2], pch = 17, col = 4, cex = 2)
  points(x[which.max(y),1], x[which.max(y),2], pch = 3, cex = 2, col = 7, lwd = 2)
  if(savePlot) {
    dev.off()
  }
  
  x <- rbind(x, xNew)
  y <- c(y, yNew)
}

#convert -delay 100 -loop 0 *.png -scale 800x600 BayOpt_2D.gif
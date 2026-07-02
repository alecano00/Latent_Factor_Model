###############################################################################
# DGP.R
# Data Generating Process for Autoencoder Simulation (Fama-French 3-Factor)
###############################################################################

# ---- Configuration ----
# Parse arguments to accept a start and end range for M
args <- commandArgs(trailingOnly=TRUE)
if (length(args) == 0) {
  # Default: run simulations 101 through 200 if no arguments are provided
  m_start <- 101
  m_end   <- 200
} else if (length(args) == 1) {
  # If only 1 argument is provided, run just that single simulation
  m_start <- as.integer(args[1])
  m_end   <- as.integer(args[1])
} else {
  # If 2 arguments are provided, use them as the start and end of the range
  m_start <- as.integer(args[1])
  m_end   <- as.integer(args[2])
}

path <- "demo_data/"
dir.create(path, showWarnings = FALSE)

# ---- Model Parameters (These stay constant for all runs) ----
F  <- 3     # Number of latent factors
N  <- 200   # Number of assets (cross-section)
T  <- 180   # Number of time periods
p  <- 50    # Number of firm characteristics
nx <- 50    # Number of macro/predictor variables

lambda   <- 0.3   # Mean of macro variables
stdita   <- 0.1   # Std dev of factor measurement noise (eta)
stde     <- 0.1   # Std dev of idiosyncratic return noise (epsilon)
std_beta <- 0.1   # Scaling factor for beta loadings

# ---- Panel Indices ----
# Create cross-section (per) and time (time) identifiers for the stacked panel
per  <- rep(1:N, T)       
time <- rep(1:T, each = N) 

cat(sprintf("Starting batch simulations from M = %d to %d...\n", m_start, m_end))

# =============================================================================
# ---- MAIN SIMULATION LOOP ----
# =============================================================================
for (M in m_start:m_end) {
  
  # Set seed inside the loop so each M generates the exact same sequence 
  # as if you ran the script manually one-by-one.
  set.seed(M * 123)    
  
  # ---- Generate Firm Characteristics (c) ----
  rho <- runif(p, 0.9, 1)
  c <- matrix(0, N * T, p)
  
  for (i in 1:p) {
    x <- matrix(0, N, T)
    x[, 1] <- rnorm(N)
    for (t in 2:T) {
      x[, t] <- rho[i] * x[, t - 1] + rnorm(N) * sqrt(1 - rho[i]^2)
    }
    # Cross-sectional rank transform to [-1, 1] at each time period
    x_ranked <- apply(x, 2, rank) * 2 / (N + 1) - 1
    c[, i] <- as.vector(x_ranked)
  }
  
  # ---- Generate Macro Variables and Latent Factors ----
  xt <- matrix(rnorm(nx * T, lambda, 1), nx, T)
  
  # Factor loading matrix W
  W <- matrix(0, F, nx)
  W[1, ] <- c(1, rep(0, nx - 1))
  W[2, ] <- c(0, 1, rep(0, nx - 2))
  W[3, ] <- c(0, 0, 1, rep(0, nx - 3))
  
  ft0 <- W %*% xt                              
  ita <- matrix(rnorm(F * T), F, T) * stdita    
  ft  <- ft0 + ita                              
  
  # ---- Generate Returns (Model 1: Linear Betas) ----
  rt  <- matrix(0, N, T)   
  rt0 <- matrix(0, N, T)   
  rtp <- matrix(0, N, T)   
  ft_mean <- apply(ft, 1, mean)
  
  for (t in 1:T) {
    Z <- c[time == t, ]           
    beta <- matrix(0, N, F)
    beta[, 1] <- Z[, 1] * 1.2
    beta[, 2] <- Z[, 2] * 1.0
    beta[, 3] <- Z[, 3] * 0.8
    beta <- beta * std_beta       
    
    rt[, t]  <- beta %*% ft[, t]       
    rt0[, t] <- beta %*% ft0[, t]      
    rtp[, t] <- beta %*% ft_mean       
  }
  
  # Add idiosyncratic noise: t-distributed with df=5, scaled by stde
  ep <- matrix(rt(N * T, 5), N, T) * stde
  rt <- rt + ep
  
  # ---- Diagnostics ----
  vol <- apply(rt, 1, sd) * sqrt(12)
  r2  <- 1 - mean((rt0 - rt)^2) / mean(rt^2)   
  r2p <- 1 - mean((rtp - rt)^2) / mean(rt^2)    
  
  cat(sprintf("Simulation %d: mean vol = %.4f, R2 = %.4f, R2 (predictable) = %.4f\n",
              M, mean(vol), r2, r2p))
  
  # ---- Save Output ----
  write.csv(as.vector(rt), paste0(path, "r1_", M, ".csv"), row.names = FALSE)
  write.csv(xt,            paste0(path, "xt_", M, ".csv"), row.names = FALSE)
  write.csv(c,             paste0(path, "c_",  M, ".csv"), row.names = FALSE)
}

cat("All simulations completed successfully!\n")
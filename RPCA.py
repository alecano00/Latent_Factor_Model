# =============================================================================
# RPCA.py — Regressed Principal Components Analysis
# =============================================================================
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

# ---- Global configuration and data loading (Matching IPCA.py) ----
path = "demo_data"
M, an = 1, 1
# Load the panel of characteristics just like IPCA.py
data = pd.read_csv(path + "/c_%d.csv" % an, delimiter=",").values 

P = 50          # Number of characteristics
K = 5           # Number of latent factors
T = 180         # Number of time periods

def RegressedPCA(X, W, Nt, time, udate):
    """
    Estimate the Regressed-PCA model of Chen, Roussanov, and Wang (2025).
    
    Parameters: Same as IPCA()
    Returns: ret_hat, ret_hatp, Gamma
    """
    T1 = int(T / 3 * 2)   # Number of training + validation periods (first 2/3)

    # ---- Step 1: Cross-Sectional Regressions ----
    # For each time t, estimate the characteristic returns: 
    # lambda_t = (Z_t' Z_t)^{-1} Z_t' r_t = W_t^{-1} X_t
    Lambda_hat = np.zeros((P, T))
    for t in range(T):
        # We use pinv (pseudo-inverse) to ensure stability if W_t is near-singular
        Lambda_hat[:, t] = np.linalg.pinv(W[:, :, t]).dot(X[:, t])

    # ---- Step 2: PCA on Estimated Characteristic Returns ----
    # Extract the top K principal components using the training period data
    Gamma, s, v = svds(Lambda_hat[:, :T1], k=K)

    # svds returns values in ascending order; we must sort them descendingly
    idx = np.argsort(s)[::-1]
    Gamma = Gamma[:, idx]

    # ---- Step 3: Estimate Latent Factors (F) ----
    # Given the estimated Gamma, solve for F across ALL periods
    F = np.zeros((K, T))
    for t in range(T):
        # FOC for F given Gamma: F_t = (Gamma' W_t Gamma)^{-1} Gamma' X_t
        F[:, t] = np.linalg.pinv(Gamma.T.dot(W[:, :, t]).dot(Gamma)).dot(Gamma.T).dot(X[:, t])

    # ---- Step 4: Compute Return Predictions ----
    ret_hat = np.zeros(len(data))      # Contemporaneous factor predictions
    ret_hatp = np.zeros(len(data))     # Predictive (expanding-window mean) predictions
    lam = np.mean(F[:, :T1], 1)        # Time-series mean factor over train+val period

    for t in range(T):
        ind = time == udate[t]
        
        # Contemporaneous: r_hat_{i,t} = z_{i,t}' * Gamma * f_t
        ret_hat[ind] = data[ind].dot(Gamma).dot(F[:, t]) #uses value of f_t

        # Predictive: use expanding-window mean of factors
        if t >= T1:
            lamnew = np.mean(F[:, :t], 1) 
        else:
            lamnew = lam

        # r_hat_pred_{i,t} = z_{i,t}' * Gamma * lamnew
        ret_hatp[ind] = data[ind].dot(Gamma).dot(lamnew) #uses average of f_t up to t-1

    return ret_hat, ret_hatp, Gamma
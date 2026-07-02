# =============================================================================
# IPCA.py — Instrumented Principal Components Analysis (IPCA)
# =============================================================================
# This module implements the IPCA estimator from Kelly, Pruitt, and Su (2019),
# "Characteristics are Covariances: A Unified Model of Risk and Return."
#
# Model
# -----
# IPCA assumes a conditional linear factor model:
#     r_{i,t} = beta(z_{i,t-1})' * f_t + u_{i,t}       (Eq. 1)
# where factor loadings are a LINEAR function of characteristics:
#     beta(z_{i,t-1}) = Gamma' * z_{i,t-1}               (Eq. 2)
# so that:
#     r_{i,t} = z_{i,t-1}' * Gamma * f_t + u_{i,t}
#
# Unlike vanilla PCA (EMPCA.py), IPCA uses asset characteristics z to guide
# the estimation of time-varying factor loadings. Unlike the autoencoders
# (Auto.py, AutoSBN.py), IPCA restricts the mapping from characteristics to
# betas to be linear.
#
# Estimation: Alternating Least Squares (ALS)
# --------------------------------------------
# IPCA solves (Eq. 17 in the paper):
#     min_{Gamma, F} sum_t || r_t - Z_t * Gamma * f_t ||^2
#
# This is a bilinear optimization, solved by alternating between:
#   1. Given Gamma, solve for F (factor estimation step)
#   2. Given F, solve for Gamma (loading-map estimation step)
# with rotation normalization applied after each iteration.
#
# The ALS iterates until convergence (max element-wise change in Gamma < 1e-6)
# or until MaxIterations is reached.
#
# Key inputs (precomputed in Simulation.ipynb):
#   X[:, t]     = (1/N_t) * Z_t' * r_t    — managed portfolio (sufficient statistic)
#   W[:, :, t]  = (1/N_t) * Z_t' * Z_t    — characteristic second moment
#   Nt[t]       = number of assets at time t
#
# Two return predictions are produced:
#   1. ret_hat:  Uses estimated factors at each t: r_hat = Z * Gamma * f_t
#   2. ret_hatp: Uses expanding-window mean factors (predictive strategy)
#
# Authors: Shihao Gu, Bryan Kelly, Dacheng Xiu
# Date:    February 2019
# =============================================================================

import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from sklearn import linear_model
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

# ---- Global configuration and data loading ----------------------------------
path = "demo_data"
M, an = 1, 1    # M: DGP variant index; an: simulation replicate index
data = pd.read_csv(path + "/c_%d.csv" % an, delimiter=",").values  # (N*T, P) characteristics panel

# ---- Simulation parameters --------------------------------------------------
P = 50          # Number of characteristics
N = 200         # Number of cross-sectional assets
m = P           # Alias for number of characteristics
T = 180         # Number of time periods
K = 5           # Number of latent factors
F = np.zeros((K, T))   # Placeholder for estimated factors


# =============================================================================
# Helper functions for Variable Importance in Projection (VIP)
# =============================================================================

def vip(Z, Gamma, X, y, r2, date):
    """
    Compute VIP scores by zeroing out one characteristic at a time in Z
    (the loading side) while holding factors F fixed. Measures each
    characteristic's marginal contribution to the total R-squared.

    Parameters
    ----------
    Z : ndarray of shape (n_total, P)
        Panel of asset characteristics (stacked).
    Gamma : ndarray of shape (P, K)
        Mapping matrix from characteristics to factor loadings.
    X : ndarray of shape (P, T)
        Managed portfolio moments (used for time dimension only).
    y : ndarray of shape (n_total,)
        Actual returns.
    r2 : float
        Baseline total R-squared with all characteristics.
    date : ndarray of shape (n_total,)
        Time period identifier.

    Returns
    -------
    v : ndarray of shape (P,)
        Normalized VIP scores (sum of absolute values = 1).
    """
    N = Z.shape[1]
    v = np.zeros(N)
    udate = np.unique(date)

    for i in range(N):
        Z1 = np.copy(Z)
        Z1[:, i] = 0          # Zero out the i-th characteristic
        ret_hat = np.zeros(len(date))

        # Recompute fitted returns: r_hat = Z1 * Gamma * F (with fixed F)
        for t in range(X.shape[1]):
            ind = date == udate[t]
            xt1 = X[:, t]
            ret_hat[ind] = Z1[ind, :].dot(Gamma).dot(F[:, t])

        # Measure R^2 drop
        r2new = 1 - np.sum(np.power(ret_hat - y, 2)) / np.sum(np.power(y, 2))
        v[i] = r2 - r2new

    # v=v[:-1]
    v = v / np.sum(np.abs(v))      # Normalize

    return v


def vip2(Z, Gamma, X, y, r2, date):
    """
    Compute VIP scores by zeroing out a characteristic in BOTH Z (loadings)
    and X (managed portfolios), then RE-ESTIMATING factors. This captures
    the total effect of removing a characteristic from the model.

    Parameters
    ----------
    Z : ndarray of shape (n_total, P)
        Panel of asset characteristics.
    Gamma : ndarray of shape (P, K)
        Mapping matrix from characteristics to loadings.
    X : ndarray of shape (P, T)
        Managed portfolio moments.
    y : ndarray of shape (n_total,)
        Actual returns.
    r2 : float
        Baseline total R-squared.
    date : ndarray of shape (n_total,)
        Time period identifier.

    Returns
    -------
    v : ndarray of shape (P,)
        Normalized VIP scores.
    """
    N = Z.shape[1]
    v = np.zeros(N)
    udate = np.unique(date)

    for i in range(N):
        Z1 = np.copy(Z)
        Z1[:, i] = 0          # Zero out characteristic i in loadings
        X1 = np.copy(X)
        X1[i, :] = 0          # Zero out characteristic i in managed portfolios

        # Re-estimate factors with the modified data
        Fnew = np.zeros((K, len(udate)))
        for t in range(len(udate)):
            ind = date == udate[t]
            xt1 = X1[:, t]
            # Recompute W_t = (1/N_t) * Z1' * Z1
            W = Z1[ind, :].T.dot(Z1[ind, :]) / 1.0 / np.sum(ind)
            # Solve for f_t: f_t = (Gamma' * W_t * Gamma)^{+} * Gamma' * x1_t
            Fnew[:, t] = np.linalg.pinv(Gamma.T.dot(W).dot(Gamma), 1e-4).dot(Gamma.T).dot(xt1)

        # Recompute fitted returns with modified Z and re-estimated factors
        ret_hat = np.zeros(len(date))
        for t in range(X1.shape[1]):
            ind = date == udate[t]
            xt1 = X1[:, t]
            ret_hat[ind] = Z1[ind, :].dot(Gamma).dot(Fnew[:, t])

        r2new = 1 - np.sum(np.power(ret_hat - y, 2)) / np.sum(np.power(y, 2))
        v[i] = r2 - r2new

    # v=v[:-1]
    v = v / np.sum(np.abs(v))

    return v


def xtr2(ret_hat, date, udate, L, T, Nt, t0, t1, t2, X):
    """
    Compute R-squared for managed-portfolio cross-sectional moments across
    time sub-periods (train, validation, test, train+val).

    Parameters
    ----------
    ret_hat : ndarray of shape (n_total,)
        Predicted returns (stacked).
    date : ndarray of shape (n_total,)
        Time period identifier.
    udate : ndarray of shape (T,)
        Unique time labels.
    L : int
        Number of characteristics.
    T : int
        Number of time periods.
    Nt : ndarray of shape (T,)
        Number of assets per period.
    t0, t1, t2 : int
        Time boundaries for train/val/test splits.
    X : ndarray of shape (L, T)
        True managed portfolio moments.

    Returns
    -------
    r2 : ndarray of shape (4,)
        R-squared for [train, validation, test, train+val].
    """
    X_hat = np.zeros((L, T))
    for t in range(T):
        ind = date == udate[t]
        xt = ret_hat[ind].reshape(1, Nt[t]).dot(data[ind].values)
        X_hat[:, t] = xt[0, :] / 1.0 / Nt[t]
    r2 = np.zeros(4)
    ind1 = (udate >= t0) * (udate <= t1)   # Training period
    ind2 = (udate >= t1) * (udate <= t2)   # Validation period
    ind3 = (udate >= t2)                    # Test (OOS) period
    ind4 = (udate <= t2)                    # Train + validation
    r2[0] = 1 - np.sum(np.power(X_hat[:, ind1] - X[:, ind1], 2)) / np.sum(np.power(X[:, ind1], 2))
    r2[1] = 1 - np.sum(np.power(X_hat[:, ind2] - X[:, ind2], 2)) / np.sum(np.power(X[:, ind2], 2))
    r2[2] = 1 - np.sum(np.power(X_hat[:, ind3] - X[:, ind3], 2)) / np.sum(np.power(X[:, ind3], 2))
    r2[3] = 1 - np.sum(np.power(X_hat[:, ind4] - X[:, ind4], 2)) / np.sum(np.power(X[:, ind4], 2))

    return r2


# =============================================================================
# Core IPCA estimation: one ALS iteration
# =============================================================================

def num_IPCA_estimate_ALS(Gamma_Old, W, X, Nt):
    """
    Perform one complete Alternating Least Squares (ALS) iteration for IPCA.

    The IPCA objective (Eq. 17) is bilinear in (Gamma, F). Each ALS iteration:
      Step A: Given Gamma_Old, estimate F_New (factor estimation)
      Step B: Given F_New, estimate Gamma_New (loading-map estimation)
      Step C: Normalize Gamma_New and F_New for unique identification

    Parameters
    ----------
    Gamma_Old : ndarray of shape (L, K)
        Current estimate of the mapping matrix from characteristics to loadings.
        L = number of characteristics, K = number of factors.
    W : ndarray of shape (L, L, T)
        Cross-sectional second moments of characteristics:
        W[:, :, t] = (1/N_t) * Z_t' * Z_t.
    X : ndarray of shape (L, T)
        Managed portfolio cross-sectional moments:
        X[:, t] = (1/N_t) * Z_t' * r_t.
    Nt : ndarray of shape (T,)
        Number of assets per period.

    Returns
    -------
    Gamma_New : ndarray of shape (L, K)
        Updated and rotation-normalized Gamma estimate.
    F_New : ndarray of shape (K, T)
        Updated and rotation-normalized factor estimates.
    """
    T = len(Nt)
    L, K = Gamma_Old.shape
    Ktilde = K

    # ---- Step A: Factor estimation (given Gamma_Old) ----
    # For each period t, solve for f_t from the first-order condition:
    #   (Gamma' * W_t * Gamma)' * (Gamma' * W_t * Gamma) * f_t = (Gamma' * W_t * Gamma)' * (Gamma' * x_t)
    # This is a least-squares solve of: XX * f_t = Y, where
    #   XX = Gamma' * W_t * Gamma   (K x K)
    #   Y  = Gamma' * x_t           (K x 1)
    F_New = np.zeros((K, T))
    for t in range(T):
        XX = Gamma_Old.T.dot(W[:, :, t]).dot(Gamma_Old)    # (K x K)
        Y = Gamma_Old.T.dot(X[:, t])                        # (K,)
        # Solve via normal equations: f_t = (XX'*XX)^{-1} * XX' * Y
        F_New[:, t] = np.linalg.inv(XX.T.dot(XX)).dot(XX.T).dot(Y)

    # ---- Step B: Gamma estimation (given F_New) ----
    # The vectorized first-order condition for Gamma is:
    #   Denom * vec(Gamma') = Numer
    # where vec(Gamma') is the (L*K x 1) vectorization of Gamma'.
    # Numer and Denom are accumulated across time periods using Kronecker products:
    #   Numer += N_t * kron(x_t, f_t)
    #   Denom += N_t * kron(W_t, f_t * f_t')
    Numer = np.zeros((L * Ktilde, 1))
    Denom = np.zeros((L * Ktilde, L * Ktilde))
    for t in range(T):
        Numer = Numer + np.kron(X[:, t].reshape(len(X), 1), F_New[:, t].reshape(len(F_New), 1)) * Nt[t]
        Denom = Denom + np.kron(W[:, :, t],
                                F_New[:, t].reshape(len(F_New), 1).dot(F_New[:, t].reshape(1, len(F_New)))) * Nt[t]

    # Solve the linear system: Denom * vec(Gamma') = Numer
    lm = linear_model.LinearRegression(fit_intercept=False, copy_X=True)
    lm.fit(Denom, Numer[:, 0])
    Gamma_New_trans_vec = lm.coef_
    # Reshape from vectorized form back to (K x L) matrix, then transpose to (L x K)
    Gamma_New_trans = Gamma_New_trans_vec.reshape(Ktilde, L, order='F')
    Gamma_New = Gamma_New_trans.T

    # ---- Step C: Rotation normalization for identification ----
    # Apply a rotation so that Gamma' * Gamma = I and factors are ordered
    # by decreasing variance. This ensures a unique solution up to sign.
    #
    # 1. Cholesky decomposition: R1 such that Gamma' * Gamma = R1' * R1
    R1 = np.linalg.cholesky(Gamma_New[:, :K].T.dot(Gamma_New[:, :K]))
    R1 = R1.T   # Upper triangular Cholesky factor

    # 2. SVD of R1 * F * F' * R1' to find the rotation that diagonalizes
    #    the factor covariance in the R1-transformed space
    R2, aa, bb = np.linalg.svd(R1.dot(F_New).dot(F_New.T).dot(R1.T))

    # 3. Apply the rotation: Gamma_new = Gamma * R1^{-1} * R2
    Gamma_New[:, :K] = (Gamma_New[:, :K].dot(np.linalg.inv(R1))).dot(R2)
    # 4. Rotate factors accordingly: F_new = R2' * R1 * F_old
    lm = linear_model.LinearRegression(fit_intercept=False, copy_X=True)
    lm.fit(R2, R1.dot(F_New))
    F_New = lm.coef_.T

    # 5. Sign normalization: ensure each factor has a positive time-series mean
    sg = np.sign(np.mean(F_New, 1))
    sg[sg == 0] = 1                        # Avoid multiplying by zero
    Gamma_New[:, :K] = Gamma_New[:, :K] * sg   # Flip loadings accordingly
    F_New = (F_New.T * sg).T                    # Flip factors accordingly

    return (Gamma_New, F_New)


# =============================================================================
# Main function: IPCA (Instrumented PCA)
# =============================================================================

def IPCA(X, W, Nt, time, udate):
    """
    Estimate the IPCA model via Alternating Least Squares (ALS).

    The procedure:
    1. Initialize Gamma via SVD of the managed portfolio matrix X (training period).
    2. Iterate ALS on the training+validation data (first 2/3 of T) until
       convergence or MaxIterations.
    3. Given the converged Gamma, estimate factors for ALL periods (including OOS).
    4. Compute two types of return predictions.

    Parameters
    ----------
    X : ndarray of shape (P, T)
        Managed portfolio moments: X[:, t] = (1/N_t) * Z_t' * r_t.
    W : ndarray of shape (P, P, T)
        Characteristic second moments: W[:, :, t] = (1/N_t) * Z_t' * Z_t.
    Nt : ndarray of shape (T,)
        Number of assets per time period.
    time : ndarray of shape (N*T,)
        Time period identifier for each stacked observation.
    udate : ndarray of shape (T,)
        Unique sorted time labels.

    Returns
    -------
    ret_hat : ndarray of shape (N*T,)
        Fitted returns using contemporaneous IPCA factors:
        r_hat_{i,t} = z_{i,t}' * Gamma * f_t.
    ret_hatp : ndarray of shape (N*T,)
        Predicted returns using expanding-window mean factors:
        r_hat_{i,t} = z_{i,t}' * Gamma * mean(f_{1:t}).
    Gamma : ndarray of shape (P, K)
        Estimated mapping matrix from characteristics to factor loadings.
    """
    # ---- Initialization ----
    MaxIterations = 10000
    Tolerance = 1e-6
    T1 = int(T / 3 * 2)   # Number of training+validation periods (first 2/3)

    # Initialize Gamma and F via truncated SVD of the managed portfolio matrix X
    # X ≈ Gamma_init * diag(s) * v  (rank-K approximation)
    Gamma_Old, s, v = svds(X[:, :(T1)], k=K)
    Factor_Old = np.diag(s).dot(v)

    # ---- ALS iteration ----
    # Alternate between updating (Gamma, F) until Gamma converges
    tol = 1
    iter = 0
    while (iter <= MaxIterations) and (tol > Tolerance):
        [Gamma_New, Factor_New] = num_IPCA_estimate_ALS(Gamma_Old, W[:, :, :T1], X[:, :T1], Nt[:T1])
        # Convergence criterion: max element-wise change in Gamma
        tol = np.max(np.abs(Gamma_New[:] - Gamma_Old[:]))
        Factor_Old = Factor_New
        Gamma_Old = Gamma_New
        iter = iter + 1

    Gamma = Gamma_New
    Factor = Factor_New
    lam = np.mean(Factor, 1)   # Time-series mean factor over train+val period

    # ---- Estimate factors for ALL T periods using converged Gamma ----
    # For each period t, solve the IPCA first-order condition:
    #   f_t = (Gamma' * W_t * Gamma)^{-1} * Gamma' * x_t
    F = np.zeros((K, T))
    for t in range(T):
        ind = time == udate[t]
        xt1 = X[:, t]
        F[:, t] = np.linalg.inv(Gamma.T.dot(W[:, :, t]).dot(Gamma)).dot(Gamma.T).dot(xt1)

    # ---- Compute return predictions ----
    ret_hat = np.zeros(len(data))      # Contemporaneous factor predictions
    ret_hatp = np.zeros(len(data))     # Predictive (expanding-window mean) predictions
    for t in range(T):
        ind = time == udate[t]
        xt1 = X[:, t]

        # Contemporaneous: r_hat_{i,t} = z_{i,t}' * Gamma * f_t
        ret_hat[ind] = data[ind].dot(Gamma).dot(F[:, t])

        # Predictive: use expanding-window mean of factors
        if t >= (T * 2 / 3):
            lamnew = np.mean(F[:, :t], 1)   # Expanding mean up to (but not including) t
        else:
            lamnew = lam                     # Fixed train+val mean for in-sample periods
        # r_hat_pred_{i,t} = z_{i,t}' * Gamma * lam
        ret_hatp[ind] = data[ind].dot(Gamma).dot(lamnew)

    return ret_hat, ret_hatp, Gamma

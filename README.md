# RPCA vs. IPCA

A comparison of **Regressed Principal Component Analysis (RPCA)** and **Instrumented Principal Component Analysis (IPCA)** using simulated asset pricing data.

This repository contains implementations of both methods, a simulation framework, and a Jupyter notebook comparing their estimation and prediction performance.

## Repository Structure

```
.
├── README.md
├── RPCAvsIPCA.ipynb          # Main notebook
├── RPCAvsIPCA.pdf            # PDF version of the notebook
├── IPCA.py                   # Instrumented Principal Component Analysis implementation
├── RPCA.py                   # Regressed Principal Component Analysis implementation
├── newDGP.r                  # Data-generating process for simulated data
```

## Overview

This project compares two characteristic-based latent factor models commonly used in empirical asset pricing.

### Instrumented Principal Component Analysis (IPCA)

- Estimates latent factors jointly with characteristic-based factor loadings.
- Uses an alternating least squares (ALS) algorithm.
- Produces in-sample fits and out-of-sample return predictions.

### Regressed Principal Component Analysis (RPCA)

- Estimates characteristic returns through cross-sectional regressions.
- Applies PCA to the estimated characteristic returns.
- Produces latent factors and predicted returns.

The accompanying notebook demonstrates the complete workflow, from simulated data generation to model estimation and performance comparison.

## Requirements

- Python 3.10+
- numpy
- pandas
- scipy
- scikit-learn
- Jupyter Notebook

Install dependencies with

```bash
pip install numpy pandas scipy scikit-learn jupyter
```

## Running the Project

1. Generate the simulated data using `newDGP.r`.
2. Place the generated data in the expected directory.
3. Open and run

```bash
jupyter notebook RPCAvsIPCA.ipynb
```

## References

- Kelly, B., Pruitt, S., & Su, Y. (2019). *Characteristics Are Covariances: A Unified Model of Risk and Return.*
- Chen, Z., Roussanov, N., & Wang, L. (2025). *Regressed Principal Component Analysis.*

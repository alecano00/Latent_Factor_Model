# Instrumented and Regressed Principal Component Analysis (IPCA & RPCA)

This repository contains the implementation and simulation materials for **Instrumented Principal Component Analysis (IPCA)** and **Regressed Principal Component Analysis (RPCA)**, together with a Jupyter notebook demonstrating their use on simulated financial data.

## Repository Structure

```
.
├── exam_Dacheng.ipynb    # Main notebook with simulations and examples
├── exam_Dacheng.pdf      # PDF version of the notebook
├── IPCA.py               # Instrumented Principal Component Analysis implementation
├── RPCA.py               # Regressed Principal Component Analysis implementation
├── newDGP.r              # R script for generating simulated return data
└── Readme.txt            # Original project notes
```

## Overview

This project compares two characteristic-based factor models used in empirical asset pricing:

- **IPCA (Instrumented Principal Component Analysis)**
  - Based on Kelly, Pruitt, and Su (2019).
  - Estimates latent factors whose loadings are linear functions of observable asset characteristics.
  - Uses an alternating least squares (ALS) algorithm to jointly estimate factor loadings and latent factors.

- **RPCA (Regressed Principal Component Analysis)**
  - Based on Chen, Roussanov, and Wang (2025).
  - First estimates characteristic returns through cross-sectional regressions.
  - Then applies PCA to the estimated characteristic returns to recover latent factors.

Both methods produce:

- In-sample fitted returns
- Out-of-sample predictive returns using expanding-window factor estimates

---

## Files

### `exam_Dacheng.ipynb`

The main notebook that:

- demonstrates the complete workflow,
- loads the simulated data,
- estimates both IPCA and RPCA models,
- compares their performance, and
- illustrates the simulation results.

### `IPCA.py`

Implementation of Instrumented Principal Component Analysis including:

- ALS estimation procedure
- factor estimation
- loading estimation
- return prediction
- Variable Importance in Projection (VIP) utilities

### `RPCA.py`

Implementation of Regressed Principal Component Analysis including:

- cross-sectional characteristic regressions
- PCA on characteristic returns
- latent factor estimation
- return prediction

### `newDGP.r`

R script used to generate the simulated return data and characteristics used throughout the notebook.

### `exam_Dacheng.pdf`

PDF export of the notebook for easier reading and presentation.

---

## Requirements

Python packages:

- numpy
- pandas
- scipy
- scikit-learn

Install them with:

```bash
pip install numpy pandas scipy scikit-learn
```

---

## Running the Project

1. Install the required Python packages.
2. Generate (or provide) the simulated data using `newDGP.r`.
3. Ensure the generated data is available in the expected `demo_data/` directory.
4. Open and run:

```bash
jupyter notebook exam_Dacheng.ipynb
```

or

```bash
jupyter lab
```

---

## References

**Kelly, B., Pruitt, S., & Su, Y. (2019).**

*Characteristics Are Covariances: A Unified Model of Risk and Return.*

Journal of Financial Economics.

**Chen, Z., Roussanov, N., & Wang, L. (2025).**

*Regressed Principal Component Analysis.*

---

## Notes

The implementations are intended for educational and research purposes and accompany the simulation notebook included in this repository.

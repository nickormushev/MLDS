# Machine Learning for Data Science – Assignments and Experiments

This repository contains my implementations and experiments for the **Machine Learning for Data Science** course. The focus was on understanding core machine learning algorithms, probabilistic inference methods, dimensionality reduction techniques, and evaluation strategies through practical assignments and statistical analysis.

## Implemented Algorithms

- **Decision Trees**  
  A recursive partitioning-based classifier with support for information gain and Gini impurity splitting criteria.

- **Logistic Regression**  
  A linear classifier trained using gradient descent and evaluated under different regularisation schemes.

- **Neural Networks**  
  A feedforward neural network implemented from scratch, with backpropagation and support for multiple hidden layers.

- **Support Vector Machines (SVMs)**  
  Implemented using both primal and dual formulations, with experiments on margin size and kernel functions.

## Bayesian Methods

- **Markov Chain Monte Carlo (MCMC)**  
  Used for posterior sampling in probabilistic models.

- **Laplace Approximation**  
  Applied to approximate Bayesian inference.

## Dimensionality Reduction

- **Principal Component Analysis (PCA)**  
  Used for visualisation and feature decorrelation in high-dimensional data.

- **t-SNE (t-distributed Stochastic Neighbour Embedding)**  
  Explored for non-linear dimensionality reduction and cluster visualisation.

## Evaluation Techniques

- **Holdout Estimation vs Cross-Validation**  
  A statistical analysis comparing how test and validation set sizes affect model performance and variance.  
  Key insights include:
  - Trade-offs between bias and variance with different splits
  - Variability in performance estimation across different dataset sizes
  - Stability and robustness of cross-validation over single holdout sets

## Requirements

- Python 3.11
- NumPy
- SciPy
- scikit-learn
- Matplotlib / Seaborn (for visualisation)


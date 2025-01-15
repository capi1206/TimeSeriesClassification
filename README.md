# Binary Classification of Multivariate Time Series Using Transformers

## Project Description

This repository contains the code and resources used to evaluate the performance of a Transformer-based model for binary classification of multivariate time series, applied to cryptocurrency time series data. The motivation for this experiment was to explore the effectiveness of attention mechanisms, widely used in natural language processing (NLP), for analyzing sequential financial data.


## Repository Structure

**train/** : Contains the model training scripts.

**models/** : Includes implemented models, auxiliary functions, and utilities.

**notebooks/** : Contains pca and mutual information for dimensionality reduction


## Model Description and Methodology

The model is based on attention mechanisms using a Transformer architecture adapted for binary time series classification.

There is also a model using boosting RNN with a lstm as the base model.

## Input Data

Each time series point was represented by a 145-dimensional vector, including:

Cryptocurrency price relative to the dollar.

Logarithmic returns over scales from 1 to 72 hours.

MACD histograms (72 values).

## Training Strategies

Separate training for each cryptocurrency using the latest 3000 observations.

Model evaluation using a common validation set including all selected cryptocurrencies.

L1 regularization to select relevant subseries.

### Overfitting Prevention

L1 regularization on the first linear layer.

Multi-head attention mechanisms.

Dropout usage.

### Handling Class Imbalance

PyTorch's BCEWithLogitsLoss() function with positive class weight adjustments to mitigate bias.

### Hyperparameter Optimization

Bayesian optimization was applied to fine-tune key hyperparameters such as batch size and dropout rate.

## Results

The optimized model achieved a validation accuracy of 62% on the selected set of cryptocurrencies. Significant performance variations were observed across different currencies. Additionally:

Variability in model input effectiveness (inclusion/exclusion of MACD histograms).

The need for more robust approaches to generalize across multiple time series.

## Conclusions

Attention mechanisms based on Transformers show potential for binary classification of multivariate time series. However, the lack of generalization across different cryptocurrencies highlights the need for continued hyperparameter tuning and exploration of additional training, regularization, and class balancing techniques.

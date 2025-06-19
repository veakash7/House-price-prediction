# House-price-prediction


This repository contains a Jupyter notebook that implements a house price prediction model using the California Housing dataset and PyTorch. The model is a simple neural network trained to predict house prices based on various features like median income, house age, and more.

## Project Overview

- **Dataset**: California Housing dataset from `sklearn.datasets`.
- **Model**: A PyTorch neural network with two linear layers and ReLU activation.
- **Objective**: Predict house prices (regression task) using normalized features.
- **Evaluation**: Mean Squared Error (MSE) is used to evaluate model performance.

## Requirements

To run the notebook, install the following dependencies:

```bash
pip install torch scikit-learn numpy
```
# Notebook Structure
## Data Preprocessing:
Loads the California Housing dataset.
Normalizes features using StandardScaler.
Splits data into training (80%) and test (20%) sets.
Converts data to PyTorch tensors and creates a DataLoader.
## Model Definition:
Defines a RegressionModel class with a neural network (input → 64 → 1).
## Training:
Trains the model for 100 epochs using Adam optimizer and MSE loss.
Prints loss every 10 epochs.
## Evaluation:
Evaluates the model on the test set and reports the test MSE.
#Results
Final test MSE: ~0.3062 (may vary slightly due to random initialization).
The model achieves reasonable performance but could be improved with hyperparameter tuning or a deeper architecture.
Future Improvements
Experiment with different network architectures (e.g., more layers, dropout).
Perform hyperparameter tuning (e.g., learning rate, batch size).
Add cross-validation for more robust evaluation.
Visualize predictions vs. actual values.
# Contributing
Feel free to open issues or submit pull requests for improvements or bug fixes.

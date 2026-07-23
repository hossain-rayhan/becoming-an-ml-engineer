# Project 1A: Optimization from Scratch

This project implements fundamental optimization algorithms without relying on ML frameworks' automatic differentiation. The goal is to deeply understand gradient descent mechanics.

## Project Structure

```
project-1a-optimization/
├── 01_linear_regression.py      # Linear regression with gradient descent
├── 02_logistic_regression.py    # Logistic regression with gradient descent
├── 03_autograd.py               # Tiny autograd engine (computational graph)
├── 04_visualizations.py         # Learning rate sensitivity & loss curves
├── utils.py                     # Shared utilities
└── README.md                    # This file
```

## Learning Objectives

By completing this project, you will understand:
- How gradient descent updates parameters iteratively
- Why gradients can vanish or explode
- Why softmax + cross-entropy pair well together
- Why optimization is sensitive to scale and initialization

## How to Run

```bash
# Activate your virtual environment first
source .venv/bin/activate

# Run each file individually
python project-1a-optimization/01_linear_regression.py
python project-1a-optimization/02_logistic_regression.py
python project-1a-optimization/03_autograd.py
python project-1a-optimization/04_visualizations.py
```

## Core Concepts

### Gradient Descent Update Rule
$$w_{t+1} = w_t - \eta \cdot \nabla_w L$$

Where:
- $w_t$ = current weights
- $\eta$ = learning rate
- $\nabla_w L$ = gradient of loss with respect to weights

### Why This Matters
Understanding these fundamentals allows you to debug training issues, recognize when optimization is failing, and make informed decisions about hyperparameters.

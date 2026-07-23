"""
Linear Regression with Gradient Descent from Scratch

This implementation shows the core mechanics of gradient descent:
1. Forward pass: compute predictions
2. Loss computation: measure error
3. Backward pass: compute gradients manually
4. Update: adjust parameters

No automatic differentiation - we derive and code gradients by hand.
"""

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# PART 1: Generate Synthetic Data
# =============================================================================

def generate_linear_data(n_samples=100, n_features=1, noise=0.1, seed=42):
    """
    Generate y = Xw + b + noise
    
    Args:
        n_samples: Number of data points
        n_features: Number of input features
        noise: Standard deviation of Gaussian noise
        seed: Random seed for reproducibility
    
    Returns:
        X: Input features (n_samples, n_features)
        y: Target values (n_samples,)
        true_w: True weights used to generate data
        true_b: True bias used to generate data
    """
    np.random.seed(seed)
    
    # True parameters (what we want to learn)
    true_w = np.random.randn(n_features) * 2  # Scale for variety
    true_b = np.random.randn() * 0.5
    
    # Generate X from uniform distribution
    X = np.random.randn(n_samples, n_features)
    
    # Generate y = Xw + b + noise
    y = X @ true_w + true_b + np.random.randn(n_samples) * noise
    
    return X, y, true_w, true_b


# =============================================================================
# PART 2: Linear Regression Class (From Scratch)
# =============================================================================

class LinearRegressionGD:
    """
    Linear Regression using Gradient Descent.
    
    Model: y_pred = X @ w + b
    Loss: MSE = (1/n) * sum((y_pred - y)^2)
    
    Gradient derivation:
    -----------------------
    L = (1/n) * sum((Xw + b - y)^2)
    
    Let e = Xw + b - y (error term)
    
    dL/dw = (2/n) * X^T @ e
    dL/db = (2/n) * sum(e)
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, verbose=True):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.verbose = verbose
        self.w = None
        self.b = None
        self.loss_history = []
        
    def _init_weights(self, n_features):
        """Initialize weights and bias to small random values."""
        # Why small random values?
        # - Zero init: all neurons learn the same thing (symmetry problem)
        # - Large init: can cause exploding gradients
        # - Small random: breaks symmetry, keeps gradients reasonable
        self.w = np.random.randn(n_features) * 0.01
        self.b = 0.0
        
    def _forward(self, X):
        """Forward pass: compute predictions."""
        return X @ self.w + self.b
    
    def _compute_loss(self, y_pred, y_true):
        """Compute Mean Squared Error loss."""
        n = len(y_true)
        mse = (1/n) * np.sum((y_pred - y_true) ** 2)
        return mse
    
    def _compute_gradients(self, X, y_pred, y_true):
        """
        Compute gradients of MSE loss with respect to w and b.
        
        This is the KEY part - doing calculus by hand:
        
        L = (1/n) * Σ(y_pred - y)²
        
        dL/dw_j = (1/n) * Σ 2*(y_pred - y) * d(y_pred)/dw_j
                = (2/n) * Σ (y_pred - y) * x_j
                = (2/n) * X^T @ (y_pred - y)
        
        dL/db = (2/n) * Σ (y_pred - y)
        """
        n = len(y_true)
        error = y_pred - y_true  # (n_samples,)
        
        # Gradient with respect to weights
        dw = (2/n) * (X.T @ error)  # (n_features,)
        
        # Gradient with respect to bias
        db = (2/n) * np.sum(error)  # scalar
        
        return dw, db
    
    def fit(self, X, y):
        """
        Train the model using gradient descent.
        
        The core loop:
        1. Forward pass: get predictions
        2. Compute loss: how wrong are we?
        3. Backward pass: compute gradients
        4. Update: take a step in the direction that reduces loss
        """
        n_samples, n_features = X.shape
        self._init_weights(n_features)
        
        for i in range(self.n_iterations):
            # Forward pass
            y_pred = self._forward(X)
            
            # Compute loss
            loss = self._compute_loss(y_pred, y)
            self.loss_history.append(loss)
            
            # Backward pass (compute gradients)
            dw, db = self._compute_gradients(X, y_pred, y)
            
            # Update weights
            # This is the gradient descent update rule:
            # w = w - learning_rate * gradient
            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db
            
            # Print progress
            if self.verbose and (i % 100 == 0 or i == self.n_iterations - 1):
                print(f"Iteration {i:4d} | Loss: {loss:.6f}")
        
        return self
    
    def predict(self, X):
        """Make predictions on new data."""
        return self._forward(X)
    
    def score(self, X, y):
        """Compute R² score (coefficient of determination)."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)  # Residual sum of squares
        ss_tot = np.sum((y - np.mean(y)) ** 2)  # Total sum of squares
        return 1 - (ss_res / ss_tot)


# =============================================================================
# PART 3: Visualization Functions
# =============================================================================

def plot_loss_curve(model, title="Training Loss Curve"):
    """Plot the loss over training iterations."""
    plt.figure(figsize=(10, 4))
    plt.plot(model.loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("MSE Loss")
    plt.title(title)
    plt.yscale('log')  # Log scale shows convergence behavior better
    plt.grid(True, alpha=0.3)
    plt.savefig("project-1a-optimization/outputs/linear_regression_loss.png", dpi=150, bbox_inches='tight')
    plt.show()
    

def plot_fit_1d(X, y, model, true_w=None, true_b=None):
    """Plot the data and fitted line (for 1D input only)."""
    if X.shape[1] != 1:
        print("Plotting only works for 1D input")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Scatter plot of data
    plt.scatter(X[:, 0], y, alpha=0.5, label="Data points")
    
    # Fitted line
    X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line = model.predict(X_line)
    plt.plot(X_line, y_line, 'r-', linewidth=2, label=f"Fitted: y = {model.w[0]:.3f}x + {model.b:.3f}")
    
    # True line (if provided)
    if true_w is not None and true_b is not None:
        y_true_line = X_line @ true_w + true_b
        plt.plot(X_line, y_true_line, 'g--', linewidth=2, label=f"True: y = {true_w[0]:.3f}x + {true_b:.3f}")
    
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Linear Regression Fit")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("project-1a-optimization/outputs/linear_regression_fit.png", dpi=150, bbox_inches='tight')
    plt.show()


# =============================================================================
# PART 4: Learning Rate Sensitivity Analysis
# =============================================================================

def learning_rate_experiment(X, y, learning_rates=[0.0001, 0.001, 0.01, 0.1, 0.5, 1.0]):
    """
    Demonstrate how learning rate affects training.
    
    Key insights:
    - Too small: convergence is very slow
    - Too large: loss may oscillate or diverge
    - Just right: smooth, fast convergence
    """
    plt.figure(figsize=(12, 5))
    
    for lr in learning_rates:
        try:
            model = LinearRegressionGD(learning_rate=lr, n_iterations=500, verbose=False)
            model.fit(X, y)
            
            # Clip very large losses for visualization
            loss_history = np.clip(model.loss_history, 0, 100)
            plt.plot(loss_history, label=f"lr={lr}")
        except (OverflowError, FloatingPointError):
            print(f"Learning rate {lr} caused overflow")
    
    plt.xlabel("Iteration")
    plt.ylabel("MSE Loss")
    plt.title("Learning Rate Sensitivity")
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("project-1a-optimization/outputs/learning_rate_sensitivity.png", dpi=150, bbox_inches='tight')
    plt.show()


# =============================================================================
# PART 5: Feature Scaling Demonstration
# =============================================================================

def demonstrate_scale_sensitivity():
    """
    Show why optimization is sensitive to feature scale.
    
    When features have very different scales:
    - Gradient magnitudes differ greatly
    - A single learning rate can be too big for some features, too small for others
    - Loss surface becomes elongated (elliptical contours)
    - Convergence is poor
    
    Solution: Normalize/standardize features
    """
    np.random.seed(42)
    
    # Create data with different scales
    n_samples = 100
    X_unscaled = np.column_stack([
        np.random.randn(n_samples) * 100,  # Feature 1: large scale
        np.random.randn(n_samples) * 0.01   # Feature 2: small scale
    ])
    true_w = np.array([0.5, 300])
    y = X_unscaled @ true_w + np.random.randn(n_samples) * 0.1
    
    # Train on unscaled data
    print("=" * 50)
    print("Training on UNSCALED data:")
    print("=" * 50)
    model_unscaled = LinearRegressionGD(learning_rate=0.0001, n_iterations=1000, verbose=True)
    model_unscaled.fit(X_unscaled, y)
    print(f"Final weights: {model_unscaled.w}")
    print(f"True weights: {true_w}")
    
    # Scale the data
    X_scaled = (X_unscaled - X_unscaled.mean(axis=0)) / X_unscaled.std(axis=0)
    y_scaled = (y - y.mean()) / y.std()
    
    print("\n" + "=" * 50)
    print("Training on SCALED data:")
    print("=" * 50)
    model_scaled = LinearRegressionGD(learning_rate=0.1, n_iterations=1000, verbose=True)
    model_scaled.fit(X_scaled, y_scaled)
    print(f"Final weights (scaled space): {model_scaled.w}")
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(model_unscaled.loss_history)
    axes[0].set_title("Unscaled Features (lr=0.0001)")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Loss")
    axes[0].set_yscale('log')
    
    axes[1].plot(model_scaled.loss_history)
    axes[1].set_title("Scaled Features (lr=0.1)")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Loss")
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig("project-1a-optimization/outputs/scale_sensitivity.png", dpi=150, bbox_inches='tight')
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import os
    os.makedirs("project-1a-optimization/outputs", exist_ok=True)
    
    print("=" * 60)
    print("LINEAR REGRESSION WITH GRADIENT DESCENT FROM SCRATCH")
    print("=" * 60)
    
    # Generate data
    print("\n1. Generating synthetic data...")
    X, y, true_w, true_b = generate_linear_data(n_samples=100, n_features=1, noise=0.5)
    print(f"   True weights: {true_w}")
    print(f"   True bias: {true_b:.4f}")
    
    # Train model
    print("\n2. Training linear regression model...")
    model = LinearRegressionGD(learning_rate=0.1, n_iterations=1000)
    model.fit(X, y)
    
    print(f"\n   Learned weights: {model.w}")
    print(f"   Learned bias: {model.b:.4f}")
    print(f"   R² score: {model.score(X, y):.4f}")
    
    # Visualizations
    print("\n3. Creating visualizations...")
    plot_loss_curve(model)
    plot_fit_1d(X, y, model, true_w, true_b)
    
    # Learning rate experiment
    print("\n4. Learning rate sensitivity experiment...")
    learning_rate_experiment(X, y)
    
    # Scale sensitivity
    print("\n5. Feature scaling sensitivity demonstration...")
    demonstrate_scale_sensitivity()
    
    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS:")
    print("=" * 60)
    print("""
    1. GRADIENT DESCENT UPDATE RULE:
       w_new = w_old - learning_rate * gradient
       
    2. MSE GRADIENTS (derived by hand):
       dL/dw = (2/n) * X^T @ (y_pred - y)
       dL/db = (2/n) * sum(y_pred - y)
       
    3. LEARNING RATE MATTERS:
       - Too small: slow convergence
       - Too large: divergence/oscillation
       
    4. FEATURE SCALING MATTERS:
       - Different scales → different gradient magnitudes
       - Hard to find one learning rate that works for all features
       - Solution: normalize/standardize features
       
    5. INITIALIZATION MATTERS:
       - Small random values break symmetry
       - Large values can cause gradient explosion
    """)

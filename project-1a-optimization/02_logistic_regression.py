"""
Logistic Regression with Gradient Descent from Scratch

This implementation adds classification concepts to gradient descent:
1. Sigmoid activation: squash output to (0, 1) for probabilities
2. Binary Cross-Entropy loss: appropriate loss for classification
3. Why softmax + cross-entropy pair well together

No automatic differentiation - we derive and code gradients by hand.
"""

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# PART 1: Core Functions
# =============================================================================

def sigmoid(z):
    """
    Sigmoid activation function: σ(z) = 1 / (1 + e^(-z))
    
    Properties:
    - Output is always in (0, 1) → interpretable as probability
    - Derivative: σ'(z) = σ(z) * (1 - σ(z))
    - Saturates for large |z| → gradient vanishing problem
    
    Numerical stability note:
    - For very large negative z, e^(-z) overflows
    - We clip z to prevent this
    """
    # Clip to prevent overflow
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(sigmoid_output):
    """
    Derivative of sigmoid, given the sigmoid output (not the input).
    
    If s = sigmoid(z), then:
    ds/dz = s * (1 - s)
    
    This is why sigmoid can cause vanishing gradients:
    - When s ≈ 0 or s ≈ 1, the derivative ≈ 0
    - Gradients become tiny in deep networks
    """
    return sigmoid_output * (1 - sigmoid_output)


def binary_cross_entropy(y_pred, y_true, epsilon=1e-15):
    """
    Binary Cross-Entropy Loss (Log Loss)
    
    L = -1/n * Σ [y*log(p) + (1-y)*log(1-p)]
    
    Why this loss for classification?
    1. MSE doesn't work well: gradient is small when prediction is confident but wrong
    2. Cross-entropy penalizes confident wrong predictions heavily
    3. It's derived from maximum likelihood for Bernoulli distribution
    
    Args:
        y_pred: Predicted probabilities (after sigmoid)
        y_true: True labels (0 or 1)
        epsilon: Small value to prevent log(0)
    
    Returns:
        Average loss over all samples
    """
    # Clip predictions to prevent log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Binary cross-entropy formula
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss


# =============================================================================
# PART 2: Generate Classification Data
# =============================================================================

def generate_binary_data(n_samples=200, seed=42):
    """
    Generate linearly separable binary classification data.
    
    Creates two clusters with a linear decision boundary.
    """
    np.random.seed(seed)
    
    n_per_class = n_samples // 2
    
    # Class 0: centered around (-1, -1)
    X0 = np.random.randn(n_per_class, 2) * 0.8 + np.array([-1, -1])
    y0 = np.zeros(n_per_class)
    
    # Class 1: centered around (1, 1)
    X1 = np.random.randn(n_per_class, 2) * 0.8 + np.array([1, 1])
    y1 = np.ones(n_per_class)
    
    # Combine
    X = np.vstack([X0, X1])
    y = np.hstack([y0, y1])
    
    # Shuffle
    shuffle_idx = np.random.permutation(n_samples)
    X, y = X[shuffle_idx], y[shuffle_idx]
    
    return X, y


# =============================================================================
# PART 3: Logistic Regression Class
# =============================================================================

class LogisticRegressionGD:
    """
    Logistic Regression using Gradient Descent.
    
    Model: 
        z = X @ w + b          (linear)
        p = sigmoid(z)         (probability)
    
    Loss: Binary Cross-Entropy
        L = -1/n * Σ [y*log(p) + (1-y)*log(1-p)]
    
    Gradient derivation (the beautiful part):
    ------------------------------------------
    This is why sigmoid + cross-entropy pair well together!
    
    L = -1/n * Σ [y*log(σ(z)) + (1-y)*log(1-σ(z))]
    
    After calculus (chain rule):
    dL/dz = 1/n * Σ (σ(z) - y) = 1/n * Σ (p - y)
    
    The gradient has a simple, elegant form!
    
    Then by chain rule:
    dL/dw = dL/dz * dz/dw = 1/n * X^T @ (p - y)
    dL/db = dL/dz * dz/db = 1/n * Σ (p - y)
    
    This is EXACTLY the same form as linear regression with MSE!
    """
    
    def __init__(self, learning_rate=0.1, n_iterations=1000, verbose=True):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.verbose = verbose
        self.w = None
        self.b = None
        self.loss_history = []
        
    def _init_weights(self, n_features):
        """Initialize weights to small random values."""
        self.w = np.random.randn(n_features) * 0.01
        self.b = 0.0
        
    def _forward(self, X):
        """Forward pass: linear transform + sigmoid."""
        z = X @ self.w + self.b
        return sigmoid(z)
    
    def _compute_gradients(self, X, y_pred, y_true):
        """
        Compute gradients of binary cross-entropy loss.
        
        The key insight: after the math simplifies, we get:
        dL/dw = 1/n * X^T @ (p - y)
        dL/db = 1/n * mean(p - y)
        
        This elegant form is WHY sigmoid + cross-entropy pair well together.
        The sigmoid derivative cancels with the cross-entropy derivative.
        """
        n = len(y_true)
        error = y_pred - y_true  # (p - y)
        
        dw = (1/n) * (X.T @ error)
        db = (1/n) * np.sum(error)
        
        return dw, db
    
    def fit(self, X, y):
        """Train using gradient descent."""
        n_samples, n_features = X.shape
        self._init_weights(n_features)
        
        for i in range(self.n_iterations):
            # Forward pass
            y_pred = self._forward(X)
            
            # Compute loss
            loss = binary_cross_entropy(y_pred, y)
            self.loss_history.append(loss)
            
            # Compute gradients
            dw, db = self._compute_gradients(X, y_pred, y)
            
            # Update weights
            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db
            
            if self.verbose and (i % 100 == 0 or i == self.n_iterations - 1):
                acc = self.accuracy(X, y)
                print(f"Iteration {i:4d} | Loss: {loss:.6f} | Accuracy: {acc:.4f}")
        
        return self
    
    def predict_proba(self, X):
        """Return probability of class 1."""
        return self._forward(X)
    
    def predict(self, X, threshold=0.5):
        """Return predicted class labels."""
        return (self.predict_proba(X) >= threshold).astype(int)
    
    def accuracy(self, X, y):
        """Compute classification accuracy."""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


# =============================================================================
# PART 4: Visualization Functions
# =============================================================================

def plot_decision_boundary(X, y, model, title="Decision Boundary"):
    """Plot data points and decision boundary."""
    plt.figure(figsize=(10, 8))
    
    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    # Get predictions for mesh
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot probability contours
    plt.contourf(xx, yy, Z, levels=50, cmap='RdYlBu', alpha=0.7)
    plt.colorbar(label='P(class=1)')
    
    # Plot decision boundary (p = 0.5)
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
    
    # Plot data points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', 
                         edgecolors='black', s=50)
    
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(title)
    plt.savefig("project-1a-optimization/outputs/logistic_decision_boundary.png", dpi=150, bbox_inches='tight')
    plt.show()


def plot_sigmoid_and_derivative():
    """Visualize sigmoid function and its derivative."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    z = np.linspace(-6, 6, 200)
    s = sigmoid(z)
    s_prime = sigmoid_derivative(s)
    
    # Sigmoid
    axes[0].plot(z, s, 'b-', linewidth=2)
    axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    axes[0].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_xlabel("z")
    axes[0].set_ylabel("σ(z)")
    axes[0].set_title("Sigmoid Function")
    axes[0].grid(True, alpha=0.3)
    
    # Annotate saturation regions
    axes[0].annotate('Saturates\n(gradient ≈ 0)', xy=(-4, 0.02), fontsize=10, color='red')
    axes[0].annotate('Saturates\n(gradient ≈ 0)', xy=(2.5, 0.98), fontsize=10, color='red')
    
    # Derivative
    axes[1].plot(z, s_prime, 'r-', linewidth=2)
    axes[1].axhline(y=0.25, color='gray', linestyle='--', alpha=0.5, label='max = 0.25')
    axes[1].set_xlabel("z")
    axes[1].set_ylabel("σ'(z)")
    axes[1].set_title("Sigmoid Derivative: σ'(z) = σ(z)(1 - σ(z))")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Annotate why this causes vanishing gradients
    axes[1].annotate('Max derivative\nis only 0.25!', xy=(0, 0.25), 
                    xytext=(2, 0.20), fontsize=10,
                    arrowprops=dict(arrowstyle='->', color='black'))
    
    plt.tight_layout()
    plt.savefig("project-1a-optimization/outputs/sigmoid_analysis.png", dpi=150, bbox_inches='tight')
    plt.show()


# =============================================================================
# PART 5: Why Softmax + Cross-Entropy Pair Well Together
# =============================================================================

def demonstrate_softmax_cross_entropy():
    """
    Explain and demonstrate why softmax + cross-entropy is the standard combo.
    
    For multi-class classification:
    - Softmax: turns logits into probabilities that sum to 1
    - Cross-entropy: penalizes wrong confident predictions
    
    Together, they have an elegant gradient:
    dL/dz_i = p_i - y_i  (where y_i is 1 for the true class, 0 otherwise)
    
    This is the multi-class generalization of what we saw with sigmoid + BCE!
    """
    print("\n" + "=" * 60)
    print("WHY SOFTMAX + CROSS-ENTROPY PAIR WELL TOGETHER")
    print("=" * 60)
    
    def softmax(z):
        """
        Softmax function: converts logits to probabilities.
        
        softmax(z_i) = exp(z_i) / Σ exp(z_j)
        
        Properties:
        - Output sums to 1 (valid probability distribution)
        - Larger logits → higher probability
        - Differentiable everywhere
        """
        # Subtract max for numerical stability (prevents overflow)
        z_stable = z - np.max(z, axis=-1, keepdims=True)
        exp_z = np.exp(z_stable)
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)
    
    def cross_entropy_loss(probs, targets):
        """
        Cross-entropy loss for multi-class classification.
        
        L = -Σ y_i * log(p_i)
        
        In practice, only the true class contributes (others have y_i = 0).
        """
        epsilon = 1e-15
        probs = np.clip(probs, epsilon, 1 - epsilon)
        return -np.sum(targets * np.log(probs))
    
    # Example: 3-class classification
    print("\nExample: 3-class classification")
    print("-" * 40)
    
    # Logits (raw model output)
    z = np.array([2.0, 1.0, 0.1])
    print(f"Logits (raw output): {z}")
    
    # Softmax converts to probabilities
    probs = softmax(z)
    print(f"Softmax probabilities: {probs.round(4)}")
    print(f"Sum of probabilities: {probs.sum():.4f}")
    
    # True label is class 0
    y_true = np.array([1, 0, 0])  # One-hot encoding
    print(f"True label (one-hot): {y_true}")
    
    # Cross-entropy loss
    loss = cross_entropy_loss(probs, y_true)
    print(f"Cross-entropy loss: {loss:.4f}")
    
    # The magic: gradient simplifies to (p - y)
    gradient = probs - y_true
    print(f"\nGradient dL/dz = p - y: {gradient.round(4)}")
    
    print("""
    KEY INSIGHT:
    ------------
    The gradient of (softmax + cross-entropy) w.r.t. the logits z is simply:
    
        dL/dz = p - y
    
    This elegant form happens because:
    1. Cross-entropy derivative has log terms
    2. Softmax derivative has exp terms
    3. They cancel out, leaving just (p - y)!
    
    This is computationally stable and avoids numerical issues that would
    arise if we computed the gradients separately.
    
    SAME PATTERN AS SIGMOID + BCE:
    Binary case: dL/dz = σ(z) - y
    Multi-class: dL/dz = softmax(z) - y
    """)


# =============================================================================
# PART 6: Vanishing Gradient Demonstration
# =============================================================================

def demonstrate_vanishing_gradients():
    """
    Show why sigmoid causes vanishing gradients in deep networks.
    """
    print("\n" + "=" * 60)
    print("WHY GRADIENTS VANISH WITH SIGMOID")
    print("=" * 60)
    
    print("""
    Consider a deep network with sigmoid activations:
    
    Layer 1: h1 = σ(W1 @ x)
    Layer 2: h2 = σ(W2 @ h1)
    Layer 3: h3 = σ(W3 @ h2)
    ...
    
    By chain rule, gradient for W1 includes:
    dL/dW1 = dL/dh3 * dh3/dh2 * dh2/dh1 * dh1/dW1
    
    Each dh_i/dh_{i-1} includes σ'(·), which is at most 0.25!
    """)
    
    # Simulate gradient flow through layers
    n_layers = 10
    gradient = 1.0  # Start with gradient of 1
    
    print(f"\nSimulating gradient flow through {n_layers} sigmoid layers:")
    print(f"(Assuming σ'(z) = 0.25 at each layer, which is the MAXIMUM)")
    print("-" * 40)
    
    gradients = [gradient]
    for i in range(n_layers):
        # Best case: sigmoid derivative at maximum (0.25)
        gradient = gradient * 0.25
        gradients.append(gradient)
        print(f"After layer {i+1}: gradient magnitude = {gradient:.2e}")
    
    # Plot
    plt.figure(figsize=(10, 5))
    plt.semilogy(range(len(gradients)), gradients, 'b-o', linewidth=2, markersize=8)
    plt.xlabel("Layer")
    plt.ylabel("Gradient Magnitude (log scale)")
    plt.title("Vanishing Gradients Through Sigmoid Layers (Best Case)")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=1e-6, color='r', linestyle='--', label='Numerically insignificant')
    plt.legend()
    plt.savefig("project-1a-optimization/outputs/vanishing_gradients.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"""
    After {n_layers} layers, even with BEST-CASE sigmoid derivatives,
    the gradient shrinks to {gradient:.2e}
    
    In practice, it's often worse because:
    1. Sigmoid rarely operates at z=0 where derivative is max
    2. Weight matrices can further reduce gradient magnitude
    
    SOLUTIONS:
    1. ReLU: f(x) = max(0, x), derivative is 0 or 1 (no shrinking!)
    2. Residual connections: gradient can flow directly through skip connections
    3. Better initialization: Xavier/He initialization keeps variance stable
    4. Batch normalization: keeps activations in good range
    """)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import os
    os.makedirs("project-1a-optimization/outputs", exist_ok=True)
    
    print("=" * 60)
    print("LOGISTIC REGRESSION WITH GRADIENT DESCENT FROM SCRATCH")
    print("=" * 60)
    
    # Generate data
    print("\n1. Generating binary classification data...")
    X, y = generate_binary_data(n_samples=200)
    print(f"   Shape: {X.shape}, Labels: {np.unique(y)}")
    
    # Train model
    print("\n2. Training logistic regression model...")
    model = LogisticRegressionGD(learning_rate=0.5, n_iterations=1000)
    model.fit(X, y)
    
    print(f"\n   Final accuracy: {model.accuracy(X, y):.4f}")
    print(f"   Learned weights: {model.w}")
    print(f"   Learned bias: {model.b:.4f}")
    
    # Visualizations
    print("\n3. Creating visualizations...")
    
    # Loss curve
    plt.figure(figsize=(10, 4))
    plt.plot(model.loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.title("Training Loss Curve")
    plt.grid(True, alpha=0.3)
    plt.savefig("project-1a-optimization/outputs/logistic_loss.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # Decision boundary
    plot_decision_boundary(X, y, model)
    
    # Sigmoid analysis
    print("\n4. Analyzing sigmoid function...")
    plot_sigmoid_and_derivative()
    
    # Softmax + cross-entropy explanation
    print("\n5. Understanding softmax + cross-entropy...")
    demonstrate_softmax_cross_entropy()
    
    # Vanishing gradients
    print("\n6. Understanding vanishing gradients...")
    demonstrate_vanishing_gradients()
    
    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS:")
    print("=" * 60)
    print("""
    1. SIGMOID FUNCTION:
       σ(z) = 1 / (1 + e^(-z))
       - Squashes any input to (0, 1)
       - Derivative: σ'(z) = σ(z) * (1 - σ(z))
       - Max derivative is 0.25 → causes vanishing gradients!
       
    2. BINARY CROSS-ENTROPY:
       L = -[y*log(p) + (1-y)*log(1-p)]
       - Penalizes confident wrong predictions heavily
       - Derived from maximum likelihood
       
    3. WHY SIGMOID + CROSS-ENTROPY PAIR WELL:
       The combined gradient simplifies to: dL/dz = p - y
       - No numerical instability from separate computations
       - Same elegant form as linear regression!
       
    4. WHY SOFTMAX + CROSS-ENTROPY PAIR WELL:
       Same reason for multi-class: dL/dz = softmax(z) - y
       - Probabilities sum to 1
       - Gradient is simple and stable
       
    5. WHY GRADIENTS VANISH:
       - Sigmoid derivative ≤ 0.25
       - Multiply through many layers → gradient shrinks exponentially
       - Solution: ReLU, skip connections, careful initialization
    """)

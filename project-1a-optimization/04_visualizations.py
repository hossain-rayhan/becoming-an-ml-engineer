"""
Visualization Suite for Optimization Experiments

This module creates comprehensive visualizations for:
1. Learning rate sensitivity
2. Loss curves and convergence behavior
3. Gradient flow analysis
4. Initialization effects
5. Loss surface visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Import our implementations
from utils import generate_linear_data, generate_binary_data


# =============================================================================
# PART 1: Learning Rate Sensitivity Experiments
# =============================================================================

def learning_rate_sweep_linear():
    """
    Comprehensive learning rate experiment for linear regression.
    Shows the full spectrum: too small, just right, too large.
    """
    print("\n" + "=" * 60)
    print("LEARNING RATE SENSITIVITY: LINEAR REGRESSION")
    print("=" * 60)
    
    # Generate data
    np.random.seed(42)
    X = np.random.randn(100, 1)
    true_w = np.array([2.5])
    true_b = 1.0
    y = X @ true_w + true_b + np.random.randn(100) * 0.3
    
    learning_rates = [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 1.5, 2.0]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, lr in enumerate(learning_rates):
        # Train
        w = np.random.randn(1) * 0.01
        b = 0.0
        losses = []
        
        for i in range(200):
            y_pred = X @ w + b
            loss = np.mean((y_pred - y) ** 2)
            
            # Clip loss for stability
            if loss > 1e10:
                losses.append(1e10)
                break
            losses.append(loss)
            
            # Gradients
            error = y_pred - y
            dw = (2/len(y)) * (X.T @ error)
            db = (2/len(y)) * np.sum(error)
            
            # Update
            w = w - lr * dw
            b = b - lr * db
        
        # Plot
        axes[idx].plot(losses)
        axes[idx].set_title(f'lr = {lr}')
        axes[idx].set_xlabel('Iteration')
        axes[idx].set_ylabel('Loss')
        axes[idx].set_yscale('log')
        axes[idx].grid(True, alpha=0.3)
        
        # Annotate behavior
        if lr <= 0.001:
            axes[idx].text(0.5, 0.95, 'TOO SLOW', transform=axes[idx].transAxes,
                          fontsize=10, color='orange', va='top', ha='center')
        elif lr >= 1.5:
            axes[idx].text(0.5, 0.95, 'DIVERGING', transform=axes[idx].transAxes,
                          fontsize=10, color='red', va='top', ha='center')
        elif 0.01 <= lr <= 0.5:
            axes[idx].text(0.5, 0.95, 'GOOD', transform=axes[idx].transAxes,
                          fontsize=10, color='green', va='top', ha='center')
    
    plt.suptitle('Learning Rate Sensitivity Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('project-1a-optimization/outputs/lr_sensitivity_full.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("""
    Observations:
    - lr = 0.0001, 0.001: Very slow convergence, needs many more iterations
    - lr = 0.01, 0.1: Good convergence, smooth loss curves
    - lr = 0.5: Fast but starting to oscillate
    - lr = 1.0: On the edge, oscillating
    - lr = 1.5, 2.0: Diverging! Loss explodes
    
    Rule of thumb: Start with lr=0.01 or 0.001 and tune from there.
    """)


def learning_rate_effect_on_path():
    """
    Visualize how learning rate affects the optimization path in 2D.
    """
    print("\n" + "=" * 60)
    print("OPTIMIZATION PATHS WITH DIFFERENT LEARNING RATES")
    print("=" * 60)
    
    # Simple quadratic loss: L(w,b) = (w-2)^2 + (b-3)^2
    # Minimum at (w=2, b=3)
    
    def loss(w, b):
        return (w - 2)**2 + (b - 3)**2
    
    def grad(w, b):
        return 2*(w - 2), 2*(b - 3)
    
    # Create mesh for contour plot
    w_range = np.linspace(-2, 6, 100)
    b_range = np.linspace(-1, 7, 100)
    W, B = np.meshgrid(w_range, b_range)
    L = loss(W, B)
    
    # Starting point
    w0, b0 = -1.0, 6.0
    
    learning_rates = [0.05, 0.2, 0.5, 0.9]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for idx, lr in enumerate(learning_rates):
        ax = axes[idx]
        
        # Plot contours
        ax.contour(W, B, L, levels=30, cmap='Blues', alpha=0.7)
        ax.plot(2, 3, 'r*', markersize=15, label='Optimum')
        
        # Run gradient descent
        w, b = w0, b0
        path_w, path_b = [w], [b]
        
        for _ in range(30):
            dw, db = grad(w, b)
            w = w - lr * dw
            b = b - lr * db
            path_w.append(w)
            path_b.append(b)
            
            # Stop if diverging
            if abs(w) > 10 or abs(b) > 10:
                break
        
        # Plot path
        ax.plot(path_w, path_b, 'ko-', markersize=4, linewidth=1, alpha=0.7)
        ax.plot(path_w[0], path_b[0], 'go', markersize=10, label='Start')
        
        ax.set_title(f'lr = {lr}')
        ax.set_xlabel('w')
        ax.set_ylabel('b')
        ax.legend()
        ax.set_xlim(-2, 6)
        ax.set_ylim(-1, 7)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Optimization Paths for Different Learning Rates', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('project-1a-optimization/outputs/optimization_paths.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("""
    Key observations:
    - Small lr (0.05): Direct path but slow
    - Medium lr (0.2): Good convergence
    - Large lr (0.5): Oscillates around optimum
    - Very large lr (0.9): Even more oscillation, near instability
    """)


# =============================================================================
# PART 2: Loss Surface Visualization
# =============================================================================

def visualize_loss_surface_3d():
    """
    3D visualization of loss surface for linear regression.
    Shows why optimization is sensitive to scale.
    """
    print("\n" + "=" * 60)
    print("3D LOSS SURFACE VISUALIZATION")
    print("=" * 60)
    
    # Simple 1D linear regression: y = wx + b
    # Generate data where true w=2, b=1
    np.random.seed(42)
    X = np.random.randn(50, 1)
    y = 2 * X.squeeze() + 1 + np.random.randn(50) * 0.3
    
    # Compute MSE loss for range of w, b values
    w_range = np.linspace(-1, 5, 50)
    b_range = np.linspace(-2, 4, 50)
    W, B = np.meshgrid(w_range, b_range)
    
    Loss = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            y_pred = X.squeeze() * W[i,j] + B[i,j]
            Loss[i,j] = np.mean((y_pred - y)**2)
    
    # Create 3D surface plot
    fig = plt.figure(figsize=(14, 5))
    
    # Surface plot
    ax1 = fig.add_subplot(131, projection='3d')
    surf = ax1.plot_surface(W, B, Loss, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('w')
    ax1.set_ylabel('b')
    ax1.set_zlabel('Loss')
    ax1.set_title('Loss Surface (3D)')
    
    # Contour plot
    ax2 = fig.add_subplot(132)
    contour = ax2.contour(W, B, Loss, levels=30, cmap='viridis')
    ax2.plot(2, 1, 'r*', markersize=15, label='True minimum')
    ax2.set_xlabel('w')
    ax2.set_ylabel('b')
    ax2.set_title('Loss Surface (Contours)')
    ax2.legend()
    
    # Log loss for better visualization
    ax3 = fig.add_subplot(133)
    contour_log = ax3.contour(W, B, np.log(Loss + 1e-10), levels=30, cmap='viridis')
    ax3.plot(2, 1, 'r*', markersize=15, label='True minimum')
    ax3.set_xlabel('w')
    ax3.set_ylabel('b')
    ax3.set_title('Log Loss Surface (Contours)')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('project-1a-optimization/outputs/loss_surface_3d.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("""
    The loss surface shows:
    - There's a single minimum (convex for linear regression)
    - The contours are elliptical
    - Gradient descent follows the steepest descent direction
    - If the ellipse is elongated (bad scaling), optimization is harder
    """)


def visualize_ill_conditioned_loss():
    """
    Show why poorly scaled features create elongated loss surfaces.
    """
    print("\n" + "=" * 60)
    print("ILL-CONDITIONED LOSS SURFACE (POOR SCALING)")
    print("=" * 60)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Well-scaled problem
    def loss_well_scaled(w1, w2):
        return (w1 - 1)**2 + (w2 - 2)**2
    
    # Ill-conditioned (poorly scaled) problem
    def loss_ill_conditioned(w1, w2):
        return 100*(w1 - 1)**2 + (w2 - 2)**2  # One dimension 100x more sensitive
    
    w1_range = np.linspace(-2, 4, 100)
    w2_range = np.linspace(-1, 5, 100)
    W1, W2 = np.meshgrid(w1_range, w2_range)
    
    # Well-scaled
    L_well = loss_well_scaled(W1, W2)
    axes[0].contour(W1, W2, L_well, levels=30, cmap='viridis')
    axes[0].plot(1, 2, 'r*', markersize=15)
    axes[0].set_xlabel('w1')
    axes[0].set_ylabel('w2')
    axes[0].set_title('Well-Scaled (circular contours)\nEasy to optimize!')
    axes[0].set_aspect('equal')
    
    # Ill-conditioned
    L_ill = loss_ill_conditioned(W1, W2)
    axes[1].contour(W1, W2, L_ill, levels=30, cmap='viridis')
    axes[1].plot(1, 2, 'r*', markersize=15)
    axes[1].set_xlabel('w1')
    axes[1].set_ylabel('w2')
    axes[1].set_title('Ill-Conditioned (elongated contours)\nHard to optimize!')
    axes[1].set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('project-1a-optimization/outputs/ill_conditioned.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("""
    Why ill-conditioning hurts optimization:
    
    1. Different features have vastly different gradient magnitudes
    2. A single learning rate can't work well for all dimensions
    3. Need tiny lr for large-gradient dims → super slow for small-gradient dims
    4. Need large lr for small-gradient dims → diverge on large-gradient dims
    
    Solutions:
    - Feature normalization/standardization
    - Adaptive learning rates (Adam, AdaGrad)
    - Second-order methods (Newton's method)
    - Batch normalization
    """)


# =============================================================================
# PART 3: Initialization Effects
# =============================================================================

def initialization_comparison():
    """
    Show how initialization affects training.
    """
    print("\n" + "=" * 60)
    print("INITIALIZATION EFFECTS ON TRAINING")
    print("=" * 60)
    
    # Generate data
    np.random.seed(42)
    X = np.random.randn(100, 5)  # 5 features
    true_w = np.array([1.0, -0.5, 2.0, -1.5, 0.3])
    true_b = 0.5
    y = X @ true_w + true_b + np.random.randn(100) * 0.2
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    inits = [
        ('Zero init', lambda n: (np.zeros(n), 0.0)),
        ('Small random (0.01)', lambda n: (np.random.randn(n) * 0.01, 0.0)),
        ('Xavier/He init', lambda n: (np.random.randn(n) * np.sqrt(2.0/n), 0.0)),
        ('Large random (10)', lambda n: (np.random.randn(n) * 10, 0.0)),
    ]
    
    for idx, (name, init_fn) in enumerate(inits):
        np.random.seed(42)  # Reset for fair comparison
        w, b = init_fn(5)
        lr = 0.01
        losses = []
        
        for _ in range(500):
            y_pred = X @ w + b
            loss = np.mean((y_pred - y)**2)
            losses.append(loss)
            
            # Cap for plotting
            if loss > 1e6:
                break
            
            error = y_pred - y
            dw = (2/len(y)) * (X.T @ error)
            db = (2/len(y)) * np.sum(error)
            
            w = w - lr * dw
            b = b - lr * db
        
        axes[idx].plot(losses)
        axes[idx].set_title(name)
        axes[idx].set_xlabel('Iteration')
        axes[idx].set_ylabel('Loss')
        axes[idx].set_yscale('log')
        axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle('Effect of Weight Initialization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('project-1a-optimization/outputs/initialization_effects.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("""
    Initialization insights:
    
    1. Zero init: Works for linear regression, but FAILS in neural networks
       (all neurons learn the same thing - symmetry problem)
    
    2. Small random: Good starting point, avoids symmetry
    
    3. Xavier/He init: Keeps variance stable across layers
       - Xavier: for tanh/sigmoid: std = sqrt(2 / (fan_in + fan_out))
       - He: for ReLU: std = sqrt(2 / fan_in)
    
    4. Large random: Can cause gradient explosion at the start
    
    For neural networks, proper initialization is CRUCIAL!
    """)


# =============================================================================
# PART 4: Gradient Behavior Analysis
# =============================================================================

def gradient_magnitude_during_training():
    """
    Track gradient magnitudes during training.
    """
    print("\n" + "=" * 60)
    print("GRADIENT MAGNITUDE DURING TRAINING")
    print("=" * 60)
    
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = X @ np.array([1.0, 2.0, 3.0]) + 0.5 + np.random.randn(100) * 0.1
    
    w = np.random.randn(3) * 0.01
    b = 0.0
    lr = 0.01
    
    losses = []
    grad_magnitudes = []
    weight_magnitudes = []
    
    for _ in range(500):
        y_pred = X @ w + b
        loss = np.mean((y_pred - y)**2)
        losses.append(loss)
        
        error = y_pred - y
        dw = (2/len(y)) * (X.T @ error)
        db = (2/len(y)) * np.sum(error)
        
        grad_magnitudes.append(np.linalg.norm(dw))
        weight_magnitudes.append(np.linalg.norm(w))
        
        w = w - lr * dw
        b = b - lr * db
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(losses)
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('MSE')
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(grad_magnitudes)
    axes[1].set_title('Gradient Magnitude')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('||∇w||')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(weight_magnitudes)
    axes[2].set_title('Weight Magnitude')
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('||w||')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('project-1a-optimization/outputs/gradient_tracking.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("""
    What we observe:
    1. Loss decreases (goal achieved!)
    2. Gradient magnitude decreases (approaching minimum where gradient = 0)
    3. Weight magnitude grows until reaching true values
    
    When gradients EXPLODE: watch for sudden increases in gradient magnitude
    When gradients VANISH: watch for gradients becoming numerically zero
    """)


# =============================================================================
# PART 5: Comparison of Optimizers
# =============================================================================

def compare_sgd_vs_momentum():
    """
    Compare vanilla SGD with momentum.
    """
    print("\n" + "=" * 60)
    print("SGD VS MOMENTUM")
    print("=" * 60)
    
    # Ill-conditioned problem where momentum helps
    def loss(w):
        return 100*w[0]**2 + w[1]**2  # Elongated bowl
    
    def grad(w):
        return np.array([200*w[0], 2*w[1]])
    
    w_init = np.array([1.0, 10.0])
    lr = 0.01
    
    # Vanilla SGD
    w_sgd = w_init.copy()
    path_sgd = [w_sgd.copy()]
    losses_sgd = [loss(w_sgd)]
    
    for _ in range(100):
        g = grad(w_sgd)
        w_sgd = w_sgd - lr * g
        path_sgd.append(w_sgd.copy())
        losses_sgd.append(loss(w_sgd))
    
    # SGD with momentum
    w_mom = w_init.copy()
    v = np.zeros_like(w_mom)  # velocity
    momentum = 0.9
    path_mom = [w_mom.copy()]
    losses_mom = [loss(w_mom)]
    
    for _ in range(100):
        g = grad(w_mom)
        v = momentum * v - lr * g
        w_mom = w_mom + v
        path_mom.append(w_mom.copy())
        losses_mom.append(loss(w_mom))
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Paths
    w1_range = np.linspace(-1.5, 1.5, 100)
    w2_range = np.linspace(-2, 12, 100)
    W1, W2 = np.meshgrid(w1_range, w2_range)
    L = 100*W1**2 + W2**2
    
    axes[0].contour(W1, W2, L, levels=30, cmap='viridis', alpha=0.7)
    
    path_sgd = np.array(path_sgd)
    path_mom = np.array(path_mom)
    
    axes[0].plot(path_sgd[:, 0], path_sgd[:, 1], 'b.-', label='SGD', alpha=0.7)
    axes[0].plot(path_mom[:, 0], path_mom[:, 1], 'r.-', label='Momentum', alpha=0.7)
    axes[0].plot(0, 0, 'g*', markersize=15, label='Optimum')
    axes[0].plot(w_init[0], w_init[1], 'ko', markersize=10, label='Start')
    axes[0].set_xlabel('w1')
    axes[0].set_ylabel('w2')
    axes[0].set_title('Optimization Paths')
    axes[0].legend()
    
    # Loss curves
    axes[1].plot(losses_sgd, 'b-', label='SGD')
    axes[1].plot(losses_mom, 'r-', label='Momentum')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Convergence')
    axes[1].set_yscale('log')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('project-1a-optimization/outputs/sgd_vs_momentum.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("""
    Momentum advantages:
    1. Accelerates in consistent gradient directions
    2. Dampens oscillations in inconsistent directions
    3. Helps escape shallow local minima
    4. Critical for training deep networks
    
    Physics analogy: Ball rolling down a hill accumulates velocity
    """)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import os
    os.makedirs("project-1a-optimization/outputs", exist_ok=True)
    
    print("=" * 60)
    print("OPTIMIZATION VISUALIZATION SUITE")
    print("=" * 60)
    
    # Run all visualizations
    learning_rate_sweep_linear()
    learning_rate_effect_on_path()
    visualize_loss_surface_3d()
    visualize_ill_conditioned_loss()
    initialization_comparison()
    gradient_magnitude_during_training()
    compare_sgd_vs_momentum()
    
    print("\n" + "=" * 60)
    print("ALL VISUALIZATIONS COMPLETE!")
    print("=" * 60)
    print("\nSaved images to project-1a-optimization/outputs/")

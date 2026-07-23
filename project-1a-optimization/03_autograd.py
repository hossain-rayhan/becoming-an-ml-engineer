"""
Tiny Autograd Engine: Computational Graph from Scratch

This implements automatic differentiation using reverse-mode autodiff (backpropagation).
Inspired by Karpathy's micrograd, but built to understand the concepts.

Key concepts:
1. Computational graph: DAG of operations
2. Forward pass: compute values
3. Backward pass: compute gradients using chain rule
4. Topological sort: ensure gradients flow in correct order

This is how PyTorch, TensorFlow, and JAX work under the hood!
"""

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# PART 1: The Value Class - Core of Autograd
# =============================================================================

class Value:
    """
    A Value wraps a scalar and tracks the computational graph.
    
    Key attributes:
    - data: the actual numerical value
    - grad: the gradient (dL/d(this_value)), initialized to 0
    - _backward: a function to compute the gradient for this node's inputs
    - _prev: the Values that were used to create this Value
    - _op: the operation that created this Value (for visualization)
    
    The magic is in _backward:
    When we call backward(), it traverses the graph in reverse topological order,
    calling each node's _backward function to propagate gradients.
    """
    
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0  # Gradient accumulates here
        self._backward = lambda: None  # Default: do nothing
        self._prev = set(_children)  # Parent nodes
        self._op = _op  # Operation name (for debugging/visualization)
        self.label = label
        
    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
    
    # -------------------------------------------------------------------------
    # Basic Operations - Each defines how gradient flows backward
    # -------------------------------------------------------------------------
    
    def __add__(self, other):
        """
        Addition: out = self + other
        
        Gradient flow:
        d(out)/d(self) = 1
        d(out)/d(other) = 1
        
        So gradients just pass through unchanged.
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            # Gradient of addition: both inputs receive the output gradient
            # (multiplied by the local gradient, which is 1)
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        """
        Multiplication: out = self * other
        
        Gradient flow:
        d(out)/d(self) = other.data
        d(out)/d(other) = self.data
        
        The gradient of each input is the value of the OTHER input.
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        
        out._backward = _backward
        return out
    
    def __pow__(self, n):
        """
        Power: out = self ** n
        
        Gradient flow:
        d(out)/d(self) = n * self.data^(n-1)
        
        This is just the power rule from calculus!
        """
        assert isinstance(n, (int, float)), "Only supporting int/float powers"
        out = Value(self.data ** n, (self,), f'**{n}')
        
        def _backward():
            self.grad += n * (self.data ** (n - 1)) * out.grad
        
        out._backward = _backward
        return out
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __truediv__(self, other):
        return self * (other ** -1)
    
    def __radd__(self, other):
        return self + other
    
    def __rmul__(self, other):
        return self * other
    
    def __rsub__(self, other):
        return (-self) + other
    
    def __rtruediv__(self, other):
        return (self ** -1) * other
    
    # -------------------------------------------------------------------------
    # Activation Functions
    # -------------------------------------------------------------------------
    
    def relu(self):
        """
        ReLU: out = max(0, self)
        
        Gradient flow:
        d(out)/d(self) = 1 if self > 0, else 0
        
        ReLU is popular because:
        1. Gradient is 0 or 1 (no vanishing!)
        2. Computationally simple
        3. Induces sparsity
        """
        out = Value(max(0, self.data), (self,), 'ReLU')
        
        def _backward():
            self.grad += (self.data > 0) * out.grad
        
        out._backward = _backward
        return out
    
    def sigmoid(self):
        """
        Sigmoid: out = 1 / (1 + exp(-self))
        
        Gradient flow:
        d(out)/d(self) = out * (1 - out)
        
        Remember: this saturates for large |self|, causing vanishing gradients!
        """
        s = 1 / (1 + np.exp(-self.data))
        out = Value(s, (self,), 'σ')
        
        def _backward():
            # Sigmoid derivative: s * (1 - s)
            self.grad += (out.data * (1 - out.data)) * out.grad
        
        out._backward = _backward
        return out
    
    def tanh(self):
        """
        Tanh: out = (exp(2*self) - 1) / (exp(2*self) + 1)
        
        Gradient flow:
        d(out)/d(self) = 1 - out^2
        """
        t = np.tanh(self.data)
        out = Value(t, (self,), 'tanh')
        
        def _backward():
            self.grad += (1 - out.data ** 2) * out.grad
        
        out._backward = _backward
        return out
    
    def exp(self):
        """
        Exp: out = exp(self)
        
        Gradient flow:
        d(out)/d(self) = exp(self) = out
        
        The derivative of exp is itself!
        """
        out = Value(np.exp(self.data), (self,), 'exp')
        
        def _backward():
            self.grad += out.data * out.grad
        
        out._backward = _backward
        return out
    
    def log(self):
        """
        Log: out = log(self)
        
        Gradient flow:
        d(out)/d(self) = 1 / self
        """
        out = Value(np.log(self.data), (self,), 'log')
        
        def _backward():
            self.grad += (1 / self.data) * out.grad
        
        out._backward = _backward
        return out
    
    # -------------------------------------------------------------------------
    # Backward Pass - The Heart of Backpropagation
    # -------------------------------------------------------------------------
    
    def backward(self):
        """
        Compute gradients for all Values in the graph.
        
        Algorithm:
        1. Topological sort: order nodes so that parents come before children
        2. Set this node's gradient to 1 (it's dL/dL = 1)
        3. Walk backward through the graph, calling _backward on each node
        
        Why topological sort?
        A node needs all its consumer gradients before it can compute its own.
        Topological sort ensures we process nodes in the right order.
        """
        # Build topological order
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        
        # Set gradient of output to 1 (dL/dL = 1)
        self.grad = 1.0
        
        # Backpropagate in reverse topological order
        for node in reversed(topo):
            node._backward()


# =============================================================================
# PART 2: Neural Network Components
# =============================================================================

class Neuron:
    """
    A single neuron: y = activation(sum(w_i * x_i) + b)
    """
    
    def __init__(self, n_inputs, activation='relu'):
        # Initialize weights with small random values
        self.w = [Value(np.random.randn() * 0.3) for _ in range(n_inputs)]
        self.b = Value(0.0)
        self.activation = activation
        
    def __call__(self, x):
        # Sum of w_i * x_i + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        
        # Apply activation
        if self.activation == 'relu':
            return act.relu()
        elif self.activation == 'tanh':
            return act.tanh()
        elif self.activation == 'sigmoid':
            return act.sigmoid()
        else:  # linear
            return act
    
    def parameters(self):
        return self.w + [self.b]


class Layer:
    """A layer of neurons."""
    
    def __init__(self, n_inputs, n_outputs, activation='relu'):
        self.neurons = [Neuron(n_inputs, activation) for _ in range(n_outputs)]
        
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    """
    Multi-Layer Perceptron: stack of fully connected layers.
    """
    
    def __init__(self, n_inputs, layer_sizes):
        """
        Args:
            n_inputs: number of input features
            layer_sizes: list of layer sizes, e.g. [4, 4, 1] for 2 hidden + output
        """
        sizes = [n_inputs] + layer_sizes
        self.layers = []
        for i in range(len(layer_sizes)):
            # Last layer is linear (no activation) for regression
            # Hidden layers use relu
            activation = 'linear' if i == len(layer_sizes) - 1 else 'relu'
            self.layers.append(Layer(sizes[i], sizes[i+1], activation))
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


# =============================================================================
# PART 3: Demonstration - Train a Neural Network
# =============================================================================

def train_neural_network():
    """
    Train a small MLP using our autograd engine.
    """
    print("\n" + "=" * 60)
    print("TRAINING NEURAL NETWORK WITH TINY AUTOGRAD ENGINE")
    print("=" * 60)
    
    # Generate simple XOR-like data
    np.random.seed(42)
    X = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ]
    # XOR: 1 if inputs differ, 0 if same
    y_true = [0.0, 1.0, 1.0, 0.0]
    
    # Create MLP: 2 inputs -> 4 hidden -> 4 hidden -> 1 output
    model = MLP(2, [4, 4, 1])
    
    print(f"Number of parameters: {len(model.parameters())}")
    
    # Training loop
    learning_rate = 0.1
    losses = []
    
    for epoch in range(500):
        # Forward pass for all samples
        total_loss = Value(0.0)
        
        for xi, yi in zip(X, y_true):
            # Convert inputs to Values
            x_values = [Value(xij) for xij in xi]
            
            # Forward pass
            y_pred = model(x_values)
            
            # MSE loss
            loss = (y_pred - yi) ** 2
            total_loss = total_loss + loss
        
        # Average loss
        total_loss = total_loss * (1.0 / len(X))
        losses.append(total_loss.data)
        
        # Backward pass
        # Reset gradients first!
        for p in model.parameters():
            p.grad = 0.0
        
        total_loss.backward()
        
        # Update parameters (gradient descent)
        for p in model.parameters():
            p.data -= learning_rate * p.grad
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch:4d} | Loss: {total_loss.data:.6f}")
    
    # Final predictions
    print("\nFinal predictions:")
    for xi, yi in zip(X, y_true):
        x_values = [Value(xij) for xij in xi]
        y_pred = model(x_values)
        print(f"  Input: {xi} -> Predicted: {y_pred.data:.4f}, True: {yi}")
    
    # Plot loss
    plt.figure(figsize=(10, 4))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss (Tiny Autograd Engine)")
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.savefig("project-1a-optimization/outputs/autograd_training.png", dpi=150, bbox_inches='tight')
    plt.show()


# =============================================================================
# PART 4: Visualize Computational Graph
# =============================================================================

def visualize_forward_backward():
    """
    Step through a simple computation to see how it works.
    """
    print("\n" + "=" * 60)
    print("STEP-BY-STEP FORWARD AND BACKWARD PASS")
    print("=" * 60)
    
    # Simple computation: L = (a * b + c)^2
    a = Value(2.0, label='a')
    b = Value(3.0, label='b')
    c = Value(1.0, label='c')
    
    print("Forward pass:")
    print(f"  a = {a.data}")
    print(f"  b = {b.data}")
    print(f"  c = {c.data}")
    
    # Step 1: d = a * b
    d = a * b
    d.label = 'd = a*b'
    print(f"  d = a * b = {d.data}")
    
    # Step 2: e = d + c
    e = d + c
    e.label = 'e = d+c'
    print(f"  e = d + c = {e.data}")
    
    # Step 3: L = e^2
    L = e ** 2
    L.label = 'L = e^2'
    print(f"  L = e^2 = {L.data}")
    
    # Backward pass
    print("\nBackward pass (computing gradients):")
    L.backward()
    
    print(f"  dL/dL = {L.grad} (by definition)")
    print(f"  dL/de = 2*e = 2*{e.data} = {e.grad}")
    print(f"  dL/dd = dL/de * de/dd = {e.grad} * 1 = {d.grad}")
    print(f"  dL/dc = dL/de * de/dc = {e.grad} * 1 = {c.grad}")
    print(f"  dL/da = dL/dd * dd/da = {d.grad} * b = {d.grad}*{b.data} = {a.grad}")
    print(f"  dL/db = dL/dd * dd/db = {d.grad} * a = {d.grad}*{a.data} = {b.grad}")
    
    # Verify manually
    print("\nManual verification:")
    print(f"  L = (a*b + c)^2")
    print(f"  dL/da = 2*(a*b + c) * b = 2*({a.data}*{b.data} + {c.data}) * {b.data}")
    expected_da = 2 * (a.data * b.data + c.data) * b.data
    print(f"        = {expected_da} ✓" if abs(a.grad - expected_da) < 1e-10 else f"        = {expected_da} ✗")


def compare_with_numerical_gradient():
    """
    Verify our analytical gradients match numerical gradients.
    """
    print("\n" + "=" * 60)
    print("GRADIENT CHECK: ANALYTICAL VS NUMERICAL")
    print("=" * 60)
    
    def compute_loss(a_val, b_val, c_val):
        """Compute L = (a*b + c)^2"""
        return (a_val * b_val + c_val) ** 2
    
    # Values
    a_val, b_val, c_val = 2.0, 3.0, 1.0
    h = 1e-5  # Small step for numerical gradient
    
    # Numerical gradient (finite differences)
    print("Numerical gradients (finite differences):")
    num_grad_a = (compute_loss(a_val + h, b_val, c_val) - compute_loss(a_val - h, b_val, c_val)) / (2 * h)
    num_grad_b = (compute_loss(a_val, b_val + h, c_val) - compute_loss(a_val, b_val - h, c_val)) / (2 * h)
    num_grad_c = (compute_loss(a_val, b_val, c_val + h) - compute_loss(a_val, b_val, c_val - h)) / (2 * h)
    print(f"  dL/da (numerical) = {num_grad_a:.6f}")
    print(f"  dL/db (numerical) = {num_grad_b:.6f}")
    print(f"  dL/dc (numerical) = {num_grad_c:.6f}")
    
    # Analytical gradient (our autograd)
    a = Value(a_val)
    b = Value(b_val)
    c = Value(c_val)
    L = (a * b + c) ** 2
    L.backward()
    
    print("\nAnalytical gradients (our autograd):")
    print(f"  dL/da (autograd) = {a.grad:.6f}")
    print(f"  dL/db (autograd) = {b.grad:.6f}")
    print(f"  dL/dc (autograd) = {c.grad:.6f}")
    
    # Check match
    print("\nDifference (should be ~0):")
    print(f"  |num - ana| for a: {abs(num_grad_a - a.grad):.2e}")
    print(f"  |num - ana| for b: {abs(num_grad_b - b.grad):.2e}")
    print(f"  |num - ana| for c: {abs(num_grad_c - c.grad):.2e}")


# =============================================================================
# PART 5: Understand Chain Rule Deeply
# =============================================================================

def explain_chain_rule():
    """
    Deep dive into why the chain rule is the key to backpropagation.
    """
    print("\n" + "=" * 60)
    print("THE CHAIN RULE: BACKBONE OF BACKPROPAGATION")
    print("=" * 60)
    
    print("""
    The Chain Rule from Calculus:
    -----------------------------
    If y = f(g(x)), then dy/dx = f'(g(x)) * g'(x)
    
    Or more generally:
    If z = f(y) and y = g(x), then dz/dx = (dz/dy) * (dy/dx)
    
    
    In Neural Networks:
    -------------------
    Consider a simple network:
    
        x → [W1] → h → [W2] → y → [Loss] → L
    
    We want dL/dW1 to update W1.
    
    By chain rule:
        dL/dW1 = dL/dy * dy/dh * dh/dW1
    
    BACKPROPAGATION = applying chain rule layer by layer, from output to input
    
    
    Why It's Called "Backpropagation":
    ----------------------------------
    1. FORWARD: x → h → y → L
       - Compute values left to right
       
    2. BACKWARD: L → y → h → W1
       - Compute gradients right to left
       - Each node "propagates" its gradient to its inputs
    
    
    The Beautiful Pattern:
    ----------------------
    At each node, gradient computation is:
    
        local_gradient = d(output) / d(input)
        
        incoming_gradient = gradient flowing from the loss
        
        gradient_to_pass_back = local_gradient × incoming_gradient
    
    This is exactly what our _backward functions do!
    """)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import os
    os.makedirs("project-1a-optimization/outputs", exist_ok=True)
    
    print("=" * 60)
    print("TINY AUTOGRAD ENGINE: COMPUTATIONAL GRAPH FROM SCRATCH")
    print("=" * 60)
    
    # Step-by-step forward/backward
    visualize_forward_backward()
    
    # Verify gradients
    compare_with_numerical_gradient()
    
    # Explain chain rule
    explain_chain_rule()
    
    # Train a neural network
    train_neural_network()
    
    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS:")
    print("=" * 60)
    print("""
    1. COMPUTATIONAL GRAPH:
       - Each operation creates a node
       - Nodes track their inputs (parents)
       - The graph is built dynamically during forward pass
       
    2. FORWARD PASS:
       - Compute values from inputs to outputs
       - Store intermediate values (needed for backward)
       
    3. BACKWARD PASS (BACKPROPAGATION):
       - Start from output with gradient = 1
       - Apply chain rule at each node
       - Accumulate gradients (+=) because a value may be used multiple times
       - Topological sort ensures correct order
       
    4. LOCAL GRADIENTS:
       Each operation has a simple local gradient:
       - Addition: ∂/∂x (x + y) = 1
       - Multiplication: ∂/∂x (x * y) = y
       - Power: ∂/∂x (x^n) = n * x^(n-1)
       - ReLU: ∂/∂x max(0,x) = 1 if x > 0, else 0
       - Sigmoid: ∂/∂x σ(x) = σ(x)(1 - σ(x))
       
    5. WHY THIS MATTERS:
       - This is exactly how PyTorch/TensorFlow work (just optimized)
       - Understanding this lets you debug gradient issues
       - Helps you design custom operations
       - Explains why certain architectures train better
    """)

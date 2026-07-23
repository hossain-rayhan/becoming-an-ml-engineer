# Concepts You Must Be Able to Explain

This document covers the core concepts from Project 1A that you need to articulate clearly in interviews.

---

## Key Terms

### Regression

**Regression** means predicting a continuous numerical value—something that can be any number on a scale (e.g., 72.5, 142.30, -3.7).

Examples: house prices, temperatures, stock prices, someone's age.

**Linear regression** assumes the relationship between input and output is a straight line:
$$y = wx + b$$

We use gradient descent to find the $w$ (slope) and $b$ (intercept) that minimize prediction error.

### Classification

**Classification** means predicting a category or label—a discrete choice from a fixed set of options.

Examples: spam/not spam, cat/dog/bird, digits 0-9.

---

## 1. Why Gradients Vanish or Explode

### The Problem

In deep networks, the backward pass multiplies gradients through many layers:

$$\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial h_n} \cdot \frac{\partial h_n}{\partial h_{n-1}} \cdot \ldots \cdot \frac{\partial h_2}{\partial h_1} \cdot \frac{\partial h_1}{\partial w_1}$$

Each layer contributes a multiplicative factor.

### Vanishing Gradients

**Cause:** When these factors are < 1, the product shrinks exponentially.

**Example with Sigmoid:**
- Sigmoid derivative: $\sigma'(z) = \sigma(z)(1-\sigma(z))$
- Maximum value is 0.25 (when z = 0)
- After 10 layers: $0.25^{10} = 9.5 \times 10^{-7}$

**Symptoms:**
- Earlier layers learn much slower than later layers
- Weights barely update
- Training stalls

**Solutions:**
1. **ReLU activation**: Derivative is 0 or 1 (no shrinking in active region)
2. **Residual connections**: Gradient can flow directly through skip connections
3. **Proper initialization**: Xavier/He keeps variance stable
4. **Batch normalization**: Keeps activations in a good range
5. **LSTM/GRU**: Gating mechanisms for sequences

### Exploding Gradients

**Cause:** When multiplicative factors are > 1, the product grows exponentially.

**Symptoms:**
- NaN or Inf in loss
- Weights grow unbounded
- Training becomes unstable

**Solutions:**
1. **Gradient clipping**: `grad = min(grad, threshold)`
2. **Lower learning rate**
3. **Proper initialization**
4. **Weight regularization**

### Interview Answer Template

> "Vanishing gradients occur when gradients shrink as they backpropagate through many layers, because each layer multiplies the gradient by values less than 1. With sigmoid activations, the maximum derivative is 0.25, so after 10 layers the gradient can shrink by a factor of a million. This causes earlier layers to learn extremely slowly.
>
> Solutions include using ReLU (which has derivative 0 or 1), skip connections (which let gradients flow directly), and proper initialization like He init. Batch normalization also helps by keeping activations in a range where gradients are reasonable."

---

## 2. Why Softmax + Cross-Entropy Pair Well Together

### The Individual Components

**Softmax** converts logits to probabilities:
$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

**Cross-Entropy Loss** measures how wrong the predictions are:
$$L = -\sum_i y_i \log(p_i)$$

### Why They Pair Well

When combined, the gradient simplifies:

$$\frac{\partial L}{\partial z_i} = p_i - y_i$$

This is remarkably elegant:
- **Numerically stable**: No separate computation of softmax derivative
- **Computationally efficient**: Simple subtraction
- **Intuitive**: Gradient is just "prediction minus target"

### The Math Behind It

Without going through the full derivation, the key insight is:

1. Cross-entropy has $\log(p)$ terms
2. Softmax has $\exp(z)$ terms
3. The derivative of $\log(\exp(z))$ simplifies cleanly

If we tried to use softmax with MSE loss, we'd get a much messier, less stable gradient.

### Same Pattern in Binary Case

**Sigmoid + Binary Cross-Entropy:**
$$\frac{\partial L}{\partial z} = \sigma(z) - y$$

Same elegant form! This isn't a coincidence—both arise from maximum likelihood estimation of categorical distributions.

### Interview Answer Template

> "Softmax and cross-entropy pair well because their combined gradient simplifies to just $(p - y)$—the prediction minus the true label. This happens because the log in cross-entropy cancels nicely with the exp in softmax during differentiation.
>
> This pairing is also numerically stable. Computing softmax and its derivative separately can cause overflow issues, but the combined gradient avoids these problems. That's why frameworks implement them together as a single operation.
>
> The same principle applies to sigmoid with binary cross-entropy—we get the same elegant $(\sigma(z) - y)$ gradient. Both cases come from maximum likelihood estimation, which is why the math works out so cleanly."

---

## 3. Why Optimization is Sensitive to Scale and Initialization

### Scale Sensitivity

**The Problem:**
When features have different scales, the loss surface becomes elongated (ill-conditioned).

**Example:**
- Feature 1: values in range [0, 1000]
- Feature 2: values in range [0, 1]

The gradient for Feature 1 will be ~1000x larger than for Feature 2.

**Consequences:**
- Can't find a single learning rate that works for both
- Small lr for Feature 1 → painfully slow on Feature 2
- Large lr for Feature 2 → diverges on Feature 1
- Optimization zigzags instead of going straight to minimum

**Solutions:**
1. **Feature normalization**: Standardize to zero mean, unit variance
2. **Batch normalization**: Normalize within the network
3. **Adaptive learning rates**: Adam, AdaGrad adapt per-parameter

### Initialization Sensitivity

**Why It Matters:**
- **Zero initialization**: All neurons compute the same thing (symmetry problem)
- **Too large**: Initial activations saturate, gradients vanish or explode
- **Too small**: Signal shrinks through layers, vanishing gradients

**Proper Initialization Strategies:**

**Xavier/Glorot (for tanh, sigmoid):**
$$\text{std} = \sqrt{\frac{2}{n_{in} + n_{out}}}$$

**He (for ReLU):**
$$\text{std} = \sqrt{\frac{2}{n_{in}}}$$

**Why These Work:**
They keep the variance of activations roughly constant across layers, preventing exponential growth or decay.

### Interview Answer Template

> "Optimization is sensitive to scale because different-scale features create elongated loss surfaces. If one feature ranges from 0-1000 and another from 0-1, their gradients differ by orders of magnitude. A single learning rate can't handle both—too small is slow, too large diverges.
>
> The solution is feature normalization or adaptive optimizers like Adam that maintain per-parameter learning rates.
>
> Initialization matters because it determines the starting point and how signals flow through the network. Zero init breaks symmetry—all neurons learn the same thing. Too large causes saturation; too small causes signal decay.
>
> He initialization with variance $2/n_{in}$ works well for ReLU networks because it maintains roughly constant variance through layers, preventing both vanishing and exploding activations."

---

## 4. The Gradient Descent Update Rule

The fundamental equation:
$$w_{t+1} = w_t - \eta \cdot \nabla_w L$$

**Components:**
- $w_t$: Current parameters
- $\eta$: Learning rate
- $\nabla_w L$: Gradient of loss with respect to parameters

**Intuition:**
- Gradient points in the direction of steepest increase
- Negative gradient points toward decrease
- We take a step in that direction
- Learning rate controls step size

**Variants:**
- **SGD**: Basic update, can be noisy
- **Momentum**: Accumulates velocity, smooths updates
- **Adam**: Adapts learning rate per parameter

---

## 5. How Backpropagation Works

**Core Idea:** Apply chain rule layer by layer, from output to input.

**Algorithm:**
1. **Forward pass**: Compute all activations
2. **Compute output gradient**: $\frac{\partial L}{\partial y}$
3. **Backward pass**: At each layer:
   - Compute local gradient
   - Multiply by incoming gradient
   - Pass to previous layer

**Code Pattern:**
```python
# At each node
local_gradient = d(output) / d(input)
gradient_to_pass_back = local_gradient * incoming_gradient
```

**Why It's Efficient:**
- Each gradient computed exactly once
- Intermediate values cached during forward pass
- Total work is O(forward pass)

---

## Summary Table

| Concept | Key Point | Solution |
|---------|-----------|----------|
| Vanishing gradients | Multiplicative < 1 factors compound | ReLU, skip connections, proper init |
| Exploding gradients | Multiplicative > 1 factors compound | Gradient clipping, lower lr |
| Softmax + CE | Gradient simplifies to (p - y) | Always use together |
| Scale sensitivity | Elongated loss surface | Normalize features, use Adam |
| Initialization | Affects signal flow through layers | Xavier/He initialization |

---

## Practice Questions

1. "Your deep network isn't learning—loss barely moves after 1000 epochs. What do you check?"
2. "Why don't we use MSE loss for classification?"
3. "You're training with lr=0.1 and loss oscillates wildly. What do you try?"
4. "Explain why BatchNorm helps training."
5. "What happens if you initialize all weights to zero?"

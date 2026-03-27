# Neural Networks Overview

A beginner-friendly guide to understanding neural networks, their components, and how they learn.

---

## What Is a Neural Network?

A neural network is a program that learns patterns from examples instead of being explicitly programmed.

### The Core Idea

Imagine you want a program to recognize handwritten digits (0-9).

**Traditional programming approach:**
Write rules like "if the image has a loop at the top, it might be a 9 or 6..."

This is extremely hard. Handwriting varies too much.

**Neural network approach:**
Show the program thousands of examples of handwritten digits with their correct labels. The program figures out the patterns on its own.

### Structure: Layers of Neurons

A neural network is organized in layers:

```
Input Layer → Hidden Layer(s) → Output Layer
```

**Input layer:** receives the raw data
- For a 28×28 pixel image: 784 input numbers (one per pixel)

**Hidden layers:** do the pattern detection
- Each layer finds increasingly complex patterns
- Layer 1 might detect edges
- Layer 2 might detect curves and corners
- Layer 3 might detect digit shapes

**Output layer:** gives the final answer
- For digit recognition: 10 outputs (one per digit 0-9)
- The highest output is the network's guess

### What Is a Neuron?

Each neuron does something simple:

1. Takes inputs from the previous layer
2. Multiplies each input by a weight
3. Adds them up
4. Adds a bias
5. Passes through an activation function

$$
\text{output} = f(w_1 x_1 + w_2 x_2 + \dots + b)
$$

One neuron is not smart. But thousands of neurons working together can recognize faces, translate languages, or play games.

### Why "Neural"?

The design was loosely inspired by how brain neurons work:
- Brain neurons receive signals from other neurons
- If the total signal is strong enough, they fire
- Learning changes the connection strengths

Neural networks are a simplified, mathematical version of this idea.

### The Simplest Mental Model

```
data → [learned function] → prediction
```

A neural network is just a flexible function with millions of adjustable knobs (weights and biases). Training finds the knob settings that make the function work for your task.

---

## Weights and Biases

Weights and biases are the learnable parameters of a neural network—the values that get adjusted during training.

### The Neuron Equation

A neuron performs two steps:

1. Combine inputs into one number
2. Pass that number through an activation function

The combination step:

$$
z = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b
$$

The output:

$$
a = f(z)
$$

Where:
- $x_1, x_2, \dots, x_n$ are the inputs
- $w_1, w_2, \dots, w_n$ are the weights
- $b$ is the bias
- $f$ is the activation function

### What Weights Mean

Weights control how important each input is.

- Large positive weight → that input pushes the neuron output up
- Large negative weight → that input pushes the neuron output down
- Weight near zero → that input barely matters

**Example:**

$$
z = 2x_1 - 3x_2 + 0.1x_3 + b
$$

This means:
- $x_1$ has a strong positive effect
- $x_2$ has a strong negative effect
- $x_3$ has very little effect

Weights are the main way the network learns which patterns matter.

### What Bias Means

Bias is an extra adjustable value added after the weighted sum. It lets the neuron shift its decision boundary.

Without bias:
$$
z = w_1 x_1 + w_2 x_2
$$

With bias:
$$
z = w_1 x_1 + w_2 x_2 + b
$$

**Simple intuition:**
- Weights control the slope or direction
- Bias controls the offset or threshold

### Real-World Intuition

Suppose a neuron decides whether an email is spam.

Inputs might be:
- $x_1$ = number of suspicious words
- $x_2$ = whether sender is unknown
- $x_3$ = number of links

Then:
$$
z = 1.5x_1 + 2.0x_2 + 0.8x_3 - 4
$$

Here:
- The weights say how much each signal matters
- The bias $-4$ says the neuron needs enough total evidence before it activates

Bias acts like a baseline threshold.

### Geometric Intuition

For one input:
$$
y = wx + b
$$

This is just a line:
- $w$ changes the slope
- $b$ moves the line up and down

For multiple inputs:
- Weights determine orientation of the decision boundary
- Bias determines position of the boundary

### Why Both Are Needed

If you only had weights and no bias, the neuron's decision boundary would always pass through the origin. That's too restrictive.

Bias gives the model flexibility to fit real patterns.

### In Plain Language

- **Weights** = "how strongly should I care about each input?"
- **Bias** = "how easy should it be for this neuron to activate at all?"

### Tiny Example

Suppose:
$$
z = 3x - 6
$$

Using a step activation that turns on when $z > 0$:

$$
3x - 6 > 0 \implies x > 2
$$

- Weight $3$ controls how sharply input affects output
- Bias $-6$ sets the threshold at $x = 2$

Without the bias, activation would happen at $x > 0$, which may be wrong for the task.

---

## Activation Functions

Activation functions introduce non-linearity into neural networks, allowing them to learn complex patterns.

### Why Activation Functions?

Without activation functions, a neural network is just a series of linear operations:

$$
\text{Layer 1: } z_1 = W_1 x + b_1
$$
$$
\text{Layer 2: } z_2 = W_2 z_1 + b_2 = W_2(W_1 x + b_1) + b_2
$$

This simplifies to another linear function. No matter how many layers you stack, you just get a linear transformation.

**Activation functions break this linearity**, letting networks learn curved decision boundaries and complex patterns.

### Common Activation Functions

| Function | Formula | Typical Use |
|----------|---------|-------------|
| ReLU | $f(x) = \max(0, x)$ | Most hidden layers (default choice) |
| Sigmoid | $f(x) = \frac{1}{1+e^{-x}}$ | Binary classification output |
| Tanh | $f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$ | Sometimes in hidden layers |
| Softmax | Converts to probabilities | Multi-class output layer |
| GELU | Smooth ReLU variant | Transformers like GPT |
| SiLU/Swish | $f(x) = x \cdot \sigma(x)$ | Modern architectures |

### ReLU (Rectified Linear Unit)

$$
f(x) = \max(0, x)
$$

- Returns 0 for negative inputs
- Returns the input unchanged for positive inputs
- Simple and fast to compute
- Most popular choice for hidden layers

### Sigmoid

$$
f(x) = \frac{1}{1 + e^{-x}}
$$

- Squashes any input to range (0, 1)
- Useful for binary classification outputs
- Historically popular, now mostly used in output layers

### Softmax

$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$

- Converts a vector of numbers into probabilities
- All outputs sum to 1
- Used for multi-class classification

### Different Layers Can Use Different Activations

A common pattern:

```
Input → [Linear + ReLU] → [Linear + ReLU] → [Linear + Softmax] → Output
```

- Hidden layers: ReLU (or GELU, SiLU in modern networks)
- Output layer: depends on task
  - Multi-class classification: Softmax
  - Binary classification: Sigmoid
  - Regression: often no activation (linear output)

### Do Activation Functions Change During Training?

**No.** Activation functions are fixed design choices.

| Component | Changes During Training? |
|-----------|-------------------------|
| Weights | Yes |
| Biases | Yes |
| Activation function | No (fixed by design) |
| Network architecture | No (fixed by design) |

You choose the activation functions when designing the network. Training only adjusts weights and biases.

---

## Loss Functions (Cost Functions)

Before a network can learn, it needs a way to measure how wrong its predictions are. This is what loss functions do.

### The Problem

A neural network makes a prediction. How do we quantify "how bad" that prediction is?

**Example:**
- True label: 7
- Network's prediction: 3
- How wrong is this? We need a number.

The loss function converts the difference between prediction and truth into a single number. **Lower loss = better predictions.**

### Why We Need Loss Functions

Training is an optimization problem:
- We want to find weights that make good predictions
- "Good predictions" means low loss
- Gradient descent minimizes the loss

Without a loss function, we have no target to optimize.

### Mean Squared Error (MSE)

Used for regression (predicting continuous values).

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

Where:
- $y_i$ = true value
- $\hat{y}_i$ = predicted value
- $n$ = number of examples

**Intuition:**
- Compute the difference between prediction and truth
- Square it (makes all errors positive, penalizes large errors more)
- Average across all examples

**Example:**

| True | Predicted | Error | Squared Error |
|------|-----------|-------|---------------|
| 10 | 8 | 2 | 4 |
| 5 | 6 | -1 | 1 |
| 3 | 3 | 0 | 0 |

$$
\text{MSE} = \frac{4 + 1 + 0}{3} = 1.67
$$

### Cross-Entropy Loss

Used for classification (predicting categories).

**Binary Cross-Entropy** (two classes):

$$
\text{BCE} = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
$$

Where:
- $y_i$ = true label (0 or 1)
- $\hat{y}_i$ = predicted probability (between 0 and 1)

**Intuition:**
- If true label is 1, we want predicted probability near 1
- If true label is 0, we want predicted probability near 0
- Cross-entropy heavily penalizes confident wrong predictions

**Example:**

| True Label | Predicted Prob | Loss Contribution |
|------------|----------------|-------------------|
| 1 | 0.9 | $-\log(0.9) = 0.105$ (low, good!) |
| 1 | 0.1 | $-\log(0.1) = 2.303$ (high, bad!) |
| 0 | 0.2 | $-\log(0.8) = 0.223$ (low, good!) |

**Categorical Cross-Entropy** (multiple classes):

$$
\text{CCE} = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)
$$

Used when there are more than two classes (e.g., classifying digits 0-9).

### Which Loss Function to Use?

| Task | Loss Function |
|------|---------------|
| Regression (predict a number) | Mean Squared Error (MSE) |
| Binary classification (yes/no) | Binary Cross-Entropy |
| Multi-class classification | Categorical Cross-Entropy |

### Loss vs. Accuracy

**Loss** and **accuracy** are different:

- **Accuracy:** Percentage of correct predictions (easy to understand)
- **Loss:** Continuous measure of prediction quality (used for training)

Why use loss instead of accuracy for training?
- Accuracy doesn't tell you *how wrong* you were
- Accuracy isn't differentiable (can't compute gradients)
- Loss provides a smooth surface for optimization

**Example:**

Two models predicting "cat" (true label = cat):
- Model A: 51% cat, 49% dog → Correct, but barely confident
- Model B: 99% cat, 1% dog → Correct, very confident

Both have the same accuracy (100%), but Model B has lower loss. Loss captures this difference; accuracy doesn't.

### Visualizing Loss

Think of loss as a landscape:
- Weights and biases define a position in this landscape
- The height at each position is the loss value
- Training means walking downhill to find the lowest point

```
        Loss
          ^
          |    /\
          |   /  \      /\
          |  /    \    /  \
          | /      \  /    \
          |/        \/      \____
          +-------------------------> Weights
                    ^
                    |
              Goal: Find this valley
```

The next section explains how gradient descent navigates this landscape.

---

## Gradient Descent

Gradient descent is the algorithm that teaches a neural network by adjusting weights and biases to reduce the loss.

### The Learning Problem

A neural network starts with random weights and biases. It makes terrible predictions.

**Goal:** Find weight and bias values that make good predictions.

**Challenge:** A network might have millions of parameters. How do you find the right values?

**Answer:** Gradient descent.

### The Core Idea

1. **Make a prediction** with current weights
2. **Measure the error** (how wrong the prediction was)
3. **Calculate the gradient** (which direction to adjust each weight)
4. **Update weights** in the direction that reduces error
5. **Repeat** thousands of times

### What Is a Gradient?

The gradient tells you:
- Which direction increases the error
- How steeply the error changes for each weight

To reduce error, you move in the **opposite direction** of the gradient.

Think of it like finding the bottom of a valley while blindfolded:
- Feel the slope under your feet (compute gradient)
- Take a step downhill (update weights)
- Repeat until you reach the bottom (minimum error)

### The Update Rule

For each weight:

$$
w_{\text{new}} = w_{\text{old}} - \eta \cdot \frac{\partial L}{\partial w}
$$

Where:
- $w$ is a weight
- $\eta$ (eta) is the learning rate—a small positive number (like 0.01) that controls how big each adjustment step is
- $\frac{\partial L}{\partial w}$ is the gradient of the loss with respect to that weight
- $\partial$ (partial) denotes a partial derivative—it measures how much $L$ changes when you nudge $w$ slightly, while keeping all other weights fixed

The same applies to biases:

$$
b_{\text{new}} = b_{\text{old}} - \eta \cdot \frac{\partial L}{\partial b}
$$

### Learning Rate

The learning rate $\eta$ controls how big each step is:

- **Too large:** Overshoots the minimum, training becomes unstable
- **Too small:** Takes forever to converge
- **Just right:** Steady progress toward minimum error

Typical values: 0.001, 0.01, 0.0001

The goal of training is to minimize the loss (covered in detail in the Loss Functions section above).

### Backpropagation

Backpropagation is the algorithm that computes gradients efficiently.

It works backward through the network:
1. Compute error at the output
2. Propagate error back through each layer
3. Calculate how much each weight contributed to the error
4. Use chain rule from calculus

This is why it's called "back" propagation—gradients flow backward from output to input.

### Variants of Gradient Descent

**Batch Gradient Descent:**
- Use all training data to compute gradient
- Slow for large datasets

**Stochastic Gradient Descent (SGD):**
- Use one example at a time
- Noisy but fast

**Mini-batch Gradient Descent:**
- Use a small batch (e.g., 32 or 64 examples)
- Best of both worlds—most common in practice

### Modern Optimizers

Plain gradient descent has limitations. Modern optimizers improve on it:

| Optimizer | Key Idea |
|-----------|----------|
| SGD with Momentum | Accumulates velocity to smooth updates |
| Adam | Adapts learning rate per parameter |
| AdamW | Adam with better weight decay |

Adam is the default choice for most deep learning today.

### The Training Loop

```python
for epoch in range(num_epochs):
    for batch in training_data:
        # Forward pass
        predictions = model(batch.inputs)
        
        # Compute loss
        loss = loss_function(predictions, batch.labels)
        
        # Backward pass (compute gradients)
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Clear gradients for next iteration
        optimizer.zero_grad()
```

### Analogy: Learning to Throw Darts

You're learning to hit the bullseye:
1. Throw a dart (make prediction)
2. See how far you missed (compute loss)
3. Adjust your aim based on the miss (compute gradient)
4. Throw again with adjusted aim (update weights)
5. Repeat until you consistently hit the target

Gradient descent is the mathematical version of this trial-and-error learning.

---

## Summary Table

| Term | Meaning |
|------|---------|
| Neural Network | A function with learnable parameters that finds patterns in data |
| Neuron | A unit that computes a weighted sum + bias + activation |
| Weight | How much one input affects the neuron |
| Bias | A threshold adjustment for the neuron |
| Layer | A group of neurons at the same depth |
| Activation Function | Adds non-linearity (ReLU, sigmoid, etc.) |
| Loss Function | Measures how wrong predictions are (MSE, Cross-Entropy) |
| Forward Pass | Running input through the network to get output |
| Gradient | Direction and magnitude of steepest increase in loss |
| Gradient Descent | Algorithm that adjusts weights to minimize loss |
| Backpropagation | Efficient algorithm to compute gradients |
| Learning Rate | Step size for weight updates |
| Epoch | One complete pass through the training data |
| Batch | A subset of training examples processed together |


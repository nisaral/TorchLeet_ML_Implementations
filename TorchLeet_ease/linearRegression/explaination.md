# Linear Regression

Linear Regression is a supervised machine learning algorithm used to predict a continuous target variable y based on one or more input features X. The model assumes a linear relationship between the inputs and the targets, expressed as:

$$y = w \cdot X + b$$

- $w$: Weight(s) that scale the input features.
- $b$: Bias term to shift the linear function.

**Goal:** Find the optimal $w$ and $b$ that minimize the difference between predicted and actual values.

In simple terms, I define Linear Regression as "Glorified Line fitting over the curve".

The loss function used is Mean Squared Error (MSE):

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$$

where $y_i$ is the actual target, $\hat{y}_i$ is the predicted target, and $n$ is the number of data points.

## PyTorch Modules

`torch.nn.Module` is the base class for all neural network models in PyTorch. It provides:

- A way to define layers (e.g., `nn.Linear` for linear transformations).
- Automatic tracking of parameters (weights and biases) for optimization.
- Methods like `forward()` to define how input data flows through the model.

When you subclass `nn.Module`, you define:

- **`__init__`**: Initialize layers and parameters.
- **`forward`**: Define the computation performed on input data.

## Forward Pass

The forward pass is the process of passing input data through the model to compute predictions. For Linear Regression:

Input $X$ is multiplied by weights $w$, and bias $b$ is added: $\hat{y} = w \cdot X + b$.

The model outputs predictions $\hat{y}$, which are compared to actual targets $y$ to compute the loss.

## Backward Pass and Backpropagation

The backward pass computes gradients of the loss function with respect to the model's parameters ($w$ and $b$) using backpropagation. Backpropagation applies the chain rule to propagate errors backward through the network:

1. Compute the loss: $\text{Loss} = \text{MSE}(y, \hat{y})$.
2. Call `loss.backward()` to compute gradients ($\frac{\partial \text{Loss}}{\partial w}, \frac{\partial \text{Loss}}{\partial b}$).
3. Update parameters using an optimizer (e.g., Stochastic Gradient Descent, SGD): $w \gets w - \eta \cdot \frac{\partial \text{Loss}}{\partial w}$, where $\eta$ is the learning rate.

## Optimizer

The optimizer (e.g., `optim.SGD`) updates the model's parameters based on gradients. SGD updates parameters iteratively to minimize the loss:

$$\theta \gets \theta - \eta \cdot \nabla_\theta \text{Loss}$$

where $\theta$ represents parameters ($w, b$), and $\nabla_\theta \text{Loss}$ is the gradient.

## Training Loop

The training loop iterates over:

1. **Forward pass**: Compute predictions and loss.
2. **Backward pass**: Compute gradients.
3. **Optimization**: Update parameters.
4. **Logging**: Monitor loss to track training progress.

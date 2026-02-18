## 1. Basic Knowledge of Perceptrons

**The perceptron is a linear classifier proposed by Frank Rosenblatt**
  - Other linear classifiers include:
    - Logistic regression
    - Support vector machines
    - Fisher linear discriminant

**Mathematical definition of the original perceptron**:

$$
f(\mathbf{x}) = \text{sign}(\mathbf{w}^T \mathbf{x} + b)
$$

where $T$ represents the transpose of the weight vector, and $\text{sign}(x)$ is the step function, defined as:

$$
\text{sign}(x) = \begin{cases}
-1, & x < 0 \\
0, & x = 0 \\
1, & x > 0
\end{cases}
$$

### Mathematical Derivation of the Original Perceptron

Let

$$
g(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + b
$$

Then

$$
f(\mathbf{x}) = \text{sign}(g(\mathbf{x}))
$$

From $g(\mathbf{x})$, we can see it's a weighted sum of inputs followed by a translation via bias $b$.

From $\text{sign}(\mathbf{x})$, it can be viewed as:

$$
\text{sign}(x) = \begin{cases}
    \frac{x}{|x|}, & x \neq 0 \\
    0, & x = 0
\end{cases}
$$

Therefore, $f(\mathbf{x})$ can be seen as only taking the sign of $g(\mathbf{x})$.

### Differentiability of Perceptrons

Differentiability is an important feature of modern perceptrons and neural networks, which means they can be optimized using methods like gradient descent. (This will be explained in detail later)

In our example of the original perceptron, $f(\mathbf{x})$ is a composite function, and its differentiation can be expressed as:

$$
\frac{\mathrm{d} f(\mathbf{x})}{\mathrm{d} \mathbf{x}} = \frac{\mathrm{d} \text{sign}(g(\mathbf{x}))}{\mathrm{d} g(\mathbf{x})} \times \frac{\mathrm{d} g(\mathbf{x})}{\mathrm{d} \mathbf{x}}
$$

We show them using the ***"chain rule"***, where the derivative of $\text{sign}(x)$ is:

$$
\frac{\mathrm{d} \text{sign}(x)}{\mathrm{d} x} = \begin{cases}
  0, & x \neq 0 \\
  \text{undefined}, & x = 0
\end{cases}
$$

Therefore, the derivative of $f(\mathbf{x})$ is:

$$
\frac{\mathrm{d} f(\mathbf{x})}{\mathrm{d} \mathbf{x}} = \begin{cases}
  0, & x \neq 0 \\
  \text{undefined}, & x = 0
\end{cases}
$$

This leads to the inability to obtain effective derivatives (i.e., the "gradient" in "gradient descent") when optimizing $f(\mathbf{x})$.

To address this issue, $\text{sigmoid}(x)$ (the S-shaped growth curve) was introduced to replace $\text{sign}(x)$ as the activation function. It is a function with range $y \in (0, 1)$, defined as:

$$
\text{sigmoid}(x) = \frac{1}{1 + e^{-x}}
$$

Its derivative is:

$$
\frac{\mathrm{d} \text{sigmoid}(x)}{\mathrm{d} x} = \text{sigmoid}(x)[1-\text{sigmoid}(x)] = \frac{e^{-x}}{(1 + e^{-x})^{2}}
$$

With this differentiable and non-zero activation function, we can now optimize $f(\mathbf{x})$.

---

## 2. Gradient Descent Optimization

**Gradient descent is a function minimization optimization algorithm proposed by Augustin-Louis Cauchy**

### Mathematical Derivation of Gradient Descent

Let $f(x)$ be a differentiable unknown function, and $n$ be the optimization iteration count:

$$
f(x_{n+1}) = f(x_{n}) + (x_{n+1} - x_{n}) \frac{f(x_{n+1}) - f(x_{n})}{x_{n+1} - x_{n}}
$$

To minimize $f(x)$, we need to find the $x$ that minimizes the function. Observing the above equation, we need the following term to be negative:

$$
(x_{n+1} - x_{n}) \frac{f(x_{n+1}) - f(x_{n})}{x_{n+1} - x_{n}} < 0
$$

Simplifying and introducing the update coefficient (later serving as the learning rate) $\gamma$:

$$
(x_{n+1} - x_{n}) f'(x_{n}) < 0 \\
x_{n+1} - x_{n} = - \gamma f'(x_{n}) \\
x_{n+1} = x_{n} - \gamma f'(x_{n})
$$

In summary, to find $x_{n+1}$ that minimizes $f(x_{n+1})$, we need to know the current $x_{n}$ and its corresponding derivatives $f'(x_{n})$.

---

## 3. Practical Implementation

Based on the two sections above, we now have a preliminary understanding of perceptrons and gradient descent. Now, we combine them.

Modern perceptron:

$$
\begin{align*}
  & g(x) = \mathbf{w}^T \mathbf{x} + b \\
  & f(\mathbf{x}) = \text{sigmoid}(g(\mathbf{x}))
\end{align*}
$$

### Loss Function

In machine learning optimization, a "loss function" is typically used to represent the discrepancy between the model's predictions and the actual data. A lower loss means better fitting of data. Common loss functions include "mean squared error" for regression models (actual values) and "cross-entropy" for prediction models (probabilities).

In common binary classification tasks, we use cross-entropy to calculate the difference between predicted and actual probabilities.

Cross-entropy:

$$
H(P, Q) = -\sum_{i} P_{i} \log Q_{i}
$$

Loss function based on cross-entropy:

$$
L = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log(p_{i,c})
$$

where $N$ is the number of samples and $C$ is the number of classes.

In binary classification, the sum of probabilities for the two classes is 1. Therefore, assuming

$$
P_1 = 0.6
$$

then

$$
P_2 = 1 - 0.6 = 0.4
$$

Thus, the cross-entropy loss function for binary classification can be written as:

$$
L = - y \log p + (1-y)\log(1-p)
$$

Since $y$ represents the true sample (one-hot encoded sample, where the probability of the corresponding label is 1) but it is also a variable, we need to obtain the partial derivative of the loss function $L$ with respect to the variable $p$:

$$
\frac{\partial L}{\partial p} = - \frac{y}{p} + \frac{1-y}{1-p}
$$

Since the output of the perceptron is $f(\mathbf{x}) \in (0,1)$, it can be treated as probability $p$.

Therefore,

$$
\begin{align*}
  \frac{\partial L}{\partial g(\mathbf{x})} &= \frac{\partial L}{\partial f(\mathbf{x})} \times \frac{\partial f(\mathbf{x})}{\partial g(\mathbf{x})} \\
  &= \left(- \frac{y}{p} + \frac{1-y}{1-p}\right) \times p(1-p) \\
  &= p-y
\end{align*}
$$

### Parameter Optimization

To minimize the loss value $L$, we need to adjust the weights $\mathbf{w}$ and bias $b$ of the function $f(\mathbf{x})$.

From

$$
g(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + b
$$

We can get:

$$
\frac{\partial g(\mathbf{x})}{\partial \mathbf{w}} = \mathbf{x} \\
$$

$$
\frac{\partial g(\mathbf{x})}{\partial b} = 1
$$

Therefore:

$$
\frac{\partial L}{\partial \mathbf{w}} = \frac{\partial L}{\partial g(\mathbf{x})} \times \frac{\partial g(\mathbf{x})}{\partial \mathbf{w}} = \mathbf{x} (p-y)
$$

$$
\frac{\partial L}{\partial b} = \frac{\partial L}{\partial g(\mathbf{x})} \times \frac{\partial g(\mathbf{x})}{\partial b} = p - y
$$

Thus, during parameter optimization, we can proceed as follows:

$$
\mathbf{w}_{n+1} = \mathbf{w}_{n} - \gamma \frac{\partial L}{\partial \mathbf{w}} = \mathbf{w}_{n} - \gamma \mathbf{x} (p - y)
$$

$$
b_{n+1} = b_{n} - \gamma \frac{\partial L}{\partial b} = b_{n} - \gamma (p - y)
$$

This concludes the basic principles and optimization methods of perceptrons. For detailed implementation, please refer to the `main.py` code.

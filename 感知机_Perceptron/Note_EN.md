## The basical knowledge of perceptron

**Perceptron is a linear classifier created by Frank Rosenblatt**
  - Linear classifiers also include the following:
    - Logistic Regression
    - Support vector machine
    - Fisher's Linear Discriminant

**The mathematical definition of perceptron**:

$$
f(\mathbf{x}) = \text{sign}(\mathbf{w}^T \mathbf{x} + b)
$$

$T$ is the transpose for weight vector， $\text{sign}(\mathbf{x})$ is the signum function, the definition is:

$$
\text{sign}(x) = \begin{cases}
-1, & x < 0 \\
0, & x = 0 \\
1, & x > 0
\end{cases}
$$

### The derivation of perceptron

Now, we define a function:

$$
g(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + b
$$

So, we can get this:

$$
f(\mathbf{x}) = \text{sign}(g(\mathbf{x}))
$$

From $g(\mathbf{x})$ , we can see the output of it is processed with weighted summation and translated with bias $b$ .

From $\text{sign}(x)$ , it acts as：

$$
\text{sign}(x) = \begin{cases}
    \frac{x}{|x|}, x \not ={0} \\
    0, x = 0
\end{cases}
$$

So, the function $f(\mathbf{x})$ is getting symbol of $g(\mathbf{x})$ .

### The differentiability of perceptron

The differentiability is the key feature for modern perceptron and neural network, it means them can be optimized with gradient descent, which will be described particularly.

In our example, $f(\mathbf{x})$ is a composite function, its differential is following：

$$
\frac{\mathrm{d} f(\mathbf{x})}{\mathrm{d} \mathbf{x}} = \frac{\mathrm{d} \text{sign}(g(\mathbf{x}))}{\mathrm{d} g(\mathbf{x})} \times \frac{\mathrm{d} g(\mathbf{x})}{\mathrm{d} \mathbf{x}}
$$

There is shown with ***Chain rules***. The differential of $\text{sign}(x)$ is following：

$$
\frac{\mathrm{d} \text{sign}(x)}{\mathrm{d} x} = \begin{cases}
  0, & x \not ={0} \\
  Can't be differentiated, x = 0
\end{cases}
$$

So, the differential of $f(\mathbf{x})$ is following：

$$
\frac{\mathrm{d} f(\mathbf{x})}{\mathrm{d} \mathbf{x}} = \begin{cases}
  0, & x \not ={0} \\
  Can't be differentiated, x = 0
\end{cases}
$$

This case will lead to $f(\mathbf{x})$ can't be optimized effectively.

So, we usually use $\text{sigmoid}(x)$ as activation function, rather than $\text{sign}(x)$, the output of it is $y \in (0, 1)$, here is definition:

$$
\text{sigmoid}(x) = \frac{1}{1 + e^{-x}}
$$

The differential is:

$$
\frac{\mathrm{d} \text{sigmoid}(x)}{\mathrm{d} x} = \text{sigmoid}(x)(1-\text{sigmoid}(x)) = \frac{e^{-x}}{(1 + e^{x})^{2}}
$$

With the change, the $f(\mathbf{x})$ is differentiable. Furtherly, the function can be optimized.

---

## Gradient descent

**Gradient descent is a method of functional minimization proposed by Augustin-Louis Cauchy.**

### The derivation of gradient desceng.

We define a function $f(\mathbf{x}$, which is differentiable, but unknown. And the $n$ is times of optimizaatio:

$$
f(x_{n+1}) = f(x_{n}) + (x_{n+1} - x_{n}) \frac{f(x_{n+1}) - f(x_{n})}{x_{n+1} - x_{n}}
$$

要使得 $f(x)$ 最小化，我们就需要寻找到能使得函数最小的 $x$ ,观察上式可知，我们需要后面的

$$
(x_{n+1} - x_{n}) \frac{f(x_{n+1}) - f(x_{n})}{x_{n+1} - x_{n}} < 0
$$

简化，并假设更新系数（在后面作为学习率） $\gamma$ ：

$$
(x_{n+1} - x_{n}) f'(x_{n}) < 0 \\
x_{n+1} - x_{n} = - \gamma f'(x_{n}) \\
x_{n+1} = x_{n} - \gamma f'(x_{n})
$$

综上，我们需要找到 $x$ 使得 $f(x)$ 最小就需要知道当前的 $x$ 和对应的 $f'(x)$ 

---

## 三、实际操作

根据上面的两个部分，我们已经对感知机和梯度下降法有了初步的认识，接下来就是将他们结合起来

现代感知机：

$$
g(x) = \mathbf{w}^T \mathbf{x} + b \\
f(\mathbf{x}) = \text{sigmoid}(g(\mathbf{x}))
$$

### 损失函数

在机器学习的优化中通常使用“损失函数”来表示模型对实际数据的拟合差距，损失越小意味着拟合效果越好。常见的可以作为损失函数的有“均方误差”和“交叉熵”，前者用于回归模型（实际值），后者用于预测模型（概率）。

在常见的二分类任务中，我们使用交叉熵来计算预测概率和实际概率的差距。

交叉熵：

$$
H(P, Q) = -\sum_{i} P_{i} \log Q_{i}
$$

基于交叉熵的损失函数：

$$
L = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log(p_{i,c})
$$

其中 $N$ 是样本数量， $C$ 是分类类别

而在二分类中，两个类别概率之和为1，因此假设

$$
P_1 = 0.6
$$

则

$$
P_2 = 1 - 0.6 =0.4
$$

因此二分类的交叉熵损失函数可以写作：

$$
L = - y \log p + (1-y)\log(1-p)
$$

由于其中的 $y$ 是真实的样本（one-hot编码样本，即对应的标签的概率为1），但也是一个可变变量，因此，我们对损失函数 $L$ 关于变量 $p$ 的导数需要通过偏微分获得，即：

$$
\frac{\partial L}{\partial p} = - \frac{y}{p} + \frac{1-y}{1-p}
$$

由于感知机的输出是 $f(\mathbf{x}) \in (0,1)$ 因此可以将其视为概率 $p$ 来进行使用。

所以

$$
\begin{align*}
  \frac{\partial L}{\partial g(\mathbf{x})} &= \frac{\partial L}{\partial f(\mathbf{x})} \times \frac{\partial f(\mathbf{x})}{\partial g(\mathbf{x})} \\
  &= (- \frac{y}{p} + \frac{1-y}{1-p}) \times p(1-p) \\
  &= p-y
\end{align*}
$$

### 参数优化

要使得损失值 $L$ 最小，需要调节函数 $f(\mathbf{x})$ 的权重 $\mathbf{w}$ 和 $b$ 。

根据 

$$
g(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + b
$$

可知：

$$
\frac{\partial g(\mathbf{x})}{\partial \mathbf{w}} = \mathbf{x} \\
$$

$$
\frac{\partial g(\mathbf{x})}{\partial b} = 1
$$

因此：

$$
\frac{\partial L}{\partial \mathbf{w}} = \frac{\partial L}{\partial g(\mathbf{x})} \times \frac{\partial g(\mathbf{x})}{\partial \mathbf{w}} = \mathbf{x} \mathbf{(p-y)}
$$

$$
\frac{\partial L}{\partial b} = \frac{\partial L}{\partial g(\mathbf{x})} \times \frac{\partial g(\mathbf{x})}{\partial b} = \mathbf{p} - \mathbf{y}
$$

所以在参数优化时我们可以按照以下方式进行：

$$
\mathbf{w}_{n+1} = \mathbf{w}_{n} - \gamma \frac{\partial L}{\partial \mathbf{w}} = \mathbf{w}_{n} - \gamma \mathbf{x} (\mathbf{p} - \mathbf{y})
$$

$$
b_{n+1} = b_{n} - \gamma \frac{\partial L}{\partial b} = b_{n} - \gamma (\mathbf{p} - \mathbf{y})
$$

至此感知机的基本原理和优化方式到此介绍，详细的实现过程可以参考  `main.py` 代码。
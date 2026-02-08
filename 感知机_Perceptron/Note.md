### 感知机的基本知识

**感知机是线性分类器中的一种**
  - 线性分类器还包括以下类别
    - 逻辑回归
    - 支持向量机
    - Fisher线性判别

**感知机的数学表达**：

$$
f(\mathbf{x}) = \text{sign}(\mathbf{w}^T \mathbf{x} + b)
$$

其中**T**代表是权重向量的转置，$\text{sign}(\mathbf{x})$是阶越函数，它的表达式是：

$$
\text{sign}(x) = \begin{cases}
-1, & x < 0 \\
0, & x = 0 \\
1, & x > 0
\end{cases}
$$

### 感知机的数学推导

设
$$
g(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + b
$$

则

$$
f(\mathbf{x}) = \text{sign}(g(\mathbf{x}))
$$

从$g(\mathbf{x})$来看，是一个将输入进行加权求和后，再由偏置$b$进行平移后得到输出。

而从$\text{sign}(\mathbf{x})$来看，则可以视为：

$$
\text{sign}(x) = \begin{cases}
    \frac{x}{|x|}, x \not ={0} \\
    0, x = 0
\end{cases}
$$

因此$f(\mathbf{x})$可以看作是只取得$g(\mathbf{x})$的符号位。

### 感知机的可微分性

可微分是感知机和神经网络的一个重要特征，这意味着它可以使用类似于梯度下降之类的方式优化。（后面会详细介绍）

$f(\mathbf{x})$是一个复合函数，它的微分可以如下表示：

$$
\frac{\delta f(\mathbf{x})}{\delta \mathbf{x}} = \frac{\delta \text{sign}(g(\mathbf{x}))}{\delta g(\mathbf{x})} \times \frac{\delta g(\mathbf{x})}{\delta \mathbf{x}}
$$

其中$\text{sign}(x)$的导数如下：

$$
\frac{\delta \text{sign}(x)}{\delta x} = \begin{cases}
  0, & x \not ={0} \\
  不可求导, x = 0
\end{cases}
$$

因此$f(\mathbf{x})$的导数为：

$$
\frac{\delta f(\mathbf{x})}{\delta \mathbf{x}} = \begin{cases}
  0, & x \not ={0} \\
  不可求导, x = 0
\end{cases}
$$

这将导致在对$f(\mathbf{x})$进行优化时无法获取有效的梯度。

针对这种情况便有了使用tanh（双曲正切）作为激活函数的解决方法，它是值域在$y \in (-1, 1)$的函数，其表达式为：

$$
\text{tanh}(x) = \frac{e^{x}-e^{-x}}{e^{x}+e^{-x}}
$$

导数为：

$$
\frac{\delta \text{tanh}}{}
$$
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
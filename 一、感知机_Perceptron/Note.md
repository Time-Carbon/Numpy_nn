## 一、感知机的基本知识

**感知机是由弗兰克·罗森布拉特（Frank Rosenblatt）提出的一种线性分类器**
  - 线性分类器还包括以下类别
    - 逻辑回归
    - 支持向量机
    - Fisher线性判别

**初代感知机的数学定义**：

$$
f(\mathbf{x}) = \text{sign}(\mathbf{w}^T \mathbf{x} + b)
$$

其中 $T$ 代表是权重向量的转置， $\text{sign}(x)$ 是阶越函数，它的表达式是：

$$
\text{sign}(x) = \begin{cases}
-1, & x < 0 \\
0, & x = 0 \\
1, & x > 0
\end{cases}
$$

### 初代感知机的数学推导

设

$$
g(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + b
$$

则

$$
f(\mathbf{x}) = \text{sign}(g(\mathbf{x}))
$$

从 $g(\mathbf{x})$ 来看，是一个将输入进行加权求和后，再由偏置$b$进行平移后得到输出。

而从 $\text{sign}(\mathbf{x})$ 来看，则可以视为：

$$
\text{sign}(x) = \begin{cases}
    \frac{x}{|x|}, x \not ={0} \\
    0, x = 0
\end{cases}
$$

因此 $f(\mathbf{x})$ 可以看作是只取得 $g(\mathbf{x})$ 的符号。

### 感知机的可微分性

可微分是现代感知机和神经网络的一个重要特征，这意味着它可以使用类似于梯度下降之类的方式优化。（后面会详细介绍）

在我们的原始感知机的例子中 $f(\mathbf{x})$ 是一个复合函数，它的微分情况可以如下表示：

$$
\frac{\mathrm{d} f(\mathbf{x})}{\mathrm{d} \mathbf{x}} = \frac{\mathrm{d} \text{sign}(g(\mathbf{x}))}{\mathrm{d} g(\mathbf{x})} \times \frac{\mathrm{d} g(\mathbf{x})}{\mathrm{d} \mathbf{x}}
$$

这里使用 ***“链式法则”*** 的形式展示的，其中 $\text{sign}(x)$ 的导数如下：

$$
\frac{\mathrm{d} \text{sign}(x)}{\mathrm{d} x} = \begin{cases}
  0, & x \not ={0} \\
  不可求导, x = 0
\end{cases}
$$

因此 $f(\mathbf{x})$ 的导数为：

$$
\frac{\mathrm{d} f(\mathbf{x})}{\mathrm{d} \mathbf{x}} = \begin{cases}
  0, & x \not ={0} \\
  不可求导, x = 0
\end{cases}
$$

这将导致在对 $f(\mathbf{x})$ 进行优化时无法获取有效的导数（也就是“梯度下降”中的“梯度”）。

针对这种情况便有了使用 $\text{sigmoid}(x)$ （s型生长曲线）取代 $\text{sign}(x)$ 作为激活函数的解决方法，它是值域在 $y \in (0, 1)$ 的函数，其表达式为：

$$
\text{sigmoid}(x) = \frac{1}{1 + e^{-x}}
$$

导数为：

$$
\frac{\mathrm{d} \text{sigmoid}(x)}{\mathrm{d} x} = \text{sigmoid}(x)[1-\text{sigmoid}(x)] = \frac{e^{-x}}{(1 + e^{x})^{2}}
$$

这样子有了可导且非0的激活函数后就可以进行对 $f(\mathbf{x})$ 的优化了

---

## 二、梯度下降优化

**梯度下降是由奥古斯丁-路易·柯西‌（Augustin-Louis Cauchy）提出的一种函数最小化优化算法**

### 梯度下降的数学推导

设可微分的未知函数 $f(x)$ ， $n$ 为优化次数：

$$
f(x_{n+1}) = f(x_{n}) + (x_{n+1} - x_{n}) \frac{f(x_{n+1}) - f(x_{n})}{x_{n+1} - x_{n}}
$$

要使得 $f(x)$ 最小化，我们就需要寻找到能使得函数最小的 $x$ ,观察上式可知，我们需要使得下面的式子为负：

$$
(x_{n+1} - x_{n}) \frac{f(x_{n+1}) - f(x_{n})}{x_{n+1} - x_{n}} < 0
$$

简化，并假设更新系数（在后面作为学习率） $\gamma$ ：

$$
(x_{n+1} - x_{n}) f'(x_{n}) < 0 \\
x_{n+1} - x_{n} = - \gamma f'(x_{n}) \\
x_{n+1} = x_{n} - \gamma f'(x_{n})
$$

综上，我们需要找到下一步的 $x_{n+1}$ 使得 $f(x_{n+1})$ 最小就需要知道当前的 $x_{n}$ 和对应的 $f'(x_{n})$ 

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
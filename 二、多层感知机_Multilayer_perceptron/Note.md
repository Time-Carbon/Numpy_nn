## 一、多层感知机的由来

想要了解为什么会有多层感知机，我们需要知道单个感知机有哪些限制。

首先，根据上一节

## 多层感知机的优化
此时我们将面临一个问题：

$$
\boxed{\text{概率P该如何获取}}
$$

由于 $f(\mathbf{x})$ 的值域是 $f(\mathbf{x}) \in (-1, 1)$ ，不匹配概率的特性，即：

- $p \in [0,1]$
- $\sum _{i=1} p_{i} = 1$

因此我们常引入 $\text{softmax}(x)$ 将 $f(\mathbf{x})$ 输出的值转换成预测概率。它的数学表达是：

$$
\text{softmax}(x_{i}) = \frac{e^{x_{i}}}{\sum _{j=q} ^{C} e^{x_{j}}}
$$

这个函数有两个实用的性质：

- $\text{softmax}(x) \in (0, 1)$，且总和为1
- 在定义域内单调递增

首先，第一个性质满足了概率的特征。其次，第二个性质使得它保留了原始输入的相对大小，而不会像二次函数之类的函数把忽略正负值的差异。

因此我们可以通过 $\text{softmax}(x)$ 将 $f(\mathbf{x})$ 的输出和损失值 $L$ 进行关联：

$$
\frac{\partial L}{\partial f(\mathbf{x})} = \frac{\partial L}{\partial p} \times \frac{\partial p}{\partial f(\mathbf{x})}
$$

其中， $p$ 为：

$$
p = \text{softmax}(f(\mathbf{x}))
$$

由于 $\text{softmax}(x)$ 的求导过于复杂，以至于需要较大篇幅来介绍，因此我们跳过这部分，直接记住最终的偏导即可：

$$
\frac{\partial L}{\partial f(\mathbf{x})} = \mathbf{p} - \mathbf{y}
$$

此外， $\text{sigmoid}(x)$ 函数和二分类的 $\text{softmax}(x)$ 等价。并且不影响最终的偏导结果。

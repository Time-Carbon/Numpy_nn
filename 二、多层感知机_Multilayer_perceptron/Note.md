## 一、多层感知机的由来

想要了解为什么会有多层感知机，我们需要知道单个感知机有哪些劣势。

首先，我们观察一下感知机的数学定义：

$$
\begin{align*}
    & g(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + b \\
    & f(\mathbf{x}) = \text{sigmoid}(g(\mathbf{x}))
\end{align*}
$$

我们可以看到 $g(\mathbf{x})$ 是一个线性函数，可以将它展开表示为：

$$
\begin{align*}
    g(\mathbf{x}) & = \mathbf{w}^T \mathbf{x} + b \\
    & = w_1 x_1 + w_2 x_2 + ... + w_n x_n + b
\end{align*}
$$

当权重和输入的数量都为1时就是一个一元一次函数，即：

$$
g(x) = wx + b
$$

它的图像是一条直线：

$$
y = 2x + 0
$$

<div align="center">
<img src="https://foruda.gitee.com/images/1771509713368365883/ef69917b_16636277.png" width="50%"/>
</div>

数量增至2时，则是在三维空间的一个平面：

$$
z = 2x + 2y + 0
$$

<div align="center">
<img src="https://foruda.gitee.com/images/1771509768357237564/7d1601a6_16636277.png" width="50%"/>
</div>

由此我们可以发现，无论函数 $g(\mathbf{x})$ 的维度提升至多少维度，它都是将一个空间平直地划分为两个空间，不会出现弯曲。

它可以有效地解决部分问题，也就是有明确界限将数据划分开的，比如感知机例子中的判断温度是否大于某个值。

但是对于没有明确界限的非线性问题将无能为力，比如异或问题（两个数据相同输出0，不同输出1）

我们假设有一组向量的表示如下：

$$
(输入a, 输入b, 异或结果)
$$

其中输入a和输入b的取值只有0和1，则a和b的异或情况可以表示如下：

$$
\begin{align*}
    & (0,1,1) \\
    & (0,0,0) \\
    & (1,1,0) \\
    & (1,0,1)
\end{align*}
$$

在三维空间中，这些点所构成的物体是一个三角锥：

<div align="center">
<img src="https://foruda.gitee.com/images/1771509638532310641/f1db1d86_16636277.png" width="50%"/>
</div>

这意味着平面已经没法正确拟合这些点了，


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

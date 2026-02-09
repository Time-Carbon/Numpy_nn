交叉熵：

$$
H(P, Q) = -\sum_{i} P(i) \log Q(i)
$$

基于交叉熵的损失函数：

$$
L = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log(p_{i,c})
$$

其中 $N$ 是样本数量， $C$ 是分类类别

由于其中的 $y$ 是真实的样本（one-hot编码样本，即对应的标签的概率为1），但也是一个可变变量，因此，我们对损失函数 $L$ 关于变量 $p$ 的导数需要通过偏微分获得，即：

$$
\frac{\partial L}{\partial p_{i,c}} = - \frac{1}{N} \times \frac{y_{i,c}}{p_{i,c}}
$$

而现在我们将面临一个问题，预测概率 $p$ 应该如何获取？因为 $f(\mathbf{x})$ 的输出是一组实数，而不是值域 $f(\mathbf{x}) \in [0,1]$ 且总和为1的数。

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
\frac{\partial L}{\partial f(\mathbf{x})} = p - y
$$

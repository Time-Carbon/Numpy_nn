import numpy as np


class MLP:
    def __init__(self, layers_shape, dtype):
        self.layers_shape = layers_shape
        self.layers_num = len(layers_shape)
        self.dtype = dtype

        self.weight = []
        self.biase = []
        self.z_cache = []
        self.d_Lrelu_cache = []

        for i in range(self.layers_num - 1):  # 创建各层的初始权重和偏置
            w = np.random.randn(self.layers_shape[i], self.layers_shape[i + 1]).astype(
                self.dtype) * np.sqrt(2 / self.layers_shape[i])  # He 初始化方法
            b = np.zeros((self.layers_shape[i + 1])).astype(self.dtype)
            self.weight.append(w)
            self.biase.append(b)

    def softmax(self, z):
        z -= np.max(z, axis=1, keepdims=True).astype(self.dtype)
        z_exp = np.exp(z).astype(self.dtype)
        z_sum = np.sum(z_exp, axis=1, keepdims=True).astype(self.dtype)
        return z_exp / z_sum

    def leaky_relu(self, z, alpha):
        self.d_Lrelu_cache.append(
            np.where(z > 0, 1, alpha).astype(self.dtype))  # 添加relu的梯度缓存
        z = np.maximum(alpha * z, z).astype(self.dtype)
        return z

    def cross_entropy(self, p, q):
        ln_q = np.log(q + 1e-8).astype(self.dtype)  # 防止过小超出浮点数表示范围
        ce = p * ln_q
        return np.mean(- np.sum(ce, axis=1, keepdims=True)).astype(self.dtype)

    def forward(self, x, alpha=0.01):
        z = x
        for i in range(self.layers_num - 1):
            z = np.dot(z, self.weight[i]).astype(self.dtype) + self.biase[i]

            if i == self.layers_num - 2:
                z = self.softmax(z)
            else:
                z = self.leaky_relu(z, alpha)

            self.z_cache.append(z)
        return self.z_cache[-1]

    def backward(self, x, y, lr):
        d_err = (self.z_cache[-1] - y) / x.shape[0]  # 经过交叉熵和softmax求导后的偏导
        for i in range(self.layers_num - 2, -1, -1):  # 倒序 从 layer_num 到 0
            if i == 0:
                d_err = self.d_Lrelu_cache[i] * d_err
                self.weight[i] -= lr * np.dot(x.T, d_err).astype(self.dtype)
                self.biase[i] -= lr * np.mean(d_err, axis=0).astype(self.dtype)
            elif i == self.layers_num - 2:
                self.weight[i] -= lr * \
                    np.dot(self.z_cache[i-1].T, d_err).astype(self.dtype)
                self.biase[i] -= lr * np.mean(d_err, axis=0).astype(self.dtype)
                d_err = np.dot(d_err, self.weight[i].T).astype(self.dtype)
            else:
                d_err = self.d_Lrelu_cache[i] * d_err
                self.weight[i] -= lr * \
                    np.dot(self.z_cache[i-1].T, d_err).astype(self.dtype)
                self.biase[i] -= lr * np.mean(d_err, axis=0).astype(self.dtype)
                d_err = np.dot(d_err, self.weight[i].T).astype(self.dtype)

    def train(self, x, y, step, note_step=1, lr=0.01, alpha=0.01):
        x_input = x  # 存储原始输入，后续将其打乱
        y_input = y

        for i in range(step):
            index = np.random.permutation(x.shape[0])  # 生成随机的不重复的序列
            x_input = x_input[index]
            y_input = y_input[index]

            self.forward(x, alpha)
            loss = self.cross_entropy(y, self.z_cache[-1])
            if i % note_step == 0:
                print(loss)
            self.backward(x, y, lr)
            self.z_cache.clear()
            self.d_Lrelu_cache.clear()


if __name__ == "__main__":
    dataType = np.float16

    x_0 = np.arange(0, 10).astype(
        dataType, copy=False).reshape(-1, 1)  # 构建异或为0的数据
    x_0 = x_0 + np.zeros((x_0.shape[0], 2), dtype=dataType)
    x_0 = x_0 / np.max(x_0)  # 输入数据归一化
    y_0 = np.zeros((x_0.shape[0]), dtype=np.int8)  # 对应数量的标签
    y_0 = np.eye(2)[y_0].astype(dataType, copy=False)
    # 转换为one-hot标签，即有两种不同的标签[输出为0，输出为1]，被选中的标签对应的值为1
    # 比如one-hot标签为[1,0]，则表示选中的标签是第一个标签，即“输出为0”

    # 构建异或为1的数据(由于有去重和筛选环节，因此需要留出多余的数来确保经过这些操作后数据能到达10)
    x_1 = np.random.randint(0, 10, size=(11, 2))
    mask = x_1[:, 0] != x_1[:, 1]
    x_1 = x_1[mask]  # 取两个数不同的数组
    x_1 = np.unique(x_1, axis=0).astype(dataType, copy=False)
    x_1 = x_1 / np.max(x_1)
    y_1 = 1 + np.zeros((x_1.shape[0]), dtype=np.int8)
    y_1 = np.eye(2)[y_1].astype(dataType, copy=False)

    x = np.concatenate((x_1, x_0), axis=0)  # 沿着行的方向合并数据
    y = np.concatenate((y_1, y_0), axis=0)

    input_dim = x.shape[1]  # 输入层维度，即输入数据有多少个
    hide_dim = 4 * input_dim  # 隐藏层维度，需要扩大维度，用于抵消激活函数导致的信息损失，详见Note.md
    output_dim = y.shape[1]  # 输出层维度，即输出时有多少个类别

    mlp = MLP([input_dim, hide_dim, hide_dim, output_dim], dtype=dataType)

    mlp.train(x, y, 2000, 200, 1e-2)

    x_test = np.array([
        [2, 3],
        [2, 2],
        [4, 5],
        [5, 5]
    ]).astype(dataType, copy=False)
    x_test = x_test / np.max(x_test)

    y_test = np.array([
        [0, 1],
        [1, 0],
        [0, 1],
        [1, 0]
    ]).astype(dataType, copy=False)

    print(y_test)  # 输出真实标签，方便我们后续对比
    print()

    pred_p = mlp.forward(x_test)  # 获取训练好的多层感知机的输出
    pred_label = np.argmax(pred_p, axis=1)
    pred_label = np.eye(2)[pred_label]  # 转换为one-hot标签

    print(pred_label)
    print()
    print(pred_p)

import numpy as np
import sys

class MLP:
    def __init__(self, layers_shape, dtype):
        self.layers_shape = layers_shape
        self.layers_num = len(layers_shape)
        self.dtype = dtype

        self.weight = []
        self.biase = []
        self.z_cache = []
        self.d_Lrelu_cache = []

        for i in range(self.layers_num - 1):
            w = (np.random.randn(self.layers_shape[i], self.layers_shape[i + 1]) * np.sqrt(
                2 / self.layers_shape[i])).astype(self.dtype)  # He 初始化方法
            b = np.zeros(self.layers_shape[i + 1]).astype(self.dtype)
            self.weight.append(w)
            self.biase.append(b)

    def softmax(self, z):
        z -= np.max(z, axis=1, keepdims=True)
        z_exp = np.exp(z)
        z_sum = np.sum(z_exp, axis=1, keepdims=True)
        return z_exp / z_sum

    def leaky_relu(self, z, alpha=0.01):
        d_z = np.where(z > 0, 1, alpha).astype(self.dtype)
        self.d_Lrelu_cache.append(d_z)
        z = z * d_z
        return z

    def cross_entropy(self, p, q):
        ln_q = np.log(q + np.finfo(self.dtype).tiny)  # 防止过小超出浮点数表示范围
        ce = p * ln_q
        ce_sum = np.sum(ce, axis=1, keepdims=True)
        loss = np.mean(-ce_sum)
        return loss

    def forward(self, x, alpha=0.01):
        z = x
        for i in range(self.layers_num - 1):
            z = np.dot(z, self.weight[i]) + self.biase[i]

            if i == self.layers_num - 2:
                z = self.softmax(z)
            else:
                z = self.leaky_relu(z, alpha)

            self.z_cache.append(z)
        return self.z_cache[-1]

    def backward(self, x, y, lr):
        d_err = (self.z_cache[-1] - y)  # 经过交叉熵和softmax求导后的偏导
        for i in range(self.layers_num - 2, -1, -1):  # 倒序 从 layer_num 到 0
            d_w = 0
            d_b = 0

            if i == 0:
                d_err = self.d_Lrelu_cache[i] * d_err
                d_w = np.dot(x.T, d_err)
                d_b = np.mean(d_err, axis=0)
            elif i == self.layers_num - 2:
                d_err_next = np.dot(d_err, self.weight[i].T)
                d_w = np.dot(self.z_cache[i-1].T, d_err)
                d_b = np.mean(d_err, axis=0)
                d_err = d_err_next
            else:
                d_err = self.d_Lrelu_cache[i] * d_err
                d_err_next = np.dot(d_err, self.weight[i].T)
                d_w = np.dot(self.z_cache[i-1].T, d_err)
                d_b = np.mean(d_err, axis=0)
                d_err = d_err_next

            self.weight[i] -= lr * d_w
            self.biase[i] -= lr * d_b

            return 0

    def train(self, x, y, step=sys.maxsize, note_step=1, lr=0.01, alpha=0.01, epoche=1, batch=4):
        x_input = x  # 存储原始输入，后续将其打乱
        y_input = y
        step_count = 0

        for i_e in range(epoche):
            index = np.random.permutation(x.shape[0])  # 生成随机的不重复的序列
            x_input = x_input[index]
            y_input = y_input[index]

            for i in range(0, np.minimum(x_input.shape[0], step), np.minimum(x_input.shape[0], batch)):
                self.forward(x_input[i:i+batch-1], alpha)
                loss = self.cross_entropy(
                    y_input[i:i+batch-1], self.z_cache[-1])
                if step_count % note_step == 0:
                    print(loss)
                self.backward(x_input[i:i+batch-1], y_input[i:i+batch-1], lr)
                self.z_cache.clear()
                self.d_Lrelu_cache.clear()
                step_count += 1

        return 0

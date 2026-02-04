import numpy as np

class BidirectionalMLP:
    def __init__(self, layers_shape, dtype):
        self.layers_shape = layers_shape
        self.layers_num = len(layers_shape)
        self.dtype = dtype

        self.weight = []
        self.biase = []
        
        # 新增：反向映射的权重和偏置
        self.reverse_weight = []
        self.reverse_biase = []
        
        self.z_cache = []
        self.d_Lrelu_cache = []
        
        # 新增：反向传播时的缓存
        self.reverse_z_cache = []
        self.reverse_d_Lrelu_cache = []

        # 初始化前向映射的权重和偏置
        for i in range(self.layers_num - 1):
            w = np.random.randn(self.layers_shape[i], self.layers_shape[i + 1]).astype(self.dtype) * np.sqrt(2 / self.layers_shape[i])
            b = np.zeros(self.layers_shape[i + 1]).astype(self.dtype)
            self.weight.append(w)
            self.biase.append(b)

        # 初始化反向映射的权重和偏置
        # 反向映射的结构应该是输出层到输入层，隐藏层结构与前向传播对应
        reverse_layers_shape = list(reversed(layers_shape))
        for i in range(len(reverse_layers_shape) - 1):
            rw = np.random.randn(reverse_layers_shape[i], reverse_layers_shape[i + 1]).astype(self.dtype) * np.sqrt(2 / reverse_layers_shape[i])
            rb = np.zeros(reverse_layers_shape[i + 1]).astype(self.dtype)
            self.reverse_weight.append(rw)
            self.reverse_biase.append(rb)

    def softmax(self, z):
        z -= np.max(z, axis=1, keepdims=True)
        z_exp = np.exp(z)
        z_sum = np.sum(z_exp, axis=1, keepdims=True)
        return z_exp / z_sum

    def leaky_relu(self, z, alpha=0.01):
        self.d_Lrelu_cache.append(np.where(z > 0, 1, alpha).astype(self.dtype))
        z = np.maximum(alpha * z, z)
        return z

    def reverse_leaky_relu(self, z, alpha=0.01):
        self.reverse_d_Lrelu_cache.append(np.where(z > 0, 1, alpha).astype(self.dtype))
        z = np.maximum(alpha * z, z)
        return z

    def cross_entropy(self, p, q):
        ln_q = np.log(q + 1e-8)
        ce = p * ln_q
        return np.mean(-np.sum(ce, axis=1, keepdims=True))

    def forward(self, x, alpha=0.01):
        # 前向传播：x -> y
        z = x
        self.z_cache.clear()
        self.d_Lrelu_cache.clear()
        self.z_cache.append(x)  # 保存输入
        
        for i in range(self.layers_num - 1):
            z = np.dot(z, self.weight[i]) + self.biase[i]
            
            if i == self.layers_num - 2:
                z = self.softmax(z)
            else:
                z = self.leaky_relu(z, alpha)
            
            self.z_cache.append(z)
        
        # 反向传播：y -> x_pred
        self.reverse_z_cache.clear()
        self.reverse_d_Lrelu_cache.clear()
        
        reverse_z = self.z_cache[-1]  # 从输出开始
        self.reverse_z_cache.append(reverse_z)
        
        for i in range(len(self.reverse_weight)):
            reverse_z = np.dot(reverse_z, self.reverse_weight[i]) + self.reverse_biase[i]
            
            # 最后一层不需要激活函数，以确保输出维度正确
            if i < len(self.reverse_weight) - 1:
                reverse_z = self.reverse_leaky_relu(reverse_z, alpha)
            
            self.reverse_z_cache.append(reverse_z)
        
        return self.z_cache[-1], reverse_z  # 返回预测输出和重构输入

    def backward(self, x, y, lr):
        # 前向映射的梯度（y方向）
        d_err_forward = (self.z_cache[-1] - y) / x.shape[0]
        
        # 反向映射的梯度（x方向）
        d_err_reverse = (self.reverse_z_cache[-1] - x) / x.shape[0]
        
        # 更新反向映射的权重和偏置（从最后一层到第一层）
        for i in range(len(self.reverse_weight) - 1, -1, -1):
            if i == len(self.reverse_weight) - 1:
                # 最后一层（连接到输入层），不需要激活函数的导数
                self.reverse_weight[i] -= lr * np.dot(self.reverse_z_cache[i].T, d_err_reverse)
                self.reverse_biase[i] -= lr * np.mean(d_err_reverse, axis=0)
                d_err_reverse = np.dot(d_err_reverse, self.reverse_weight[i].T)
            else:
                # 中间层，需要考虑激活函数的导数
                d_err_reverse = self.reverse_d_Lrelu_cache[i] * d_err_reverse
                self.reverse_weight[i] -= lr * np.dot(self.reverse_z_cache[i].T, d_err_reverse)
                self.reverse_biase[i] -= lr * np.mean(d_err_reverse, axis=0)
                if i > 0:  # 不是第一层才继续反向传播
                    d_err_reverse = np.dot(d_err_reverse, self.reverse_weight[i].T)
        
        # 更新前向映射的权重和偏置
        for i in range(self.layers_num - 2, -1, -1):
            if i == 0:
                d_err_forward = self.d_Lrelu_cache[i] * d_err_forward
                self.weight[i] -= lr * np.dot(x.T, d_err_forward)
                self.biase[i] -= lr * np.mean(d_err_forward, axis=0)
            elif i == self.layers_num - 2:
                self.weight[i] -= lr * np.dot(self.z_cache[i].T, d_err_forward)
                self.biase[i] -= lr * np.mean(d_err_forward, axis=0)
                d_err_forward = np.dot(d_err_forward, self.weight[i].T)
            else:
                d_err_forward = self.d_Lrelu_cache[i] * d_err_forward
                self.weight[i] -= lr * np.dot(self.z_cache[i].T, d_err_forward)
                self.biase[i] -= lr * np.mean(d_err_forward, axis=0)
                d_err_forward = np.dot(d_err_forward, self.weight[i].T)

    def train(self, x, y, step, note_step=1, lr=0.01, alpha=0.01):
        for i in range(step):
            y_pred, x_recon = self.forward(x, alpha)
            loss_y = self.cross_entropy(y, y_pred)
            loss_x = np.mean((x_recon - x) ** 2)  # 使用均方误差作为重构损失
            total_loss = loss_y + loss_x
            
            if i % note_step == 0:
                print(f"Step {i}, Total Loss: {total_loss:.6f}, Y Loss: {loss_y:.6f}, X Loss: {loss_x:.6f}")
            
            self.backward(x, y, lr)

if __name__ == "__main__":
    dtype = np.float16
    x = np.array([[0,0],
                  [1,0],
                  [0,1],
                  [1,1]]).astype(dtype)
    y_label = np.array([1, 0, 0, 1])
    y = np.eye(2)[y_label].astype(dtype)

    batch = x.shape[0]
    input_dim = x.shape[1]
    hide_dim = 16
    type_num = y.shape[1]
    layer_num = 4

    layer_shape = []
    for i in range(layer_num):
        if i == 0:
            layer_shape.append(input_dim)
        elif i == layer_num - 1:
            layer_shape.append(type_num)
        else:
            layer_shape.append(hide_dim)

    mlp = BidirectionalMLP(layer_shape, dtype=dtype)

    mlp.train(x, y, 1000, 100, 1e-2)

    print("\nTrue labels:")
    print(y)
    print()

    pred_p, x_recon = mlp.forward(x)
    pred_label = np.argmax(pred_p, axis=1)
    pred_label = np.eye(2)[pred_label]
    print("Predicted labels:")
    print(pred_label)
    print()
    print("Predicted probabilities:")
    print(pred_p)
    print()
    print("Reconstructed inputs:")
    print(x_recon)
    print()
    print("Original inputs:")
    print(x)

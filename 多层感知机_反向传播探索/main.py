import numpy as np

class BidirectionalMLP:
    def __init__(self, layers_shape, dtype):
        self.layers_shape = layers_shape
        self.layers_num = len(layers_shape)
        self.dtype = dtype

        # 正向网络参数 (x -> y)
        self.forward_weight = []
        self.forward_biase = []
        
        # 反向网络参数 (y -> x)
        self.backward_weight = []
        self.backward_biase = []
        
        # 缓存
        self.z_cache = []
        self.d_Lrelu_cache = []
        self.backward_z_cache = []
        self.backward_d_Lrelu_cache = []

        # 初始化正向网络权重
        for i in range(self.layers_num - 1):
            w = np.random.randn(self.layers_shape[i], self.layers_shape[i + 1]).astype(self.dtype) * np.sqrt(2 / self.layers_shape[i])
            b = np.zeros(self.layers_shape[i + 1]).astype(self.dtype)
            self.forward_weight.append(w)
            self.forward_biase.append(b)

        # 初始化反向网络权重 (注意层数顺序相反)
        for i in range(self.layers_num - 1):
            # 反向网络的输入是输出层，输出是输入层
            w = np.random.randn(self.layers_shape[self.layers_num-1-i], self.layers_shape[self.layers_num-2-i]).astype(self.dtype) * np.sqrt(2 / self.layers_shape[self.layers_num-1-i])
            b = np.zeros(self.layers_shape[self.layers_num-2-i]).astype(self.dtype)
            self.backward_weight.append(w)
            self.backward_biase.append(b)

    def softmax(self, z):
        z -= np.max(z, axis=1, keepdims=True)
        z_exp = np.exp(z)
        z_sum = np.sum(z_exp, axis=1, keepdims=True)
        return z_exp / z_sum

    def leaky_relu(self, z, alpha=0.01):
        self.d_Lrelu_cache.append(np.where(z > 0, 1, alpha).astype(self.dtype))
        z = np.maximum(alpha * z, z)
        return z

    def backward_leaky_relu(self, z, alpha=0.01):
        self.backward_d_Lrelu_cache.append(np.where(z > 0, 1, alpha).astype(self.dtype))
        z = np.maximum(alpha * z, z)
        return z

    def cross_entropy(self, p, q):
        ln_q = np.log(q + 1e-8)
        ce = p * ln_q
        return np.mean(-np.sum(ce, axis=1, keepdims=True))

    def mse_loss(self, pred, true):
        return np.mean(np.square(pred - true))

    def forward(self, x, alpha=0.01):
        # 正向传播 x -> y
        z = x
        self.z_cache = [z]  # 保存每一层的输出，包括输入层
        self.d_Lrelu_cache = []
        
        for i in range(self.layers_num - 1):
            z = np.dot(z, self.forward_weight[i]) + self.forward_biase[i]
            
            if i == self.layers_num - 2:
                z = self.softmax(z)
            else:
                z = self.leaky_relu(z, alpha)
            
            self.z_cache.append(z)
        
        # 反向传播 y -> x (用于计算重构误差)
        y = self.z_cache[-1]  # 获取输出
        backward_z = y
        self.backward_z_cache = [backward_z]  # 保存反向网络每一层的输出
        self.backward_d_Lrelu_cache = []
        
        for i in range(self.layers_num - 1):
            backward_z = np.dot(backward_z, self.backward_weight[i]) + self.backward_biase[i]
            
            if i == self.layers_num - 2:
                # 最后一层不需要激活函数，直接输出
                pass
            else:
                backward_z = self.backward_leaky_relu(backward_z, alpha)
            
            self.backward_z_cache.append(backward_z)
        
        return self.z_cache[-1], self.backward_z_cache[-1]  # 返回预测输出和重构输入

    def backward(self, x, y, lr, reconstruction_weight=0.1):
        batch_size = x.shape[0]
        
        # 计算正向任务的梯度 (交叉熵损失)
        d_err_forward = (self.z_cache[-1] - y) / batch_size
        
        # 计算重构任务的梯度 (均方误差损失)
        d_err_reconstruct = 2 * (self.backward_z_cache[-1] - x) / batch_size
        
        # 更新反向网络参数 (y -> x)
        for i in range(self.layers_num - 2, -1, -1):
            if i == self.layers_num - 2:
                # 输出层到倒数第二层
                self.backward_weight[i] -= lr * np.dot(self.backward_z_cache[i].T, d_err_reconstruct)
                self.backward_biase[i] -= lr * np.mean(d_err_reconstruct, axis=0)
                d_err_reconstruct = np.dot(d_err_reconstruct, self.backward_weight[i].T)
            else:
                # 隐藏层
                d_err_reconstruct = self.backward_d_Lrelu_cache[i] * d_err_reconstruct
                self.backward_weight[i] -= lr * np.dot(self.backward_z_cache[i].T, d_err_reconstruct)
                self.backward_biase[i] -= lr * np.mean(d_err_reconstruct, axis=0)
                if i > 0:  # 不是第一层才继续反向传播
                    d_err_reconstruct = np.dot(d_err_reconstruct, self.backward_weight[i].T)
        
        # 更新正向网络参数 (x -> y)
        for i in range(self.layers_num - 2, -1, -1):
            if i == self.layers_num - 2:
                # 输出层
                self.forward_weight[i] -= lr * np.dot(self.z_cache[i].T, d_err_forward)
                self.forward_biase[i] -= lr * np.mean(d_err_forward, axis=0)
                d_err_forward = np.dot(d_err_forward, self.forward_weight[i].T)
            else:
                # 隐藏层
                d_err_forward = self.d_Lrelu_cache[i] * d_err_forward
                self.forward_weight[i] -= lr * np.dot(self.z_cache[i].T, d_err_forward)
                self.forward_biase[i] -= lr * np.mean(d_err_forward, axis=0)
                if i > 0:  # 不是第一层才继续反向传播
                    d_err_forward = np.dot(d_err_forward, self.forward_weight[i].T)

    def train(self, x, y, step, note_step=1, lr=0.01, alpha=0.01, reconstruction_weight=0.1):
        for i in range(step):
            pred_y, reconstruct_x = self.forward(x, alpha)
            task_loss = self.cross_entropy(y, pred_y)
            reconstruction_loss = self.mse_loss(reconstruct_x, x)
            total_loss = task_loss + reconstruction_weight * reconstruction_loss
            
            if i % note_step == 0:
                print(f"Step {i}: Total Loss: {total_loss:.6f}, Task Loss: {task_loss:.6f}, Recon Loss: {reconstruction_loss:.6f}")
            
            self.backward(x, y, lr, reconstruction_weight)
            
            # 清空缓存
            self.z_cache.clear()
            self.d_Lrelu_cache.clear()
            self.backward_z_cache.clear()
            self.backward_d_Lrelu_cache.clear()

if __name__ == "__main__":
    dtype = np.float64  # 使用float64提高精度
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
    layer_num = 4  # 减少层数以简化调试

    layer_shape = []
    for i in range(layer_num):
        if i == 0:
            layer_shape.append(input_dim)
        elif i == layer_num - 1:
            layer_shape.append(type_num)
        else:
            layer_shape.append(hide_dim)

    mlp = BidirectionalMLP(layer_shape, dtype=dtype)

    mlp.train(x, y, 1000, 100, 1e-2, reconstruction_weight=0.1)

    print("\nTrue labels:")
    print(y)
    print()

    pred_p, reconstruct_x = mlp.forward(x)
    pred_label = np.argmax(pred_p, axis=1)
    pred_label = np.eye(2)[pred_label]
    print("Predicted labels:")
    print(pred_label)
    print()
    print("Prediction probabilities:")
    print(pred_p)
    print()
    print("Reconstructed inputs:")
    print(reconstruct_x)
    print()
    print("Original inputs:")
    print(x)

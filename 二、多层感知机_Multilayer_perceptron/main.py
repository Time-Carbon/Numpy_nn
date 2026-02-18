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

        for i in range(self.layers_num - 1):
            w = np.random.randn(self.layers_shape[i], self.layers_shape[i + 1]).astype(self.dtype) * np.sqrt(2 / self.layers_shape[i]) # He 初始化方法
            b = np.zeros(self.layers_shape[i + 1]).astype(self.dtype)
            self.weight.append(w)
            self.biase.append(b)

    def softmax(self, z):
        z -= np.max(z, axis = 1, keepdims = True)
        z_exp = np.exp(z)
        z_sum = np.sum(z_exp, axis = 1, keepdims = True)
        return z_exp / z_sum

    def leaky_relu(self, z, alpha = 0.01):
        self.d_Lrelu_cache.append(np.where(z > 0, 1, alpha).astype(dtype))
        z = np.maximum(alpha * z, z)
        return z

    def cross_entropy(self, p, q):
        ln_q = np.log(q + 1e-8) # 防止过小超出浮点数表示范围
        ce = p * ln_q
        return np.mean(- np.sum(ce, axis = 1, keepdims = True))

    def forward(self, x, alpha = 0.01):
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
        d_err = (self.z_cache[-1] - y) / x.shape[0] # 经过交叉熵和softmax求导后的偏导
        for i in range(self.layers_num - 2, -1, -1): # 倒序 从 layer_num 到 0
            if i == 0:
                d_err = self.d_Lrelu_cache[i] * d_err
                self.weight[i] -= lr * np.dot(x.T, d_err) # 取梯度平均
                self.biase[i] -= lr * np.mean(d_err, axis = 0)
            elif i == self.layers_num - 2:
                self.weight[i] -= lr * np.dot(self.z_cache[i-1].T, d_err)
                self.biase[i] -= lr * np.mean(d_err, axis = 0)
                d_err = np.dot(d_err, self.weight[i].T)
            else:
                d_err = self.d_Lrelu_cache[i] * d_err
                self.weight[i] -= lr * np.dot(self.z_cache[i-1].T, d_err)
                self.biase[i] -= lr * np.mean(d_err, axis = 0)
                d_err = np.dot(d_err, self.weight[i].T)

    def train(self, x, y, step, note_step = 1, lr = 0.01, alpha = 0.01):
        for i in range(step):
            self.forward(x, alpha)
            loss = self.cross_entropy(y, self.z_cache[-1])
            if i % note_step == 0:
                print(loss)
            self.backward(x, y, lr)
            self.z_cache.clear()
            self.d_Lrelu_cache.clear()

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
    layer_num = 18

    layer_shape = []
    for i in range(layer_num):
        if i == 0:
            layer_shape.append(input_dim)
        elif i == layer_num - 1:
            layer_shape.append(type_num)
        else:
            layer_shape.append(hide_dim)

    mlp = MLP(layer_shape, dtype = dtype)

    mlp.train(x, y, 100, 10, 1e-2)

    print(y)
    print()

    pred_p = mlp.forward(x)
    pred_label = np.argmax(pred_p, axis = 1)
    pred_label = np.eye(2)[pred_label]
    print(pred_label)
    print()
    print(pred_p)
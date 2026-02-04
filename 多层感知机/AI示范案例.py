import numpy as np

class MLP:
    def __init__(self, layer_sizes, learning_rate=0.01):
        """
        初始化MLP
        
        参数:
        layer_sizes: 列表，每一层的神经元数量（包括输入层和输出层）
        learning_rate: 学习率
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.num_layers = len(layer_sizes)
        
        # 初始化权重和偏置
        self.weights = []
        self.biases = []
        
        # 使用Xavier初始化方法
        for i in range(self.num_layers - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def relu(self, x):
        """ReLU激活函数"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """ReLU激活函数的导数"""
        return (x > 0).astype(float)
    
    def sigmoid(self, x):
        """Sigmoid激活函数"""
        # 防止溢出
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """Sigmoid激活函数的导数"""
        return x * (1 - x)
    
    def forward(self, X):
        """
        前向传播
        
        参数:
        X: 输入数据
        
        返回:
        outputs: 每一层的输出（包括输入层）
        activations: 每一层激活后的输出
        """
        # 存储每一层的输出和激活值
        outputs = [X]
        activations = [X]
        
        # 前向传播过程
        for i in range(self.num_layers - 1):
            # 线性变换
            output = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            outputs.append(output)
            
            # 激活函数（最后一层通常不使用激活函数，或者使用特定的输出激活函数）
            if i < self.num_layers - 2:  # 隐藏层使用ReLU
                activation = self.relu(output)
            else:  # 输出层使用线性激活（可根据需要修改）
                activation = output
                
            activations.append(activation)
        
        return outputs, activations
    
    def backward(self, X, y, outputs, activations):
        """
        反向传播
        
        参数:
        X: 输入数据
        y: 真实标签
        outputs: 前向传播中每一层的输出
        activations: 前向传播中每一层激活后的输出
        """
        m = X.shape[0]  # 样本数量
        
        # 计算输出层误差
        # 使用均方误差损失函数
        output_error = activations[-1] - y
        # 如果使用其他损失函数，这里需要相应修改
        
        # 存储每一层的梯度
        dW = [None] * (self.num_layers - 1)
        db = [None] * (self.num_layers - 1)
        
        # 计算输出层的梯度
        dW[-1] = np.dot(activations[-2].T, output_error) / m
        db[-1] = np.sum(output_error, axis=0, keepdims=True) / m
        
        # 反向传播误差
        error = output_error
        for i in range(self.num_layers - 2, 0, -1):
            # 计算第i层的误差
            error = np.dot(error, self.weights[i].T) * self.relu_derivative(outputs[i])
            
            # 计算第i层的梯度
            dW[i-1] = np.dot(activations[i-1].T, error) / m
            db[i-1] = np.sum(error, axis=0, keepdims=True) / m
        
        # 更新权重和偏置
        for i in range(self.num_layers - 1):
            self.weights[i] -= self.learning_rate * dW[i]
            self.biases[i] -= self.learning_rate * db[i]
    
    def compute_loss(self, y_true, y_pred):
        """计算均方误差损失"""
        return np.mean((y_true - y_pred) ** 2)
    
    def train(self, X, y, epochs):
        """
        训练模型
        
        参数:
        X: 输入数据
        y: 真实标签
        epochs: 训练轮数
        """
        for epoch in range(epochs):
            # 前向传播
            outputs, activations = self.forward(X)
            
            # 计算损失
            loss = self.compute_loss(y, activations[-1])
            
            # 反向传播
            self.backward(X, y, outputs, activations)
            
            # 打印损失
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")
    
    def predict(self, X):
        """
        预测
        
        参数:
        X: 输入数据
        
        返回:
        预测结果
        """
        _, activations = self.forward(X)
        return activations[-1]

# 使用示例
if __name__ == "__main__":
    # 生成一些示例数据
    np.random.seed(42)
    X = np.random.randn(100, 256)  # 100个样本，4个特征
    y = np.random.randn(100, 10)  # 100个样本，1个输出
    
    # 创建MLP模型 (4个输入，10个隐藏单元，1个输出)
    mlp = MLP([256, 32, 32, 32, 10], learning_rate=0.01)
    
    # 训练模型
    print("开始训练...")
    mlp.train(X, y, epochs=20000)
    
    # 预测
    predictions = mlp.predict(X)
    print(f"\n前5个样本的预测结果: {predictions[:5].flatten()}")
    print(f"前5个样本的真实标签: {y[:5].flatten()}")

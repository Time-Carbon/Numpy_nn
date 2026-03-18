import numpy as np

# 基于感知机的二分类任务：判断数值大小
# 假设有一个这样的场景，我们在测量水温，当它达到一个阈值时，这个装置就会报警。

# 构建训练数据和标签
np.random.seed(1024)  # 控制随机数，方便复现
x = np.random.uniform(0, 100, size=(500, 1))  # 生成500个训练样本
y = np.where(x >= 60, 1, 0)  # 温度≥60报警

# 特征归一化，防止梯度爆炸，即 x/max(x)
x_normalized = x / 100.0  # 归一化到[0,1]

# 初始化权重和偏置
w = np.array([[0.01]])  # 小的正权重
b = np.array([0.0])

# 训练参数
batch = 20  # 单次训练使用的样本量
lr = 1e-2  # 学习率
epochs = 20  # 训练轮次

print(f"初始权重：{w[0][0]:.6f}，初始偏置：{b[0]:.6f}")


def cross_entropy(p, y):
    return -(y * np.log(p + 1e-8) + (1-y) * np.log(1 - p + 1e-8))


def sigmoid(z):
    # 防止溢出
    z = np.clip(z, -500, 500)
    return 1/(1+np.exp(-z))


# 训练过程
losses = []
for epoch in range(epochs):
    epoch_losses = []
    # 每轮打乱数据
    indices = np.random.permutation(len(x_normalized))
    x_shuffled = x_normalized[indices]
    y_shuffled = y[indices]

    for i in range(0, len(x_shuffled)-batch, batch):
        # 前向传播
        z = np.dot(x_shuffled[i:i+batch], w) + b
        p = sigmoid(z)

        # 计算损失
        loss = np.mean(cross_entropy(p, y_shuffled[i:i+batch]))
        epoch_losses.append(loss)

        # 反向传播梯度
        dz = p - y_shuffled[i:i+batch]
        dw = np.dot(x_shuffled[i:i+batch].T, dz) / batch
        db = np.sum(dz) / batch

        # 参数更新
        w -= lr * dw
        b -= lr * db

    avg_loss = np.mean(epoch_losses)
    losses.append(avg_loss)

    # 每2个epoch打印一次
    if epoch % 2 == 0:
        print(f"Epoch {epoch}, 平均损失：{avg_loss:.6f}")

print(f"\n最终权重：{w[0][0]:.6f}, 最终偏置：{b[0]:.6f}")

# 测试
x_test = np.random.uniform(0, 100, size=(20, 1))
x_test_normalized = x_test / 100.0

z_test = np.dot(x_test_normalized, w) + b
pred = sigmoid(z_test)

print(f"\n测试结果：")
print(f"温度(°C)\t概率\t预测")
for temp, prob in zip(x_test.flatten(), pred.flatten()):
    prediction = "报警" if prob > 0.5 else "不报警"
    print(f"{temp:6.1f}\t\t{prob:.3f}\t{prediction}")

# 验证训练效果
train_pred = sigmoid(np.dot(x_normalized, w) + b)
train_accuracy = np.mean((train_pred > 0.5) == (y == 1))
print(f"\n训练准确率：{train_accuracy*100:.2f}%")

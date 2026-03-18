import numpy as np
from src._2_Multi_Layer_Perceptron.mlp import MLP

def build_data(dataType, min, max):
    x_0 = np.random.uniform(min, max, size=(
        100, 1)).astype(dataType)  # 构建异或为0的数据
    x_0 = x_0 + np.zeros((x_0.shape[0], 2), dtype=dataType)
    x_0 = x_0 / (max - min)  # 输入数据归一化
    y_0 = np.zeros((x_0.shape[0]), dtype=np.int8)  # 对应数量的标签
    y_0 = np.eye(2)[y_0].astype(dataType)
    # 转换为one-hot标签，即有两种不同的标签[输出为0，输出为1]，被选中的标签对应的值为1
    # 比如one-hot标签为[1,0]，则表示选中的标签是第一个标签，即“输出为0”

    # 构建异或为1的数据(由于有去重和筛选环节，因此需要留出多余的数来确保经过这些操作后数据能到达和x_0同等规模)
    x_1 = np.random.uniform(min, max, size=(
        np.round(1.1 * x_0.shape[0]).astype(np.int64), 2)).astype(dataType)
    # 生成shape=(x_0的行数， 特征为2)的数据
    mask = x_1[:, 0] != x_1[:, 1]
    x_1 = x_1[mask]  # 取两个数不同的数组
    x_1 = np.unique(x_1, axis=0)
    x_1 = x_1 / np.max(x_1)
    y_1 = np.ones((x_1.shape[0]), dtype=np.int8)
    y_1 = np.eye(2)[y_1].astype(dataType)

    x = np.concatenate((x_1, x_0), axis=0)  # 沿着行的方向合并数据
    y = np.concatenate((y_1, y_0), axis=0)

    return x, y


if __name__ == "__main__":
    dataType = np.float32  # 设置存储和计算时使用的数据类型，可以节约内存，并提高计算速度

    x, y = build_data(dataType, -10, 10)

    input_dim = x.shape[1]  # 输入层维度，即输入数据有多少个
    hide_dim = 4 * input_dim
    # 隐藏层的维度拓展成4倍于输入维度是基于模型性能和计算成本的平衡考量。至于该比例最初的出现是在《Attention Is All You Need》中，研究团队通过实验得出的经验数值，并以此作为基准进行实验。
    # 具体比例需要在实际操作时，根据拟合能力和计算效率进行平衡确定，可能大于4倍，也可能小于4倍。
    output_dim = y.shape[1]  # 输出层维度，即输出时有多少个类别

    mlp = MLP([input_dim, hide_dim, output_dim], dtype=dataType)

    mlp.train(x, y, 1000, 100, 1e-3)

    x_test = np.array([
        [2, 3],
        [2, 2],
        [4, 5],
        [5, 5]
    ]).astype(dataType)
    x_test = x_test / np.max(x_test)

    y_test = np.array([
        [0, 1],
        [1, 0],
        [0, 1],
        [1, 0]
    ]).astype(dataType)

    print(y_test)  # 输出真实标签，方便我们后续对比
    print()

    pred_p = mlp.forward(x_test)  # 获取训练好的多层感知机的输出
    pred_label = np.argmax(pred_p, axis=1)
    pred_label = np.eye(2)[pred_label]  # 转换为one-hot标签

    print(pred_label)
    print()
    print(pred_p)

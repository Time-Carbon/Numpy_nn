import numpy as np
from mlp import MLP

dataType = np.float16  # 设置存储和计算时使用的数据类型，可以节约内存，并提高计算速度


def normal(x, min, max):
    norm = (x - min) / (max - min)
    return norm


def build_data(dataType, min, max):
    '''
    生成未标注的数据
    '''
    x_0 = np.random.uniform(min, max, size=(
        100, 1)).astype(dataType)  # 构建异或为0的数据
    x_0 = x_0 + np.zeros((x_0.shape[0], 2), dtype=dataType)

    # 构建异或为1的数据(由于有去重和筛选环节，因此需要留出多余的数来确保经过这些操作后数据能到达和x_0同等规模)
    x_1 = np.random.uniform(min, max, size=(
        np.round(1.1 * x_0.shape[0]).astype(np.int64), 2)).astype(dataType)
    # 生成shape=(x_0的行数， 特征为2)的数据
    mask = x_1[:, 0] != x_1[:, 1]
    x_1 = x_1[mask]  # 取两个数不同的数组
    x_1 = np.unique(x_1, axis=0)

    x_0 = normal(x_0, min, max)
    x_1 = normal(x_1, min, max)

    x = np.concatenate((x_1, x_0), axis=0)  # 沿着行的方向合并数据
    index = np.random.permutation(x.shape[0])
    x = x[index]

    return x


def p_to_l(label_p: np.ndarray, top_k: int, min_p: float):
    '''
    将预测概率转换为伪标签
    '''
    label = []
    index = []

    sample_size = top_k // label_p.shape[1]
    sample_size = np.maximum(1, sample_size)  # 每个类别至少一个
    cls_index = np.arange(label_p.shape[1])

    for cls in cls_index:
        cls_eye = np.zeros(label_p.shape[0], dtype=np.int64) + cls
        cls_eye = np.eye(label_p.shape[1])[cls_eye]
        cls_mask = (cls_eye == 1)

        cls_p = label_p[cls_mask]
        sorted_cls_p_index = np.argsort(cls_p)[::-1]  # 由大到小排序

        keep_mask = (cls_p > min_p)

        label_masked_index = sorted_cls_p_index[keep_mask]
        label_masked = cls_eye[keep_mask]

        index += label_masked_index[:sample_size].tolist()
        label += label_masked[:sample_size].tolist()

    return np.array(label, dtype=dataType), np.array(index[:top_k], dtype=np.int64)


def weaky_sl(model: MLP, basedata_x: np.ndarray, basedata_y: np.ndarray, unlabel_data: np.ndarray, lr: float, top_k: int, min_p: float):
    model.train(basedata_x, basedata_y, lr=1e-2, epoch=1000, note_step=10)
    print("预训练完成，进行标注")

    train_x: list = basedata_x.tolist()  # 训练数据转换为列表，方便添加内容
    train_y: list = basedata_y.tolist()

    step = unlabel_data.shape[0] // top_k  # 确保每个数据都能被训练
    for i in range(step):
        fake_label_p = model.forward(unlabel_data)
        fake_label, fake_label_index = p_to_l(
            fake_label_p, top_k, min_p)

        if fake_label.size != 0:
            unlabel_x = unlabel_data[fake_label_index].tolist()
            train_x = train_x + unlabel_x
            train_y = train_y + fake_label.tolist()

            unchoose_index = ~np.isin(
                np.arange(unlabel_data.shape[0]), fake_label_index)
            unlabel_data = unlabel_data[unchoose_index]

        x = np.array(train_x)
        y = np.array(train_y)

        index = np.random.permutation(x.shape[0])

        x = x[index]
        y = y[index]

        model.train(x, y, epoch=20, lr=1e-4, batch=16, note_step=10)

    return model


if __name__ == "__main__":
    base_data_x = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1]
    ], dtype=dataType)
    base_data_y = np.array([
        # [标签0，标签1]
        [1, 0],
        [0, 1],
        [0, 1],
        [1, 0]
    ], dtype=dataType)

    unlabel_x = build_data(dataType, -10, 10)

    input_dim = base_data_x.shape[1]  # 输入层维度，即输入数据有多少个
    hide_dim = 4 * input_dim
    output_dim = base_data_y.shape[1]  # 输出层维度，即输出时有多少个类别

    mlp = MLP([input_dim, hide_dim, output_dim], dtype=dataType)

    mlp = weaky_sl(mlp, base_data_x, base_data_y,
                   unlabel_x, 1e-4, 4, 0.6)

    x_test = np.array([
        [2, 3],
        [2, 2],
        [4, 5],
        [5, 5]
    ]).astype(dataType)
    x_test = (x_test - x_test.min()) / (x_test.max() - x_test.min())

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

"""
测试 weaky_SL.py 中的非 MLP 模块
"""
import pytest
import numpy as np
from weaky_SL import normal, build_data, p_to_l


class TestNormal:
    """测试 normal 函数"""

    def test_normal_zero_to_one(self):
        """测试归一化到 [0, 1]"""
        result = normal(x=np.array([0, 5, 10]), min=0, max=10)
        expected = np.array([0.0, 0.5, 1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_normal_negative(self):
        """测试负数归一化"""
        # (-5 - (-10)) / (10 - (-10)) = 5 / 20 = 0.25
        # (0 - (-10)) / (10 - (-10)) = 10 / 20 = 0.5
        # (5 - (-10)) / (10 - (-10)) = 15 / 20 = 0.75
        result = normal(x=np.array([-5, 0, 5]), min=-10, max=10)
        expected = np.array([0.25, 0.5, 0.75])
        np.testing.assert_array_almost_equal(result, expected)

    def test_normal_large_values(self):
        """测试大数值归一化"""
        result = normal(x=np.array([1000, 2000, 3000]), min=0, max=10000)
        expected = np.array([0.1, 0.2, 0.3])
        np.testing.assert_array_almost_equal(result, expected)


class TestBuildData:
    """测试 build_data 函数"""

    def test_build_data_creates_xor_data(self):
        """测试构建异或数据"""
        data = build_data(min=-10, max=10, data_size=100)
        assert data.shape[0] > 90
        assert data.shape[1] == 2  # 特征维度为 2

    def test_build_data_returns_normalized_data(self):
        """测试返回的数据是归一化的"""
        data = build_data(min=-10, max=10, data_size=100)
        min_val = data.min()
        max_val = data.max()
        assert min_val >= -0.1  # 接近 0
        assert max_val <= 1.1   # 接近 1

    def test_build_data_creates_xor_xor_data(self):
        """测试构建的数据满足异或特性"""
        # 异或为 0 的数据：x[0] == x[1]
        # 异或为 1 的数据：x[0] != x[1]
        # 由于有去重和筛选，实际数据中异或为 1 的应该更多
        data = build_data(min=-10, max=10, data_size=200)
        xor_zero = np.sum(data[:, 0] == data[:, 1])
        xor_one = np.sum(data[:, 0] != data[:, 1])
        # 异或为 1 的数据应该更多（因为筛选了异或为 0 的）
        assert xor_one >= xor_zero


class TestPToL:
    """测试 p_to_l 函数"""

    def test_p_to_l_basic(self):
        """测试基本的伪标签生成"""
        # 创建一个简单的预测概率矩阵
        label_p = np.array([
            [0.9, 0.1],  # 样本 1: 类别 0 概率高
            [0.1, 0.9],  # 样本 2: 类别 1 概率高
            [0.5, 0.5],  # 样本 3: 不确定
            [0.8, 0.2],  # 样本 4: 类别 0 概率高
        ])
        label, index = p_to_l(label_p, top_k=4, min_p=0.5)

        assert len(label) > 0
        assert len(index) > 0

    def test_p_to_l_min_p_threshold(self):
        """测试 min_p 阈值"""
        # label_p.shape[1] = 2, sample_size = 4 // 2 = 2
        # 类别 0: [0.95, 0.1, 0.3, 0.2]，> 0.7 的只有 [0.95]，取 min(1, 2) = 1 个
        # 类别 1: [0.05, 0.9, 0.7, 0.8]，> 0.7 的有 [0.9, 0.7, 0.8]，取 min(3, 2) = 2 个
        # 总共返回 1 + 2 = 3 个样本
        label_p = np.array([
            [0.95, 0.05],
            [0.1, 0.9],
            [0.3, 0.7],
            [0.2, 0.8],
        ])
        label, index = p_to_l(label_p, top_k=4, min_p=0.7)
        
        assert len(index) == 3  # 1 + 2 = 3 个样本

    def test_p_to_l_top_k(self):
        """测试 top_k 参数"""
        # label_p.shape[1] = 2, sample_size = 2 // 2 = 1
        # 类别 0: [0.9, 0.8, 0.7, 0.6]，> 0.5 的有 4 个，取 min(4, 1) = 1 个
        # 类别 1: [0.1, 0.2, 0.3, 0.4]，> 0.5 的有 0 个，取 0 个
        # 总共返回 1 个标签
        label_p = np.array([
            [0.9, 0.1],
            [0.8, 0.2],
            [0.7, 0.3],
            [0.6, 0.4],
        ])
        label, index = p_to_l(label_p, top_k=2, min_p=0.5)
        
        assert len(label) == 1  # 只有类别 0 有满足条件的样本
        assert len(index) == 1

    def test_p_to_l_empty_result(self):
        """测试当没有样本满足条件时的情况"""
        label_p = np.array([
            [0.3, 0.7],
            [0.4, 0.6],
        ])
        label, index = p_to_l(label_p, top_k=2, min_p=0.8)
        
        # 没有样本满足条件，应该返回空数组
        assert len(label) == 0
        assert len(index) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
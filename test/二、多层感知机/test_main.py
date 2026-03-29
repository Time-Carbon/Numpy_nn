import numpy as np
import src
from src._2_Multi_Layer_Perceptron.mlp import MLP
import pytest

class TestClass:
    def test_init(self):
        mlp = MLP([2,4,2], np.float32)

        assert mlp.biase.__len__ == 2
        assert mlp.weight.__len__ == 2
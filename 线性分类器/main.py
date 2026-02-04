from typing import Any
import numpy as np

dtype = np.float32

S = 64 # 样本数量
T = 10 # 分类数量
D = 32 # 维度

X = np.random.uniform(-1, 1, size = (S,D)).astype(dtype) # 模拟样本
W = np.random.uniform(-1, 1, size = (D,T)).astype(dtype)
B = np.zeros(T).astype(dtype) 

true_label = np.random.randint(0, T, size = S)
Y_true = np.eye(T)[true_label].astype(dtype) # 模拟真实值 shape = (S, T)
Y_hat = np.zeros((S, T)).astype(dtype)

lr = 0.01

def softmax(z):
    z -= np.max(z, axis = 1, keepdims = True)
    z -= np.min(z, axis = 1, keepdims = True)
    z_exp = np.exp(z)
    z_sum = np.sum(z_exp, axis = 1, keepdims = True)
    return z_exp / z_sum

def cross_entropy(p,q):
    ce = -np.sum(p*np.log(q), axis = 0)
    return np.mean(ce)

for i in range(0, 1000):
    Y_hat = np.dot(X, W) + B # shape = (S, T)
    Y_hat = softmax(Y_hat)

    d_Y = Y_hat - Y_true
    Loss = cross_entropy(Y_true, Y_hat)

    W -= lr*np.dot(X.T, d_Y)
    B -= lr*np.mean(d_Y, axis = 0)

    if i%100 == 0:
        print(Loss)

print(Y_true)
print(Y_hat) 
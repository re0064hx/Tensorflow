import numpy as np

rng = np.random.RandomState(123)

d = 2       # dim of input
N = 10      # Num of data in each pattern
mean = 5    # Mean of data which generate output from neuron

def y(x):
    return step(np.dot(w, x) + b)

def step(x):
    return 1*(x>0)

# 真値を出力する関数
def t(i):
    if i > N:
        return 0
    else:
        return 1


# データの生成
x1 = rng.randn(N, d) + np.array([0, 0])
x2 = rng.randn(N, d) + np.array([mean, mean])
# 連結
x = np.concatenate((x1, x2), axis=0)

# 重み
w = np.zeros(d)
# bias
b = 0

# 学習
while True:
    classified = True           # データがすべて分類されているかを判別するフラグ
    for i in range(N*2):
        delta_w = (t(i) - y(x[i]))*x[i]
        delta_b = (t(i) - y(x[i]))
        w += delta_w
        b += delta_b
        classified *= all(delta_w == 0) * (delta_b == 0)    # 重みすべてとバイアスの値がすべて0ならTrue
    if classified:
        break

print(y([0,0]))
print(y([7,7]))

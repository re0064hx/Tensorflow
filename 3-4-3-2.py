import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

# データ設定
X = np.array([[0,0], [0,1], [1, 0], [1,1]])
Y = np.array([[0], [1], [1], [1]])

# 層の定義
'''
Sequential: 層構造の定義
Dense: 入力，出力の次元設定
Activation: 活性化関数の設定
'''
# Pattern 1
model = Sequential([
    Dense(input_dim=2, units=1),
    Activation('sigmoid')
])

# # Pattern 2
# model = Sequential()
# model.add(Dense(input_dim=2, units=1))
# model.add(Activation('sigmoid')

# 確率的勾配降下法の設定
model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.1))

# 学習の実行
model.fit(X, Y, epochs=200, batch_size=1)

# 学習結果
classes = model.predict_classes(X, batch_size=1)
prob = model.predict_proba(X, batch_size=1)

print('classified:')
print(Y == classes)
print('Probability:')
print(prob)

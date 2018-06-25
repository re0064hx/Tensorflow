import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

GAMMA = 0.1

# XORゲートの値
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

'''
ネットワークの設定
truncated_normal():切断正規分布に従うデータの生成
    重みを最初に０で初期化してしまうと，誤差逆伝搬がうまく行かない可能性があるため
活性化関数：sigmoid関数を設定
correct_prediction:
    出力yの値が0.5以上かどうかで0, 1判定を行い，その後真値t との比較を行いTrue/Falseを返す
'''
# ネットワーク変数
x = tf.placeholder(tf.float32, shape=[None, 2])
t = tf.placeholder(tf.float32, shape=[None, 1])
## 出力層の設定
W = tf.Variable(tf.truncated_normal([2, 2]))
b = tf.Variable(tf.zeros([2]))
h = tf.nn.sigmoid(tf.matmul(x, W) + b)
## 隠れ層の設定
V = tf.Variable(tf.truncated_normal([2, 1]))
c = tf.Variable(tf.zeros([1]))
y = tf.nn.sigmoid(tf.matmul(h, V) + c)

# 交差エントロピー誤差関数(二値分類の場合の関数)
cross_entropy = -tf.reduce_sum(t*tf.log(y) + (1-t)*tf.log(1-y))
# 確率的勾配降下法
train_step = tf.train.GradientDescentOptimizer(GAMMA).minimize(cross_entropy)
# 学習結果判定用
correct_prediction = tf.equal(tf.to_float(tf.greater(y, 0.5)), t)

# セッション設定・初期化
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

'''
学習の実行
'''
for epoch in range(4000):
    sess.run(train_step, feed_dict={
        x: X,
        y: Y
    })
    if epoch % 1000 == 0:
        print('epoch:　%d' % epoch)

'''
学習結果
'''
classified = correct_prediction.eval(session=sess, feed_dict={
    x: X,
    t: Y
})
prob = y.eval(session=sess, feed_dict={
    x: X
})

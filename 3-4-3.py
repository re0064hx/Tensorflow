import numpy as np
import tensorflow as tf

GAMMA = 0.1

# データ設定
X = np.array([[0,0], [0,1], [1, 0], [1,1]])
Y = np.array([[0], [1], [1], [1]])

#　変数群初期化
w = tf.Variable(tf.zeros([2, 1]))
b = tf.Variable(tf.zeros([1]))

#　モデルの構成
x = tf.placeholder(tf.float32, shape=[None, 2])
t = tf.placeholder(tf.float32, shape=[None, 1])
y = tf.nn.sigmoid(tf.matmul(x, w) + b)          # Sigmoid function (Output)
# 交差エントロピー誤差関数
cross_entropy = - tf.reduce_sum(t*tf.log(y) + (1-t)*tf.log(1-y))
# 勾配計算，交差エントロピー誤差関数の最小化
train_step = tf.train.GradientDescentOptimizer(GAMMA).minimize(cross_entropy)
# yの値が0.5以上かどうかによって発火を計算し，それが真値tと等しいかを計算
correct_prediction = tf.equal(tf.to_float(tf.greater(y, 0.5)), t)

# セッションの設定・初期化, ここで初めてモデルの定義で宣言した変数・式の初期化が行われる
init = tf.global_variables_initializer()
sess = tf.Session()         # クラスの設定
sess.run(init)              # initの実行

# 学習の開始
for epoch in range(200):
    sess.run(train_step, feed_dict={x: X, t: Y})        # train_stepの実行

# 学習結果の確認
classified = correct_prediction.eval(session=sess, feed_dict={x:X, t:Y})
print('classified:')
print(classified)

prob = y.eval(session=sess, feed_dict={x:X, t:Y})
print('Probability:')
print(prob)

# tf.variableの値はsess.run()にて取得可能
print('w:', sess.run(w))
print('b:', sess.run(b))

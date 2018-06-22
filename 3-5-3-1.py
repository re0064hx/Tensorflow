'''
多クラスロジスティック回帰の実装
ランダムに生成されたデータｘと正解データｙにより，学習の結果を確認
学習には，交差エントロピー誤差関数に対して，確率的勾配降下法を適用することで求める
'''
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

GAMMA = 0.1 # 学習率の設定

M = 2       # 入力データ次元
K = 3       # クラス数
n = 100     # クラスごとのデータ数
N = n*K     # 全データ数

# サンプルデータ群
X1 = np.random.randn(n, M) + np.array([0, 10])          # randn(行，列)
X2 = np.random.randn(n, M) + np.array([5, 5])
X3 = np.random.randn(n, M) + np.array([10, 0])
# 正解データ群
Y1 = np.array([[1, 0, 0] for i in range(n)])
Y2 = np.array([[0, 1, 0] for i in range(n)])
Y3 = np.array([[0, 0, 1] for i in range(n)])
# データの結合
X = np.concatenate((X1, X2, X3), axis=0)
Y = np.concatenate((Y1, Y2, Y3), axis=0)
# # 乱数生成データの描画
# plt.scatter(X[:,0], X[:,1])
# plt.show()
# sleep(3)
# plt.close()

'''
分類器の設定
重み：M＊K行列
バイアス：K個の要素を持つ行ベクトル
tf.placeholder内のshapeについて：
    M個で構成されるベクトル（一次のテンソル）が何個でも入ってこれるように，Noneにしている
matmul:行列の積
reduction_indicesはどの方向に向かって和を取るかを決めている
    0:全要素で計算（和なら総和）
    １：一列ごと？（一階なので？？）
学習結果の確認
    tf.argmax(変数，パラメータ)：第2パラメーターに1をセットすると、行ごとに最大値となっている列の要素のインデックスを返す
    tf.equal（ベクトル，ベクトル）:２つのベクトルのそれぞれの要素が一致しているかどうかを判定，True / Falseを返す
'''
W = tf.Variable(tf.zeros([M,K]))
b = tf.Variable(tf.zeros([K]))

x = tf.placeholder(tf.float32, shape=[None, M])
t = tf.placeholder(tf.float32, shape=[None, K])
y = tf.nn.softmax(tf.matmul(x, W) + b)

# ミニバッチごとの平均を求めて，交差エントロピー誤差関数を求める
cross_entropy = tf.reduce_mean(-tf.reduce_sum(t*tf.log(y), reduction_indices=[1]))
# この関数を確率的勾配降下法によって最適化
train_step = tf.train.GradientDescentOptimizer(GAMMA).minimize(cross_entropy)
# 学習結果確認用
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(t,1))

# セッションの設定・初期化, ここで初めてモデルの定義で宣言した変数・式の初期化が行われる
init = tf.global_variables_initializer()
sess = tf.Session()         # クラスの設定
sess.run(init)              # initの実行

# for debug
print("setup finished.")

'''
学習の実行
epoch:一つの訓練データを繰り返し学習する回数
feed_dict:バッチサイズ分ひとかたまりにして学習をする
'''
batch_size = 50
# 切り捨て除算: //
n_batches = N // batch_size

for epoch in range(20):
    X_, Y_ = shuffle(X, Y)
    print('\r epoch: %d' % epoch, end='')

    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size

        sess.run(train_step, feed_dict={
            x: X_[start:end],
            t: Y_[start:end]
        })
print()

'''
学習結果の確認
'''
# 学習確認用にデータを取得
X_, Y_ = shuffle(X, Y)

classified = correct_prediction.eval(session=sess, feed_dict={
    x: X_[0:10],
    t: Y_[0:10]
})
prob = y.eval(session=sess, feed_dict={
    x: X_[0:10]
})

print('classified:')
print(classified)
print()
print('Output Probability:')
print(prob)

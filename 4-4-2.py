import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

np.random.seed(0)
tf.set_random_seed(1234)

''' ====== Deep Neural Network クラスの設定 ====== '''
class DNN(object):
    def __init__(self, n_in, n_hiddens, n_out):
        # 初期化処理(コンストラクタ)
        # モデルの構成をここで決定する（層の数などを引数に受け取り，クラス内の変数を初期化）
        self.n_in = n_in
        self.n_hiddens = n_hiddens
        self.n_out = n_out
        self.weights = []
        self.biases = []
        self._x = None
        self._t = None
        self._keep_prob = None
        self._sess = None
        # dict型で初期化
        self._history={
            'accuracy': [],
            'loss': []
        }

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_variables(self, shape):
        initial = tf.zeros(shape)
        return tf.Variable(initial)

    def interface(self, x, keep_prob):
        '''
        モデルの定義

        for文内のenumerate:
            配列（リスト）の値をインデックス付きで取得
            今回は，配列のインデックス値をiに，配列の要素値をn_hidden
        入力層→隠れ層，隠れ層→隠れ層 (前の出力が次の入力になるように設定)
        ※ 重み，バイアスの変数群のリストのインデックスが-1なのは，リストの最後尾の値を選択するため
        '''
        for i, n_hidden in enumerate(self.n_hiddens):
            if i == 0:
                input = x
                input_dim = self.n_in
            else:
                input = output
                imput_dim = self.n_hiddens[i-1]

            # 重み・バイアスの設定（関数呼び出し）
            self.weights.append(self.weight_variable([input_dim, n_hidden]))
            self.biases.append(self.bias_variables([n_hidden]))

            h = tf.nn.relu(tf.matmul(input, self.weights[-1]) + self.biases[-1])
            output = tf.nn.dropout(h, keep_prob)

        # 隠れ層→出力層
        self.weights.append(self.weight_variable([self.n_hiddens[-1], self.n_out]))
        self.biases.append(self.bias_variable([self.n_out]))

        y = tf.nn.softmax(tf.matmul(output, self.weights[-1]) + self.biases[-1])
        return y

    def loss(self, y, t):
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(t*tf.log(y), reduction_indices=[1]))
        return cross_entropy

    def training(self, loss):
        optimizer = tf.train.GradientDescentOptimizer(GAMMA)
        train_step = optimizer.minimize(loss)
        return train_step

    def accuracy(self, y, t):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy

    def fit(self, X_train, Y_train, epochs=100, batch_size=100, p_keep=0.5, varbose=1):
        '''
        学習の処理
        '''
        # evaluate()用に保持
        self._x = x
        self._t = t
        self._keep_prob = keep_prob

        y = self.interface(x, keep_prob)
        loss = self.loss(y, t)
        train_step = self.training(loss)
        accuracy = self.accuracy(y, t)

        init = tf.global_variables_initializer()
        sess = tf.Sesson()
        sess.run(init)

        # evaluate() 用に保持
        self._sess = sess

        N_train = len(X_train)
        n_batches = N_train // batch_size

        for epoch in range(epochs):
            X_, Y_ = shuffle(X_train, Y_train)

            for i in range(n_batches):
                start = i * batch_size
                end = start + batch_size

                sess.run(train_step, feed_dict={
                    x: X_[start:end],
                    t: Y_[start:end],
                    keep_prob: p_keep
                })

            loss_ = loss.eval(session=sess, feed_dict={
                x: X_train,
                t: Y_train,
                keep_prob: 1.0
            })

            accuracy_ = accuracy.eval(session=sess, feed_dict={
                x: X_train,
                t: Y_train,
                keep_prob: 1.0
            })

            # 値の記録
            self._history['loss'].append(loss_)
            self._history['accuracy'].append(accuracy_)

            if verbose:
                print('epoch: ', epoch,
                    ' loss: ', loss_,
                    ' accuracy', accuracy_)
        return self._history


    def evaluate(self, X_test, Y_test):
        # 評価の処理
        return self.accuracy.eval(session=self._sess, feed_dict={
            self._x: X_test,
            self._t: Y_test,
            self._keep_prob: 1.0
        })

''' ====== main処理 ====== '''
if __name__ == '__main__':
    # データの生成
    N_train = 20000
    N_validation = 4000

    mnist = datasets.fetch_mldata('MNIST original', data_home='.')

    n = len(mnist.data)
    N = 10000  # MNISTの一部を使う
    indices = np.random.permutation(range(n))[:N]  # ランダムにN枚を選択

    X = mnist.data[indices]
    y = mnist.target[indices]
    Y = np.eye(10)[y.astype(int)]  # 1-of-K 表現に変換

    X_train, X_test, Y_train, Y_test = \
        train_test_split(X, Y, train_size=N_train)

    # 生成した訓練データを更に訓練データと検証データに分類
    X_train, X_validation, Y_train, Y_varidation = \
        train_test_split(X_train, Y_train, test_size=N_validation)

    # DNNクラス設定
    model = DNN(n_in=len(X[0]),
        n_hiddens=[200, 200, 200],
        n_out=len(Y[0]))

    # 学習の設定
    model.fit(X_train, Y_train, nb_epoch=30, batch_size=200, p_keep=0.5)

    model.evaluate(X_test, Y_test)

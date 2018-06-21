import numpy as np
import tensorflow as tf

sess = tf.InteractiveSession()

# 画像をグレースケールで読み込み，浮動小数点型データに変換
png = tf.read_file('03-02-original.png')
image = tf.image.decode_png(png, channels=1, dtype=tf.uint8)
image_float = tf.to_float(image)
# tf.nn.conv2dメソッドを適用するために4階のテンソルに変換
image_reshape = tf.reshape(image_float, [-1, 32, 32, 1])

#カーネルの作成
kernel = tf.constant(
    [
        [ 0, -1, -1, -1,  0],
        [-1,  0,  3,  0, -1],
        [-1,  3,  0,  3, -1],
        [-1,  0,  3,  0, -1],
        [ 0, -1, -1, -1,  0]
    ],
    dtype=tf.float32)
#tf.nn.conv2dメソッドを適用するために4階のテンソルに変換
kernel_reshape = tf.reshape(kernel, [5, 5, 1, 1])

#ストライド幅
#3ピクセルずつ動かす
strides = [1,3,3,1]

# 畳み込み
convolution_result = tf.nn.conv2d(
    image_reshape,
    kernel_reshape,
    strides=strides,
    padding='VALID'
)

# 結果を見やすいように2階のテンソルに変換
tf.reshape(convolution_result, [10, 10]).eval()


# ゼロパディングありの畳み込み
convolution_result_with_padding = tf.nn.conv2d(
  image_reshape,
  kernel_reshape,
  strides=strides,
  padding='SAME'
)
# ゼロパディングで画像サイズが変わっているため結果のサイズも異なる
tf.reshape(convolution_result_with_padding, [11, 11]).eval()

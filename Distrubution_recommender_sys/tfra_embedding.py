import os, sys, json,time
import tensorflow as tf
from tensorflow.keras.layers import Dense
import tensorflow_datasets as tfds
import tensorflow_recommenders_addons as tfra

tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_resource_variables()

# tfra.dynamic_embedding.get_variable(
#         name="user_dynamic_embeddings",
#         dim=self.embedding_size,
#         devices=self.devices,
#         initializer=tf.keras.initializers.RandomNormal(-1, 1))

ps_num = 4
devices = ["/job:ps/replica:0/task:{}".format(idx) for idx in range(ps_num)]


embedding_size = 32
# 随机生成一个embedding
word_embed = tfra.dynamic_embedding.get_variable(
        name="user_dynamic_embeddings",
        dim=embedding_size,
        devices=devices,
        initializer=tf.keras.initializers.RandomNormal(-1, 1))

movie_id_val = [1, 2, 3, 0, 3, 2, 1]

movie_id_val = tf.convert_to_tensor(movie_id_val, dtype=tf.int64)


movie_id_weights, movie_id_trainable_wrapper = tfra.dynamic_embedding.embedding_lookup(
        params=word_embed,
        ids=movie_id_val,
        name="movie-id-weights",
        return_trainable=True)


tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))

# movie_id_weights_nn = tf.nn.embedding_lookup(word_embed, movie_id_val)

print(movie_id_weights)
print(movie_id_trainable_wrapper)
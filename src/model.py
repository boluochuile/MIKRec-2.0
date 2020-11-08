from modules import *
import numpy as np
import os
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, \
    Dropout, Embedding, Flatten, Input

class MSARec(tf.keras.Model):
    def __init__(self, n_mid, embedding_dim, ffn_hidden_unit, batch_size, num_heads, num_interest,
                 dropout_rate=0.2, maxlen=256, norm_training=True, num_blocks=2):

        super(MSARec, self).__init__()

        self.maxlen = maxlen
        self.batch_size = batch_size
        self.n_mid = n_mid
        self.neg_num = 10
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate

        self.item_embed = Embedding(
            input_dim=n_mid + 1,
            output_dim=self.embedding_dim,
            embeddings_initializer='random_uniform',
            embeddings_regularizer=tf.keras.regularizers.l2(0.01),
            input_length=maxlen,
            name='item_embed'
        )

        self.attention_block = [SelfAttentionBlock(self.embedding_dim, num_heads, ffn_hidden_unit,
                                                   dropout_rate, norm_training, False) for _ in range(num_blocks)]

        self.dense = tf.keras.layers.Dense(self.embedding_dim, activation='relu')

        self.emb_dropout = tf.keras.layers.Dropout(self.dropout_rate)

    def output_item(self):
        return self.item_embed.weights[0]

    def sampled_softmax_loss(self, labels, logits):
        labels = tf.cast(labels, tf.int64)
        labels = tf.reshape(labels, [-1, 1])
        logits = tf.cast(logits, tf.float32)
        proj_w = self.item_embed.weights[0]
        proj_b = tf.constant(0., shape=(self.n_mid, ))

        return tf.reduce_mean(
            tf.nn.sampled_softmax_loss(
                proj_w,
                proj_b,
                labels,
                logits,
                num_sampled=10 * self.batch_size,
                num_classes=self.n_mid
            )
        )

    def positional_encoding(self, seq_inputs):
        encoded_vec = [pos / np.power(10000.0, 2 * i / self.embedding_dim)
                       for pos in range(seq_inputs.shape[-1]) for i in range(self.embedding_dim)]
        encoded_vec[::2] = np.sin(encoded_vec[::2])
        encoded_vec[1::2] = np.cos(encoded_vec[1::2])
        encoded_vec = tf.reshape(tf.convert_to_tensor(encoded_vec, dtype=tf.float32), shape=(-1, self.embedding_dim))

        return encoded_vec

    def call(self, inputs):
        hist_item, hist_mask = inputs

        self.mask = tf.expand_dims(hist_mask, -1)

        seq_embed = self.item_embed(hist_item)
        pos_encoding = tf.expand_dims(self.positional_encoding(hist_item), axis=0)
        seq_embed += pos_encoding
        seq_embed = self.emb_dropout(seq_embed)

        seq_embed *= self.mask

        # (b, sql, dim)
        att_outputs = seq_embed
        for block in self.attention_block:
            att_outputs = block(att_outputs)

        # (b, sql * dim)
        att_outputs = Flatten()(att_outputs)

        self.user_eb = self.dense(att_outputs)

        # outputs = tf.nn.sigmoid(tf.reduce_sum(tf.multiply(self.user_eb, item_embed), axis=1, keepdims=True))
        # outputs = self.user_eb

        return self.user_eb
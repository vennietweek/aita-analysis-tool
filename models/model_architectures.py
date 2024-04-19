import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, GRU, SimpleRNN, Conv1D, GlobalMaxPooling1D, MultiHeadAttention, Dropout, Flatten, Bidirectional, LayerNormalization
from tensorflow.keras.regularizers import l2

class RNNModel(Model):
    def __init__(self, input_shape, vocab_size, embedding_dim):
        super(RNNModel, self).__init__()
        self.embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_shape=input_shape)
        self.biLSTM = Bidirectional(LSTM(32, return_sequences=False))
        self.dense = Dense(64, activation="relu")
        self.dropout = Dropout(0.1)
        self.output_layer = Dense(1, activation='sigmoid')
    
    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.biLSTM(x)
        x = self.dense(x)
        x = self.dropout(x)
        return self.output_layer(x)

class RNNAttentionModel(Model):
    def __init__(self, input_shape, vocab_size, embedding_dim):
        super(RNNAttentionModel, self).__init__()
        self.embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_shape=(input_shape))
        self.attention = MultiHeadAttention(num_heads=4, key_dim=32)
        self.biLSTM1 = Bidirectional(LSTM(32, return_sequences=True))
        self.biLSTM2 = Bidirectional(LSTM(32, return_sequences=False))
        self.dense = Dense(32, activation="relu")
        self.output_layer = Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.attention(x, x)
        x = self.biLSTM1(x)
        x = self.biLSTM2(x)
        x = self.dense(x)
        return self.output_layer(x)

class CNNModel(Model):
    def __init__(self, input_shape, vocab_size, embedding_dim):
        super(CNNModel, self).__init__()
        self.embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_shape=input_shape)
        self.conv1 = Conv1D(filters=64, kernel_size=10, activation='relu')
        self.pool = GlobalMaxPooling1D()
        self.conv2 = Conv1D(filters=64, kernel_size=5, activation='relu')
        self.pool2 = GlobalMaxPooling1D()
        self.conv3 = Conv1D(filters=64, kernel_size=3, activation='relu')
        self.pool3 = GlobalMaxPooling1D()
        self.concat = tf.keras.layers.Concatenate()
        self.dense = Dense(32, activation='relu')
        self.output_layer = Dense(1, activation='sigmoid')
    
    def call(self, inputs):
        x = self.embedding(inputs)
        conv1 = self.conv1(x)
        conv1 = self.pool(x)
        conv2 = self.conv2(x)
        conv2 = self.pool2(x)
        conv3 = self.conv3(x)
        conv3 = self.pool3(x)
        x = self.concat([conv1, conv2 , conv3])
        x = self.dense(x)
        return self.output_layer(x)

class CNNAttentionModel(Model):
    def __init__(self, input_shape, vocab_size, embedding_dim):
        super(CNNAttentionModel, self).__init__()
        self.embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_shape=input_shape)
        self.attention = MultiHeadAttention(num_heads=4, key_dim=32)
        self.conv1 = Conv1D(filters=32, kernel_size=10, activation='relu')
        self.pool1 = GlobalMaxPooling1D()
        self.conv2 = Conv1D(filters=32, kernel_size=5, activation='relu')
        self.pool2 = GlobalMaxPooling1D()
        self.conv3 = Conv1D(filters=32, kernel_size=3, activation='relu')
        self.pool3 = GlobalMaxPooling1D()
        self.concat = tf.keras.layers.Concatenate()
        self.dense = Dense(64, activation='relu')
        self.output_layer = Dense(1, activation='sigmoid')
    
    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.attention(query=x, key=x, value=x)
        conv1 = self.conv1(x)
        conv1 = self.pool1(conv1)
        conv2 = self.conv2(x)
        conv2 = self.pool2(conv2)
        conv3 = self.conv3(x)
        conv3 = self.pool3(conv3)
        x = self.concat([conv1, conv2 , conv3])
        x = self.dense(x)
        return self.output_layer(x)
    

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.ffn = tf.keras.Sequential([
            Dense(dff, activation='relu'),
            Dense(d_model)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.mha(inputs, inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2


class TransformerModel(Model):
    def __init__(self, input_shape, vocab_size, embedding_dim):
        super(TransformerModel, self).__init__()
        self.embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_shape=input_shape)
        self.transformer_block1 = TransformerBlock(d_model=128, num_heads=8, dff=2048, rate=0.001)
        self.transformer_block2 = TransformerBlock(d_model=128, num_heads=8, dff=2048, rate=0.001)
        self.maxpool = GlobalMaxPooling1D()
        self.dense = Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.transformer_block1(x)
        x = self.transformer_block2(x)
        x = self.maxpool(x)
        x = self.dense(x)
        return x
    

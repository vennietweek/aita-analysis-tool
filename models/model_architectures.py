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
    

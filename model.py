#import tensorflow as tf
from tensorflow.keras.layers import SimpleRNN, Embedding, Dense, LSTM
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn import metrics
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def model_main(x_train, y_train, cnt, epoch, batchsize, x_test, y_test):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=5,
                     strides=1, padding="causal",
                     activation="relu",
                     input_shape=[None, 1]))
    # model.add(Embedding(3, 32)) # embedding vector 32 levels
    model.add(LSTM(64, input_shape=(cnt, 3), return_sequences=True))  # RNN cell hidden_size 32, SimpleRNN
    model.add(LSTM(64, input_shape=(cnt, 3), return_sequences=True))  # RNN cell hidden_size 32, SimpleRNN
    model.add(Dense(30, activation="relu"))
    model.add(Dense(3, activation='relu'))  # if classify->sigmoid

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10 ** (epoch / 20))

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
    mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

    # optimizer rmsprop
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['acc', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])

    history = model.fit(x_train, y_train, epochs=epoch, batch_size=batchsize, validation_split=0.2,
                        callbacks=[lr_schedule, es, mc])

    loaded_model = load_model('best_model.h5')
    _loss, _acc, _precision, _recall = loaded_model.evaluate(x_test, y_test)
    print("\n test accuracy: %.4f" % _acc)
    print("\n test precision: %.4f" % _precision)
    print("\n test recall: %.4f" % _recall)

    epochs = range(1, len(history.history['acc']) + 1)
    plt.plot(epochs, history.history['loss'])
    plt.plot(epochs, history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


def LSTM_RNN(_X, _weights, _biases, n_input, n_steps, n_hidden):
    # input shape: (batch_size, n_steps, n_input)
    _X = tf.transpose(_X, [1, 0, 2])
    # Reshape to prepare input to hidden activation
    _X = tf.reshape(_X, [-1, n_input])
    # new shape: (n_steps*batch_size, n_input)

    _X = tf.nn.relu(tf.matmul(_X, _weights['hidden']) + _biases['hidden'])
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(_X, n_steps, 0)
    # new shape: n_steps * (batch_size, n_hidden)

    # LSTM cells
    lstm_cell_1 = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cells = tf.compat.v1.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
    # Get LSTM cell output
    outputs, states = tf.compat.v1.nn.static_rnn(lstm_cells, _X, dtype=tf.float32)
    # many-to-one
    lstm_last_output = outputs[-1]

    # Linear activation
    return tf.matmul(lstm_last_output, _weights['out']) + _biases['out']

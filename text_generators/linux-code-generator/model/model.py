from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dropout, Dense, TimeDistributed 

def make_text_generator_model(batch_size, vocab_size, embedding_size=64, rnn_size=128, num_layers=2):
    # Conversely if your data is large (more than about 2MB), feel confident to increase rnn_size and train a bigger model (see details of training below).
    # It will work significantly better. For example with 6MB you can easily go up to rnn_size 300 or even more.
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_size, batch_input_shape=(batch_size, None)))
    for layer in range(num_layers):
        model.add(LSTM(rnn_size, stateful=True, return_sequences=True))
        model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model



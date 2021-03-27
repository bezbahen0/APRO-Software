import numpy as np
from utils import Vectorizer

def _batch_sort_for_stateful_rnn(sequences, batch_size):
    # Now the tricky part, we need to reformat our data so the first
    # sequence in the nth batch picks up exactly where the first sequence
    # in the (n - 1)th batch left off, as the RNN cell state will not be
    # reset between batches in the stateful model.
    num_batches = sequences.shape[0] // batch_size
    num_samples = num_batches * batch_size
    reshuffled = np.zeros((num_samples, sequences.shape[1]), dtype=np.int32)
    for batch_index in range(batch_size):
        # Take a slice of num_batches consecutive samples
        slice_start = batch_index * num_batches
        slice_end = slice_start + num_batches
        index_slice = sequences[slice_start:slice_end, :]
        # Spread it across each of our batches in the same index position
        reshuffled[batch_index::batch_size, :] = index_slice
    return reshuffled


def _create_sequences(vector, seq_length, seq_step):
    # Take strips of our vector at seq_step intervals up to our seq_length
    # and cut those strips into seq_length sequences
    passes = []
    for offset in range(0, seq_length, seq_step):
        pass_samples = vector[offset:]
        num_pass_samples = pass_samples.size // seq_length
        pass_samples = np.resize(pass_samples,
                                 (num_pass_samples, seq_length))
        passes.append(pass_samples)
    # Stack our sequences together. This will technically leave a few "breaks"
    # in our sequence chain where we've looped over are entire dataset and
    # return to the start, but with large datasets this should be neglegable
    return np.concatenate(passes)


def  shape_for_stateful_rnn(data, batch_size, seq_length, seq_step):
    """
    Reformat our data vector into input and target sequences to feed into our RNN. Tricky with stateful RNNs.
    """
    # Our target sequences are simply one timestep ahead of our input sequences.
    # e.g. with an input vector "wherefore"...
    # targets:   h e r e f o r e
    # predicts   ^ ^ ^ ^ ^ ^ ^ ^
    # inputs:    w h e r e f o r
    inputs = data[:-1]
    targets = data[1:]

    # We split our long vectors into semi-redundant seq_length sequences
    inputs = _create_sequences(inputs, seq_length, seq_step)
    targets = _create_sequences(targets, seq_length, seq_step)

    # Make sure our sequences line up across batches for stateful RNNs
    inputs = _batch_sort_for_stateful_rnn(inputs, batch_size)
    targets = _batch_sort_for_stateful_rnn(targets, batch_size)

    # Our target data needs an extra axis to work with the sparse categorical
    # crossentropy loss function
    targets = targets[:, :, np.newaxis]
    return inputs, targets



def load_data(data_file, word_tokens, pristine_input, pristine_output, batch_size, seq_length=50, seq_step=25):
    try:
        with open(data_file, encoding='utf-8') as input_file:
            all_text = input_file.read()
    except FileNotFoundError:
        print("No input.txt in data_dir")
        sys.exit(1)

    # try:
    #     with open(os.path.join(data_dir, 'validate.txt'), encoding='utf-8') as validate_file:
    #         text_val = validate_file.read()
    #         skip_validate = False
    # except FileNotFoundError:
    #     pass  # Validation text optional

    # Find some good default seed string in our source text.
    # self.seeds = find_random_seeds(text)
    # Include our validation texts with our vectorizer
    vectorizer = Vectorizer.Vectorizer(all_text, word_tokens, pristine_input, pristine_output)
    data = vectorizer.vectorize(all_text)
    x, y = shape_for_stateful_rnn(data, batch_size, seq_length, seq_step)
    print("Word_tokens:", word_tokens)
    print('x.shape:', x.shape)
    print('y.shape:', y.shape)

    return x, y, vectorizer

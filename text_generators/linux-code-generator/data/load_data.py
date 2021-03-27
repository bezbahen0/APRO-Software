from utils import Vectorizer

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
    vectorizer = Vectorizer(all_text, word_tokens, pristine_input, pristine_output)
    data = vectorizer.vectorize(text)
    x, y = shape_for_stateful_rnn(data, batch_size, seq_length, seq_step)
    print("Word_tokens:", word_tokens)
    print('x.shape:', x.shape)
    print('y.shape:', y.shape)

    return x, y, vectorizer

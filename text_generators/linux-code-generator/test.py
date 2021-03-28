import argparse
import numpy as np
from model import model
from data import load_data

def sample_preds(preds, temperature=1.0):
    """
    Samples an unnormalized array of probabilities. Use temperature to
    flatten/amplify the probabilities.
    """
    preds = np.asarray(preds).astype(np.float64)
    # Add a tiny positive number to avoid invalid log(0)
    preds += np.finfo(np.float64).tiny
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate(model, vectorizer, seed, length=100, diversity=0.5):
    seed_vector = vectorizer.vectorize(seed)

    # Feed in seed string
    print("Seed:", seed, end=' ' if vectorizer.word_tokens else '')
    model.reset_states()
    preds = None
    for char_index in np.nditer(seed_vector):
        preds = model.predict(np.array([[char_index]]), verbose=0)

    sampled_indices = []  # np.array([], dtype=np.int32)
    # Sample the model one token at a time
    for i in range(length):
        char_index = 0
        if preds is not None:
            char_index = sample_preds(preds[0][0], diversity)
        sampled_indices.append(char_index)  # = np.append(sampled_indices, char_index)
        preds = model.predict(np.array([[char_index]]), verbose=0)
    sample = vectorizer.unvectorize(sampled_indices)
    return sample

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--text", action="store", required=False, dest="text", help="Input text file")
    parser.add_argument("-w", "--weights", action="store", required=False, dest="weights", help="Model weights path")
    parser.add_argument("-i", "--input", action="store", required=False, dest="input", help="Input string for complete")
    parser.add_argument("-o", "--out_len", action="store", required=False, dest="out_len", help="Out length")
    args = parser.parse_args()
    
    _, _, vectorizer = load_data.load_data(args.text, False, False, False, 1)
    model = model.make_text_generator_model(1, vectorizer.vocab_size)
    model.load_weights(args.weights)
    
    res = generate(model, vectorizer, seed=args.input, length=int(args.out_len))
    print(res)
    

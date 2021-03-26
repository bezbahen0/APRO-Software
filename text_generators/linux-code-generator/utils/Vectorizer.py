import re
import numpy as np

from collections import Counter

# Transforms text to vectors of integer numbers representing in text tokens and back. Handles word and character level tokenization.
class Vectorizer:

    def __init__(self, text, word_tokens, pristine_input, pristine_output):
        self.word_tokens = word_tokens
        self._pristine_input = pristine_input
        self._pristine_output = pristine_output

        tokens = self._tokenize(text)
        # print('corpus length:', len(tokens))
        token_counts = Counter(tokens)
        # Sort so most common tokens come first in our vocabulary
        tokens = [x[0] for x in token_counts.most_common()]
        self._token_indices = {x: i for i, x in enumerate(tokens)}
        self._indices_token = {i: x for i, x in enumerate(tokens)}
        self.vocab_size = len(tokens)
        print('Vocab size:', self.vocab_size)

    def _tokenize(self, text):
        if not self._pristine_input:
            text = text.lower()
        if self.word_tokens:
            if self._pristine_input:
                return text.split()
            return Vectorizer.word_tokenize(text)
        return text

    def _detokenize(self, tokens):
        if self.word_tokens:
            if self._pristine_output:
                return ' '.join(tokens)
            return Vectorizer.word_detokenize(tokens)
        return ''.join(tokens)

    def vectorize(self, text):
        """Transforms text to a vector of integers"""
        tokens = self._tokenize(text)
        indices = []
        for token in tokens:
            if token in self._token_indices:
                indices.append(self._token_indices[token])
            else:
                print('Ignoring unrecognized token:', token)
        return np.array(indices, dtype=np.int32)

    def unvectorize(self, vector):
        """Transforms a vector of integers back to text"""
        tokens = [self._indices_token[index] for index in vector]
        return self._detokenize(tokens)

    @staticmethod
    def word_detokenize(tokens):
        # A heuristic attempt to undo the Penn Treebank tokenization above. Pass the
        # --pristine-output flag if no attempt at detokenizing is desired.
        regexes = [
            # Newlines
            (re.compile(r'[ ]?\\n[ ]?'), r'\n'),
            # Contractions
            (re.compile(r"\b(can)\s(not)\b"), r'\1\2'),
            (re.compile(r"\b(d)\s('ye)\b"), r'\1\2'),
            (re.compile(r"\b(gim)\s(me)\b"), r'\1\2'),
            (re.compile(r"\b(gon)\s(na)\b"), r'\1\2'),
            (re.compile(r"\b(got)\s(ta)\b"), r'\1\2'),
            (re.compile(r"\b(lem)\s(me)\b"), r'\1\2'),
            (re.compile(r"\b(mor)\s('n)\b"), r'\1\2'),
            (re.compile(r"\b(wan)\s(na)\b"), r'\1\2'),
            # Ending quotes
            (re.compile(r"([^' ]) ('ll|'re|'ve|n't)\b"), r"\1\2"),
            (re.compile(r"([^' ]) ('s|'m|'d)\b"), r"\1\2"),
            (re.compile(r'[ ]?”'), r'"'),
            # Double dashes
            (re.compile(r'[ ]?--[ ]?'), r'--'),
            # Parens and brackets
            (re.compile(r'([\[\(\{\<]) '), r'\1'),
            (re.compile(r' ([\]\)\}\>])'), r'\1'),
            (re.compile(r'([\]\)\}\>]) ([:;,.])'), r'\1\2'),
            # Punctuation
            (re.compile(r"([^']) ' "), r"\1' "),
            (re.compile(r' ([?!\.])'), r'\1'),
            (re.compile(r'([^\.])\s(\.)([\]\)}>"\']*)\s*$'), r'\1\2\3'),
            (re.compile(r'([#$]) '), r'\1'),
            (re.compile(r' ([;%:,])'), r'\1'),
            # Starting quotes
            (re.compile(r'(“)[ ]?'), r'"')
        ]

        text = ' '.join(tokens)
        for regexp, substitution in regexes:
            text = regexp.sub(substitution, text)
        return text.strip()

    @staticmethod
    def word_tokenize(text):
        # Basic word tokenizer based on the Penn Treebank tokenization script, but
        # setup to handle multiple sentences. Newline aware, i.e. newlines are
        # replaced with a specific token. You may want to consider using a more robust
        # tokenizer as a preprocessing step, and using the --pristine-input flag.
        regexes = [
            # Starting quotes
            (re.compile(r'(\s)"'), r'\1 “ '),
            (re.compile(r'([ (\[{<])"'), r'\1 “ '),
            # Punctuation
            (re.compile(r'([:,])([^\d])'), r' \1 \2'),
            (re.compile(r'([:,])$'), r' \1 '),
            (re.compile(r'\.\.\.'), r' ... '),
            (re.compile(r'([;@#$%&])'), r' \1 '),
            (re.compile(r'([?!\.])'), r' \1 '),
            (re.compile(r"([^'])' "), r"\1 ' "),
            # Parens and brackets
            (re.compile(r'([\]\[\(\)\{\}\<\>])'), r' \1 '),
            # Double dashes
            (re.compile(r'--'), r' -- '),
            # Ending quotes
            (re.compile(r'"'), r' ” '),
            (re.compile(r"([^' ])('s|'m|'d) "), r"\1 \2 "),
            (re.compile(r"([^' ])('ll|'re|'ve|n't) "), r"\1 \2 "),
            # Contractions
            (re.compile(r"\b(can)(not)\b"), r' \1 \2 '),
            (re.compile(r"\b(d)('ye)\b"), r' \1 \2 '),
            (re.compile(r"\b(gim)(me)\b"), r' \1 \2 '),
            (re.compile(r"\b(gon)(na)\b"), r' \1 \2 '),
            (re.compile(r"\b(got)(ta)\b"), r' \1 \2 '),
            (re.compile(r"\b(lem)(me)\b"), r' \1 \2 '),
            (re.compile(r"\b(mor)('n)\b"), r' \1 \2 '),
            (re.compile(r"\b(wan)(na)\b"), r' \1 \2 '),
            # Newlines
            (re.compile(r'\n'), r' \\n ')
        ]

        text = " " + text + " "
        for regexp, substitution in regexes:
            text = regexp.sub(substitution, text)
        return text.split()



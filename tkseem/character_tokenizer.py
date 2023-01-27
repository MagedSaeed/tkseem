import pickle
import re
from collections import defaultdict

from ._base import BaseTokenizer


class CharacterTokenizer(BaseTokenizer):
    """Character based tokenization"""

    def train(self, file_path):
        """Train data using characters

        Args:
            file_path (str): file to train
        """
        print("Training CharacterTokenizer ...")
        rx = re.compile(r"\B(.)")

        text = open(file_path, "r").read()
        text = rx.sub(r" ##\1", text)

        tokens_frequency = defaultdict(int)
        for word in text.split(" "):
            tokens_frequency[word] += 1

        self.vocab = self._truncate_dict(dict(tokens_frequency))
        self.vocab_size = len(self.vocab)

    def tokenize(self, text):
        """Tokenize using the frequency dictionary

        Args:
            text (str): input string

        Returns:
            list: generated tokens
        """
        rx = re.compile(r"\B(.)")
        text = rx.sub(r" ##\1", text)
        output_tokens = []

        for token in text.split():
            if token in self.vocab:
                output_tokens.append(token)
            else:
                output_tokens.append(self.unk_token)
        return output_tokens

    def tokenize_from_splits(self, text):
        """Tokenize with basic tokenization
        That is, tokenize the text then select pieces that are in the vocab. Do not optimize on the best splits like in self.tokenize() method
        in the case of Characters tokenization, they are the same
        Args:
            text (str): input string
        Returns:
            list: generated tokens
        """
        text = self.segmenter.segment(text).replace("+", " ##")
        output_tokens = []

        for token in text.split():
            if token in self.vocab:
                output_tokens.append(token)
            else:
                output_tokens.append(self.unk_token)
        return output_tokens

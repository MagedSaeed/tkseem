from functools import lru_cache
import pickle
import re
from collections import defaultdict

from ._base import BaseTokenizer


class DisjointLetterTokenizer(BaseTokenizer):
    """Disjoint Letters based tokenization"""

    def train(self, file_path):
        """Train data using disjoint letters

        Args:
            file_path (str): file to train
        """
        print("Training DisjointLetterTokenizer ...")
        rx = re.compile(r"([اأإآءؤﻵﻹﻷدذرزو])")

        text = open(file_path, "r").read()
        text = rx.sub(r"\1## ", text)
        text = text.replace("## ", " ##")

        tokens_frequency = defaultdict(int)
        for word in text.split(" "):
            tokens_frequency[word] += 1

        self.vocab = self._truncate_dict(dict(tokens_frequency))
        self.vocab_size = len(self.vocab)

    def tokenize_from_splits(self, text):
        """Tokenize with basic tokenization
        That is, tokenize the text then select pieces that are in the vocab. Do not optimize on the best splits like in self.tokenize() method
        Args:
            text (str): input string
        Returns:
            list: generated tokens
        """

        output_tokens = []

        for token in self.split_text(text):
            if token in self.vocab:
                output_tokens.append(token)
            else:
                output_tokens.append(self.unk_token)
        return output_tokens

    @lru_cache(maxsize=10_000)
    def split_text(self, text):
        rx = re.compile(r"([اأإآءؤﻵﻹﻷدذرزو])")
        text = rx.sub(r"\1## ", text)
        text = text.replace("## ", " ##")
        tokens = text.split()
        tokens = list(filter(lambda token: token != "##", tokens))
        return tokens

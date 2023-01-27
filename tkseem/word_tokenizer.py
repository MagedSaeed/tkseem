from functools import lru_cache
from ._base import BaseTokenizer


class WordTokenizer(BaseTokenizer):
    """
    White space based tokenization
    """

    tokens_frequency = None

    def train(self, file_path):
        """Train using words' frequency

        Args:
            file_path (str): file to train
        """

        print("Training WordTokenizer ...")
        self.vocab = self._truncate_dict(self._get_tokens_frequency(file_path))
        self.vocab_size = len(self.vocab)

    def tokenize(self, text):
        """Tokenize using the frequency dictionary

        Args:
            text (str): input string

        Returns:
            list: generated tokens
        """
        assert self.vocab

        output_tokens = []
        for word in self.split_text(text):
            if word in self.vocab.keys():
                output_tokens.append(word)
            else:
                output_tokens.append(self.unk_token)
        return output_tokens

    def tokenize_from_splits(self, text):
        """Tokenize with basic tokenization
        That is, tokenize the text then select pieces that are in the vocab. Do not optimize on the best splits like in self.tokenize() method
        In the case of Words tokenization, they are the same
        Args:
            text (str): input string
        Returns:
            list: generated tokens
        """
        return self.tokenize(text)

    def detokenize(self, tokens):
        """Convert tokens to a string

        Args:
            tokens (list): list of tokens

        Returns:
            str: detokenized string
        """
        detokenized = " ".join(tokens)
        return detokenized

    @classmethod
    @lru_cache(maxsize=10_000)
    def split_text(cls, text):
        return text.split()

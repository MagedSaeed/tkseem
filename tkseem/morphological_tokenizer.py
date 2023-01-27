import os
import pickle

from ._base import BaseTokenizer


class MorphologicalTokenizer(BaseTokenizer):
    """Auto tokenization using a saved dictionary"""

    def train(self):
        """Use a default dictionary for training"""
        print("Training MorphologicalTokenizer ...")
        vocab_path = os.path.join(self.rel_path, "dictionaries/vocab.pl")
        self.vocab = self._truncate_dict(pickle.load(open(vocab_path, "rb")))

    def tokenize_from_splits(self, text):
        """Tokenize with basic tokenization
        That is, tokenize the text then select pieces that are in the vocab. Do not optimize on the best splits like in self.tokenize() method
        In the case of morphological tokenizer, this type of tokenization is not available at the moment as it is costly to tokenize from madamira
        Args:
            text (str): input string
        Returns:
            list: generated tokens
        """
        raise ValueError(
            "This type of tokenization is not implemented for Morphological tokenizer"
        )

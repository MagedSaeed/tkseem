from collections import defaultdict
from functools import lru_cache

from farasa.segmenter import FarasaSegmenter

from ._base import BaseTokenizer


class FarasaMorphologicalTokenizer(BaseTokenizer):
    """tokenize text based on farasa segmentation"""

    def __init__(self, interactive_segmentation=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.segmenter = FarasaSegmenter(interactive=interactive_segmentation)

    def train(self, file_path):
        """Train data using farasa

        Args:
            file_path (str): file to train
        """

        print("Training FarasaMorphologicalTokenizer...")

        with open(file_path, "r") as f:
            text = f.read()

        segmented_lines = list(
            map(
                self.split_text,
                (line for line in text.splitlines()),
            )
        )

        tokens_frequency = defaultdict(int)
        for split_line in segmented_lines:
            for token in split_line:
                tokens_frequency[token] += 1

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

    @classmethod
    @lru_cache(maxsize=10_000)
    def split_text(cls, text, interactive_segmentation=True):
        segmenter = FarasaSegmenter(interactive=interactive_segmentation)
        text = segmenter.segment(text).replace("+", " ##")
        return text.split()

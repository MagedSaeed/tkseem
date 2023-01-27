from collections import defaultdict

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
                self.segmenter.segment,
                (line for line in text.splitlines()),
            )
        )

        tokens_frequency = defaultdict(int)
        for line in segmented_lines:
            line = line.replace("+", " ##")
            for word in line.split(" "):
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
        text = self.segmenter.segment(text).replace("+", " ##")
        output_tokens = []

        for token in text.split():
            if token in self.vocab:
                output_tokens.append(token)
            else:
                output_tokens.append(self.unk_token)
        return output_tokens

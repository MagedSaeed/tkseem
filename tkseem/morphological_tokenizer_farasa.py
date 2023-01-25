from collections import defaultdict

from farasa.segmenter import FarasaSegmenter

from ._base import BaseTokenizer


class FarasaMorphologicalTokenizer(BaseTokenizer):
    """tokenize text based on farasa segmentation"""

    def train(self, file_path):
        """Train data using farasa

        Args:
            file_path (str): file to train
        """

        print("Training FarasaMorphologicalTokenizer...")

        segmenter = FarasaSegmenter(interactive=False)

        with open(file_path, "r") as f:
            text = f.read()

        segmented = segmenter.segment(text)

        tokens_frequency = defaultdict(int)
        for line in segmented.splitlines():
            for word in line.split(" "):
                tokens_frequency[word] += 1

        self.vocab = self._truncate_dict(dict(tokens_frequency))
        self.vocab_size = len(self.vocab)

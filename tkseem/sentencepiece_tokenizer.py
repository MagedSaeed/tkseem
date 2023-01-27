import io

import sentencepiece as spm

from ._base import BaseTokenizer


class SentencePieceTokenizer(BaseTokenizer):
    """Sentencepiece based tokenization."""

    def train(self, file_path, model_type="bpe"):
        """Train using sentence piece

        Args:
            file_path (str): file to train
            model_type (str, optional): train using sp. Defaults to "bpe".
        """
        print("Training SentencePiece ...")
        self.model = io.BytesIO()

        spm.SentencePieceTrainer.train(
            input=file_path,
            model_writer=self.model,
            vocab_size=self.vocab_size,
            model_type=model_type,
            character_coverage=1.0,
            unk_id=0,
            pad_id=1,
            bos_id=-1,
            eos_id=-1,
            user_defined_symbols=self.special_tokens,
            normalization_rule_name="identity",
        )
        self.save_model("m.model")
        self.sp = spm.SentencePieceProcessor(model_file="m.model")
        self.vocab_size = self.sp.vocab_size()

    def tokenize(self, text):
        """Tokenize using the frequency dictionary

        Args:
            text (str): input string

        Returns:
            list: generated tokens
        """
        return self.sp.encode(text, out_type=str)

    def tokenize_from_splits(self, text):
        """Tokenize with basic tokenization
        That is, tokenize the text then select pieces that are in the vocab. Do not optimize on the best splits like in self.tokenize() method
        In the case of SentencePiece tokenization, it is not available
        Args:
            text (str): input string
        Returns:
            list: generated tokens
        """
        raise ValueError(
            "This type of tokenization is not implemented for SentencePiece tokenizer"
        )

    def load_model(self, file_path):
        """Load a saved sp model

        Args:
            file_path (str): file path of the trained model
        """
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(file_path)

    def save_model(self, file_path):
        """Save a model as a freqency dictionary

        Args:
            file_path (str): file path to save the model
        """
        with open(file_path, "wb") as f:
            f.write(self.model.getvalue())

    def id_to_token(self, id):
        return self.sp.id_to_piece(int(id))

    def token_to_id(self, token):
        return self.sp.piece_to_id(token)

    def encode(self, text):
        """Convert string to a list of ids

        Args:
            text (str): input string

        Returns:
            list: list of ids
        """
        return self.sp.encode(text, out_type=int)

    def decode(self, encoded):
        """Decode ids

        Args:
            encoded (list): list of ids to decode

        Returns:
            list: tokens
        """
        return self.sp.id_to_piece(encoded)

    def detokenize(self, tokens):
        """Convert tokens to a string

        Args:
            tokens (list): list of tokens

        Returns:
            str: detokenized string
        """
        return "".join(tokens).replace("▁", " ")

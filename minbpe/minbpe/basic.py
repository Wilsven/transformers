"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

But:
- Does not handle the regular expression splitting pattern.
- Does not handle any special tokens.
"""

from .base import Tokenizer, get_stats, merge


class BasicTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()

    def train(self, text: str, vocab_size: int, verbose: bool = False):
        """
        Train a vocabulary of size vocab_size from the input text.

        Args:
            text (str): The input text to train the vocabulary from.
            vocab_size (int): The size of the final vocabulary size.
            verbose (bool, optional): Flag indicating whether to display verbose output. Defaults to False.
        """
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # input text preprocessing
        text_bytes = text.encode("utf-8")  # raw bytes
        ids = list(text_bytes)  # list of integers in range 0..255

        # iteratively merge the most common pairs to create new tokens
        merges = {}  # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)}  # int -> bytes
        for i in range(num_merges):
            # count up the number of times every consecutive pair appears
            stats = get_stats(ids)
            # find the pair with the highest count
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = merge(ids, pair, idx)
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # prints
            if verbose:
                print(
                    f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences"
                )

        # save class variables
        self.merges = merges  # used in encode()
        self.vocab = vocab  # used in decode()

    def encode(self, text: str) -> list[int]:
        """
        A function to encode the given text using a tokenizer.

        Args:
            text (str): The input text to be encoded.

        Returns:
            list[int]: The encoded list of integers.
        """
        # given a string text, return the token ids
        text_bytes = text.encode("utf-8")  # raw bytes
        ids = list(text_bytes)  # list of integers in range 0..255

        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                break  # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def decode(self, ids: list[int]) -> str:
        """
        A function to decode the given list of integers into a string using a tokenizer.

        Args:
            ids (list[int]): The list of integers to be decoded.

        Returns:
            str: The decoded text.
        """
        # given ids (list of integers), return Python string
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

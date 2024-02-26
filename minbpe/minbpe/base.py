"""
Contains the base Tokenizer class and a few common helper functions.
The base class also contains the (common) save/load functionality.
It would be possible to be a lot more strict about the interface and
e.g. isolating all regex/pattern parts to the RegexTokenizer, but
some concessions are made for simplicity.
"""

from typing import Optional
import unicodedata

# -----------------------------------------------------------------------------
# a few helper functions useful for both BasicTokenizer and RegexTokenizer


def get_stats(ids: list[int], counts: Optional[dict] = None) -> dict:
    """Given a list of integers, return a dictionary of counts of consecutive pairs.

    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    Optionally allows to update an existing dictionary of counts

    Args:
        ids (list[int]): List of integers
        counts (Optional[dict], optional): Dictionary of counts of consecutive pairs. Defaults to None.

    Returns:
        dict: Dictionary of counts of consecutive pairs
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):  # iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids: list[int], pair: tuple[int], idx: int) -> list[int]:
    """
    In the list of integers (ids), replace all consecutive occurrences
    of pair with the new integer token idx

    Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]

    Args:
        ids (list[int]): The original list of ids.
        pair (tuple[int]): Pair to replace in list of ids.
        idx (int): New integer token to replace pair.

    Returns:
        list[int]: The modified list of ids after merging the pair to the new token.
    """
    newids = []
    i = 0
    while i < len(ids):
        if (i < len(ids) - 1) and (ids[i] == pair[0] and ids[i + 1] == pair[1]):
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


# first two helper functions...
def replace_control_characters(s: str) -> str:
    """
    A function to replace control characters in a string with escape sequences.

    Args:
        s (str): The input string.

    Returns:
        str: The modified string with control characters replaced.
    """
    # we don't want to print control characters
    # which distort the output (e.g. \n or much worse)
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
    # http://www.unicode.org/reports/tr44/#GC_Values_Table
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch)  # this character is ok
        else:
            chars.append(f"\\u{ord(ch):04x}")  # escape
    return "".join(chars)


def render_token(t: bytes) -> str:
    """
    A function that pretty prints a token, escaping control characters.

    Args:
        t (bytes): The byte tokens to be pretty printed.

    Returns:
        str: The pretty printed token with control characters escaped.
    """
    # pretty print a token, escaping control characters
    s = t.decode("utf-8", errors="replace")
    s = replace_control_characters(s)
    return s


# -----------------------------------------------------------------------------
# the base Tokenizer class


class Tokenizer:
    """Base class for Tokenizers"""

    def __init__(self):
        # default: vocab size of 256 (all bytes), no merges, no patterns
        self.merges = {}  # (int, int) -> int
        self.pattern = ""  # str
        self.special_tokens = {}  # str -> int, e.g. {'<|endoftext|>': 100257}
        self.vocab = self._build_vocab()  # int -> bytes

    def train(self, text: str, vocab_size: int, verbose: bool = False):
        """
        Train a vocabulary of size vocab_size from the input text.

        Args:
            text (str): The input text to train the vocabulary from.
            vocab_size (int): The size of the final vocabulary size.
            verbose (bool, optional): Flag indicating whether to display verbose output. Defaults to False.

        Raises:
            NotImplementedError: Raises if method is not implemented.
        """
        # Tokenizer can train a vocabulary of size vocab_size from text
        raise NotImplementedError

    def encode(self, text: str):
        """
        A function to encode the given text using a tokenizer.

        Args:
            text (str): The input text to be encoded.

        Raises:
            NotImplementedError: Raises if method is not implemented.
        """
        # Tokenizer can encode a string into a list of integers
        raise NotImplementedError

    def decode(self, ids: list[int]):
        """
        A function to decode the given list of integers into a string using a tokenizer.

        Args:
            ids (list[int]): The list of integers to be decoded.

        Raises:
            NotImplementedError: Raises if method is not implemented.
        """
        # Tokenizer can decode a list of integers into a string
        raise NotImplementedError

    def _build_vocab(self) -> dict:
        """
        Build the vocabulary dictionary based on merges and special tokens.

        Returns:
            dict: The built vocabulary dictionary.
        """
        # vocab is simply and deterministically derived from merges
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def save(self, file_prefix: str):
        """Save the model to files.

        Saves two files: file_prefix.vocab and file_prefix.model
        This is inspired (but not equivalent to!) sentencepiece's model saving:
            - model file is the critical one, intended for load()
            - vocab file is just a pretty printed version for human inspection only

        Args:
            file_prefix (str): The prefix for the file names.
        """
        # write the model: to be used in load() later
        model_file = file_prefix + ".model"
        with open(model_file, "w") as f:
            # write the version, pattern and merges, that's all that's needed
            f.write("minbpe v1.0\n")
            f.write(f"{self.pattern}\n")
            # write the special tokens, first the number of them, then each one
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            # the merges dict
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")
        # write the vocab: for the human to look at
        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                # note: many tokens may be partial utf-8 sequences
                # and cannot be decoded into valid strings. Here we're using
                # errors='replace' to replace them with the replacement char �.
                # this also means that we couldn't possibly use .vocab in load()
                # because decoding in this way is a lossy operation!
                s = render_token(token)
                # find the children of this token, if any
                if idx in inverted_merges:
                    # if this token has children, render it nicely as a merge
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    # otherwise this is leaf token, just print it
                    # (this should just be the first 256 tokens, the bytes)
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file: str):
        """Load a model from the specified file.

        Inverse of save() but only for the model file.

        Args:
            model_file (str): The file path of the model to be loaded.
        """
        assert model_file.endswith(".model")
        # read the model file
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, "r", encoding="utf-8") as f:
            # read the version
            version = f.readline().strip()
            assert version == "minbpe v1.0"
            # read the pattern
            self.pattern = f.readline().strip()
            # read the special tokens
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            # read the merges
            for line in f:
                idx0, idx1 = line.split()
                idx0 = int(idx0)
                idx1 = int(idx1)
                merges[(idx0, idx1)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()

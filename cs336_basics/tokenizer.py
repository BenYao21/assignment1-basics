from typing import Iterable, Iterator
import re



class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

    def from_files(self, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] = None):
        pass
    def encode(self, text: str) -> list[int]:
        pass
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pass
    def decode(self, ids: list[int]) -> str:
        pass
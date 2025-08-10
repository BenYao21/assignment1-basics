from typing import Iterable, Iterator
import regex as re
import json


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab2idx = {v: k for k, v in vocab.items()}

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] = None):
        with open(vocab_filepath, 'r') as f:
            vocab = json.load(f)
            vocab2idx = {v: k for k, v in vocab.items()}
        bpe_merges = []
        with open(merges_filepath, 'r') as f:
            for line in f:
                line_cleaned = line.rstrip()
                if line_cleaned and len(line_cleaned.split(" ")) == 2:
                    bpe_merges.append(tuple(line_cleaned.split(" ")))
        merges = [
            (
                bytes([vocab2idx[token] for token in merge_token_1]),
                bytes([vocab2idx[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in bpe_merges
        ]
        return cls(vocab, merges, special_tokens)
    def encode(self, text: str) -> list[int]:
        return [self.vocab2idx[text.encode("utf-8")]]
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield self.encode(text)
    def decode(self, ids: list[int]) -> str:
        return "".join([self.vocab[id].decode("utf-8") for id in ids])
from typing import Iterable, Iterator
import regex as re
import json

def get_chunks(text: str, max_length: int) -> list[str]:
    pass

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.special_token_bytes = [i.encode("utf-8") for i in special_tokens]
        self.vocab2idx = {v: k for k, v in vocab.items()}

        for st in self.special_token_bytes:
            self.vocab2idx[st] = len(self.vocab)
            self.vocab[len(self.vocab)] = st

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] = None):
        with open(vocab_filepath, 'r') as f:
            vocab_data = json.load(f)
            vocab = {int(k): bytes(v, "utf-8") for k, v in vocab_data.items()}
        bpe_merges = []
        with open(merges_filepath, 'r') as f:
            for line in f:
                line_cleaned = line.rstrip()
                if line_cleaned and len(line_cleaned.split(" ")) == 2:
                    bpe_merges.append(tuple(line_cleaned.split(" ")))
        merges = [
            (
                a.encode("utf-8"),
                b.encode("utf-8"),
            )
            for a, b in bpe_merges
        ]
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
    def encode(self, text: str) -> list[int]:
        chunks = get_chunks(text, 1000)
        tokens = []
        for chunk in chunks:
            if chunk in self.special_token_bytes:
                tokens.append(self.vocab2idx[chunk])
            else:
                # TODO: Implement this
                tokens.extend()
        return tokens
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield self.encode(text)
    def decode(self, ids: list[int]) -> str:
        return b"".join([self.vocab[id] for id in ids]).decode("utf-8", errors="replace")
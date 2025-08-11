from typing import Iterable, Iterator
import regex as re
import json

def get_chunks(text: str) -> list[str]:
    PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    return PAT.findall(text)

def apply_merges(word_bytes: bytes, merge_set: set, vocab_to_idx: dict[bytes, int]) -> tuple[bytes, ...]:
    word_bytes = list(word_bytes)
    while True:
        min_token_id = float('inf')
        best_pair_id = -1
        merged = None
        for i in range(len(word_bytes) - 1):
            pair = tuple(word_bytes[i:i+2])
            if pair in merge_set:
                combined = pair[0] + pair[1]
                token_id = vocab_to_idx[combined]
                if token_id is not None and token_id < min_token_id:
                    min_token_id = token_id
                    best_pair_id = i
                    merged = pair
        if best_pair_id == -1:
            break
        word_bytes = (word_bytes[:best_pair_id] + [vocab_to_idx[combined]] + word_bytes[best_pair_id+2:])
    return tuple(word_bytes)

def encode_merged(text: str, merges: list, vocab_to_idx: dict[bytes, int]) -> list[int]:
    word_list = get_chunks(text)
    tokens = []
    for word in word_list:
        word_bytes = word.encode("utf-8")
        word_bytes = apply_merges(word_bytes, merges, vocab_to_idx)
        tokens.extend(vocab_to_idx[i] for i in word_bytes)
    return tokens

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
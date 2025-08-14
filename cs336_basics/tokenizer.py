from typing import Iterable, Iterator
import regex as re
import json

def get_chunks(text: str) -> list[str]:
    PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    return PAT.findall(text)

def apply_merges(word_bytes: bytes, merges: set[tuple[bytes, bytes]], vocab_to_idx: dict[bytes, int]) -> tuple[bytes, ...]:
    # 1. 将 bytes 对象分割成由单个字节的 bytes 对象组成的列表
    #    例如：b"word" -> [b"w", b"o", b"r", b"d"]
    parts = [word_bytes[i:i+1] for i in range(len(word_bytes))]

    while True:
        if len(parts) < 2:
            break

        min_token_id = float('inf')
        best_pair_idx = -1
        best_combined_bytes = None

        for i in range(len(parts) - 1):
            # 2. 从 parts 列表中创建字节对
            pair = (parts[i], parts[i+1])
            if pair in merges:
                combined = pair[0] + pair[1]
                if combined in vocab_to_idx:
                    token_id = vocab_to_idx[combined]
                    if token_id < min_token_id:
                        min_token_id = token_id
                        best_pair_idx = i
                        best_combined_bytes = combined
        
        if best_pair_idx == -1:
            break
        
        # 3. 使用找到的最佳合并字节对来更新 parts 列表
        parts = (
            parts[:best_pair_idx]
            + [best_combined_bytes]
            + parts[best_pair_idx + 2:]
        )
    
    return tuple(parts)

def encode_merged(text: str, merges: set[tuple[bytes, bytes]], vocab_to_idx: dict[bytes, int]) -> list[int]:
    word_list = get_chunks(text)
    tokens = []
    for word in word_list:
        word_bytes = word.encode("utf-8")
        merged_word_bytes = apply_merges(word_bytes, merges, vocab_to_idx)
        tokens.extend(vocab_to_idx[i] for i in merged_word_bytes)
    return tokens

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens = None):
        self.vocab = vocab
        self.merges = set(merges)
        self.special_tokens = special_tokens if special_tokens else []
        self.special_token_bytes = [i.encode("utf-8") for i in self.special_tokens]
        self.vocab2idx = {v: k for k, v in vocab.items()}

        for st_bytes in self.special_token_bytes:
            if st_bytes not in self.vocab2idx:
                new_id = len(self.vocab)
                self.vocab[new_id] = st_bytes
                self.vocab2idx[st_bytes] = new_id

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
        chunks = get_chunks(text)
        tokens = []
        for chunk in chunks:
            chunk_bytes = chunk.encode("utf-8")
            if chunk_bytes in self.vocab2idx:
                tokens.append(self.vocab2idx[chunk_bytes])
            else:
                merged_bytes = apply_merges(chunk_bytes, self.merges, self.vocab2idx)
                tokens.extend(self.vocab2idx[b] for b in merged_bytes)
        return tokens
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield self.encode(text)
    def decode(self, ids: list[int]) -> str:
        return b"".join([self.vocab[id] for id in ids]).decode("utf-8", errors="replace")
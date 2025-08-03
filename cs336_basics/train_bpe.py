import regex as re
import collections

def train(data: str, pat_str: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    #ranks: bytes -> int
    ranks = {}
    #vocab: int -> bytes
    vocab = {}
    for i in range(2**8):
        ranks[bytes([i])] = i
        vocab[i] = bytes([i])
    #merges: list of tuples of bytes
    merges = []

    words: list[list[bytes]] = [
        [bytes([b]) for b in word.encode('utf-8')] for word in re.findall(pat_str, data)
    ]

    while len(ranks) < vocab_size - len(special_tokens):
        stats = collections.Counter()
        for piece in words:
            for pair in zip(piece[:-1],piece[1:]):
                stats[pair] += 1
        if len(stats) == 0:
            break
        most_common_pair = max(stats, key=lambda p: (stats[p], p))
        token_bytes = most_common_pair[0] + most_common_pair[1]
        token = len(ranks)
        #Add the new token
        ranks[token_bytes] = token
        vocab[token] = token_bytes
        merges.append(most_common_pair)
        new_words = []
        for word in words:
            new_word = []
            i = 0
            while i < len(word) - 1:
                if (word[i], word[i+1]) == most_common_pair:
                    new_word.append(token_bytes)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            if i == len(word) - 1:
                new_word.append(word[i])
            new_words.append(new_word)
        words = new_words
    for token in special_tokens:
        vocab[len(ranks)] = token.encode('utf-8')
        ranks[token.encode('utf-8')] = len(ranks)
    return vocab, merges

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    pattern = (
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
    with open(input_path) as f:
        data = f.read()

    vocab, merges = train(data, pattern, vocab_size, special_tokens)
    return vocab, merges
import regex as re
from collections import Counter

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    test_input = f'''
    low low low low low
    lower lower widest widest widest
    newest newest newest newest newest newest
    '''
    new_words = re.findall(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""", test_input)
    word_freqs = Counter()
    for word in new_words:
        word_freqs[word] += 1
    alphabet = []
    for w in word_freqs.keys():
        for c in w:
            if c not in alphabet:
                alphabet.append(c)
    alphabet.sort()

    vocabs = ["<|endoftext|>"] + alphabet.copy()
    splits = {word: [c for c in word] for word in word_freqs.keys()}
    def compute_pair_freqs(splits: dict[str, list[str]], word_freqs: Counter) -> dict[tuple[str, str], int]:
        pair_freqs = Counter()
        for word, freq in word_freqs.items():
            split = splits[word]
            if len(split) == 1:
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq
        return pair_freqs
    pair_freqs = compute_pair_freqs(splits, word_freqs)
    best_pair = pair_freqs.most_common(1)[0]
    def merge_pair(a, b, splits: dict[str, list[str]]):
        for word in splits.keys():
            split = splits[word]
            if len(split) == 1:
                continue
            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    split = split[:i] + [a + b] + split[i + 2:]
                    splits[word] = split
                else:
                    i += 1
            splits[word] = split
        return splits
    merges = {}
    while len(vocabs) < vocab_size:
        pair_freqs = compute_pair_freqs(splits, word_freqs)
        best_pair = ""
        max_freq = None
        for pair, freq in pair_freqs.items():
            if max_freq is None or freq > max_freq:
                best_pair = pair
                max_freq = freq
        splits = merge_pair(best_pair[0], best_pair[1], splits)
        merges[best_pair] = best_pair[0] + best_pair[1]
        vocabs.append(best_pair[0] + best_pair[1])
        print(best_pair)
    print(merges)


train_bpe("test.txt", 20, [], {})
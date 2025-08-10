import regex as re
from collections import defaultdict
from tqdm.contrib.concurrent import process_map

PAT = re.compile(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")

def word_to_byte_tuple(word: str) -> tuple[bytes, ...]:
    """Helper to convert a word string to a tuple of single-byte bytes objects."""
    return tuple(bytes([b]) for b in word.encode('utf-8'))

def count_words_in_chunk(data: str) -> defaultdict[tuple[bytes, ...], int]:
    """Counts frequency of words (represented as byte tuples) in a text chunk."""
    word_counts = defaultdict(int)
    for match in re.finditer(PAT, data):
        word_bytes_tuple = word_to_byte_tuple(match.group(0))
        # Now we don't filter by length, we count everything
        word_counts[word_bytes_tuple] += 1
    return word_counts

def count_pairs(word_counts: defaultdict) -> defaultdict[tuple[bytes, bytes], int]:
    pair_counts = defaultdict(int)
    for word_tuple, cnt in word_counts.items():
        if len(word_tuple) < 2:  # Correctly check for pairs
            continue
        for pair in zip(word_tuple[:-1], word_tuple[1:]):
            pair_counts[pair] += cnt
    return pair_counts

def apply_merge(word_tuple: tuple[bytes, ...], merge_pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    merged_token = merge_pair[0] + merge_pair[1]
    new_word_tuple = []
    i = 0
    while i < len(word_tuple):
        if i < len(word_tuple) - 1 and (word_tuple[i], word_tuple[i+1]) == merge_pair:
            new_word_tuple.append(merged_token)
            i += 2
        else:
            new_word_tuple.append(word_tuple[i])
            i += 1
    return tuple(new_word_tuple)

def update_counts(
    merge_pair: tuple[bytes, bytes], 
    word_counts: defaultdict, 
    pair_counts: defaultdict
) -> tuple[defaultdict, defaultdict]:
    new_word_counts = defaultdict(int)
    new_pair_counts = defaultdict(int, pair_counts)

    for word_tuple, cnt in word_counts.items():
        if len(word_tuple) < 2:
            new_word_counts[word_tuple] += cnt # Keep words that can't be merged
            continue
        
        old_pairs = list(zip(word_tuple[:-1], word_tuple[1:]))
        if merge_pair not in old_pairs:
            new_word_counts[word_tuple] += cnt
            continue
        
        new_word = apply_merge(word_tuple, merge_pair)
        new_word_counts[new_word] += cnt

        # Decrement old pair counts
        for pair in old_pairs:
            new_pair_counts[pair] -= cnt
            if new_pair_counts[pair] == 0:
                del new_pair_counts[pair]
        
        # Increment new pair counts
        new_pairs = list(zip(new_word[:-1], new_word[1:]))
        for p in new_pairs:
            new_pair_counts[p] += cnt

    return new_word_counts, new_pair_counts

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    with open(input_path, "r", encoding="utf-8") as f:
        data = f.read()

    # 1. Split text by special tokens
    chunks = [data]
    if special_tokens:
        special_tokens.sort(key=len, reverse=True)
        pattern_str = '|'.join([re.escape(token) for token in special_tokens])
        pattern = re.compile(pattern_str)
        chunks = pattern.split(data)
        chunks = [c for c in chunks if c]
    
    # 2. Parallel count words in chunks
    if len(chunks) < 4:
        word_counts_list = list(map(count_words_in_chunk, chunks))
    else:
        word_counts_list = process_map(count_words_in_chunk, chunks, chunksize=1)
    
    # 3. Merge word counts from all chunks
    #word_counts: defaultdict[tuple[bytes, ...], int]
    word_counts = defaultdict(int)
    for d in word_counts_list:
        for word, cnt in d.items():
            word_counts[word] += cnt

    # 4. Initial pair counts
    #pair_counts: defaultdict[tuple[bytes, bytes], int]
    pair_counts = count_pairs(word_counts)

    # 5. Initialize vocab and merges
    merges = []
    vocab = {i: bytes([i]) for i in range(256)}
    for i, token_str in enumerate(special_tokens):
        token_id = 256 + i
        vocab[token_id] = token_str.encode('utf-8')
    
    # 6. BPE training loop
    num_merges = vocab_size - len(vocab)
    for i in range(num_merges):
        if not pair_counts:
            break # No more pairs to merge
        # Correctly find the best pair
        most_common_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
        
        # Add new token to vocab
        new_token_bytes = most_common_pair[0] + most_common_pair[1]
        vocab[len(vocab)] = new_token_bytes
        merges.append(most_common_pair)

        # Update counts
        word_counts, pair_counts = update_counts(most_common_pair, word_counts, pair_counts)

    return vocab, merges
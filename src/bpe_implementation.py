from collections import Counter
from typing import List, Tuple, Dict


def prepare_corpus(corpus: List[str]) -> List[List[str]]:
    """
    Add end-of-word token '</w>' to each word in the corpus.

    Args:
        corpus (List[str]): List of words in the corpus.

    Returns:
        List[List[str]]: Corpus where each word is represented as a list of characters
                         with an added end-of-word token.
    """
    return [list(word) + ['</w>'] for word in corpus]


def get_pair_counts(corpus: List[List[str]]) -> Counter:
    """
    Count the frequency of symbol pairs in the corpus.

    Args:
        corpus (List[List[str]]): The prepared corpus with end-of-word tokens.

    Returns:
        Counter: A counter mapping symbol pairs to their frequency in the corpus.
    """
    pairs = Counter()
    for word in corpus:
        pairs.update(zip(word, word[1:]))
    return pairs


def merge_pair(corpus: List[List[str]], pair: Tuple[str, str]) -> List[List[str]]:
    """
    Merge the most frequent pair of symbols in the corpus.

    Args:
        corpus (List[List[str]]): The current corpus of tokenized words.
        pair (Tuple[str, str]): The most frequent pair of symbols to be merged.

    Returns:
        List[List[str]]: Updated corpus with the pair merged.
    """
    new_corpus = []
    for word in corpus:
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                new_word.append(pair[0] + pair[1])
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_corpus.append(new_word)
    return new_corpus


def byte_pair_encoding(corpus: List[str], num_merges: int) -> Tuple[List[List[str]], List[Tuple[str, str]]]:
    """
    Perform Byte Pair Encoding (BPE) on the given corpus for a specified number of merges.

    Args:
        corpus (List[str]): A list of words in the corpus.
        num_merges (int): The number of merge operations to perform.

    Returns:
        Tuple[List[List[str]], List[Tuple[str, str]]]:
            - Updated corpus after BPE merges.
            - List of merge operations performed.
    """
    corpus = prepare_corpus(corpus)
    merges = []

    for _ in range(num_merges):
        pair_counts = get_pair_counts(corpus)
        if not pair_counts:
            break
        best_pair = pair_counts.most_common(1)[0][0]
        merges.append(best_pair)
        corpus = merge_pair(corpus, best_pair)

    return corpus, merges


def build_bpe_vocab(merges: List[Tuple[str, str]]) -> Dict[str, int]:
    """
    Build a vocabulary from the learned BPE merge operations.

    Args:
        merges (List[Tuple[str, str]]): List of BPE merge operations.

    Returns:
        Dict[str, int]: A dictionary mapping merged tokens to unique indices.
    """
    return {''.join(pair): i for i, pair in enumerate(merges)}


# def encode_word(word: str, merges: List[Tuple[str, str]]) -> List[str]:
#     """
#     Encode a new word based on the learned BPE merges.

#     Args:
#         word (str): The word to encode.
#         merges (List[Tuple[str, str]]): List of BPE merge operations.

#     Returns:
#         List[str]: Encoded word with the BPE merge rules applied.
#     """
#     word = list(word) + ['</w>']
    
#     while len(word) > 1:
#         # Find all pairs in the current word
#         pairs = list(zip(word, word[1:]))
        
#         # Find the first (highest priority) merge operation that applies
#         for pair in merges:
#             if pair in pairs:
#                 i = pairs.index(pair)
#                 word[i:i+2] = [''.join(pair)]  # Merge the pair
#                 break
#         else:
#             # If no merge operation applies, we're done
#             break

#     return word

def encode_word(word: str, merges: List[Tuple[str, str]]) -> List[str]:
    """
    Encode a word using the learned BPE merges.

    Args:
        word (str): The word to encode.
        merges (List[Tuple[str, str]]): List of BPE merge operations.

    Returns:
        List[str]: Encoded word with the BPE merge rules applied.
    """
    word = list(word) + ['</w>']
    
    def get_pairs(word):
        return set(zip(word, word[1:]))

    while True:
        pairs = get_pairs(word)
        if not pairs:
            break

        best_pair = min(pairs, key=lambda pair: merges.index(pair) if pair in merges else float('inf'))
        
        if best_pair not in merges:
            break

        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and tuple(word[i:i+2]) == best_pair:
                new_word.append(''.join(best_pair))
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        word = new_word

    return word


def main():
    """
    Demonstrates the Byte Pair Encoding (BPE) process on a sample corpus.
    """
    corpus = ["low", "lower", "newest", "widest"]
    num_merges = 10

    # Perform BPE
    bpe_corpus, merges = byte_pair_encoding(corpus, num_merges)

    # Print the BPE tokenized corpus
    print("BPE tokenized corpus:")
    for word in bpe_corpus:
        print(" ".join(word))

    # Build BPE vocabulary
    vocab = build_bpe_vocab(merges)
    print("\nBPE Vocabulary:")
    for token, idx in vocab.items():
        print(f"{token}: {idx}")

    # Encode a new word
    new_word = "lowest"
    encoded = encode_word(new_word, merges)
    print(f"\nEncoded '{new_word}': {' '.join(encoded)}")


if __name__ == "__main__":
    main()
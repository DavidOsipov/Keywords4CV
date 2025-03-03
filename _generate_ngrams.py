def _generate_ngrams(self, tokens: List[str], n: int) -> Set[str]:
    """
    Generate n-grams from a list of tokens.

    Args:
        tokens: List of tokens to generate n-grams from
        n: Size of n-grams to generate

    Returns:
        Set[str]: Set of generated n-grams

    Raises:
        ValueError: If n is not a positive integer
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError(f"Invalid ngram size: {n}. Must be positive integer")

    filtered_tokens = [
        token
        for token in tokens
        if len(token.strip()) > 1 and token not in self.preprocessor.stop_words
    ]

    if len(filtered_tokens) < n:
        return set()

    # Simplified - removed redundant check since filtered_tokens already ensures len(t) > 1
    ngrams = {
        " ".join(filtered_tokens[i : i + n])
        for i in range(len(filtered_tokens) - (n - 1))
    }
    return ngrams

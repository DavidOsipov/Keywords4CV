import logging
from typing import Union, List, Set, Dict, Optional

import spacy
from spacy.tokens import Doc, Token

logger = logging.getLogger(__name__)


class SemanticValidator:
    """Unified validation helper for keyword extraction pipeline."""

    def __init__(self, config: Dict, nlp):
        """
        Initialize validator with configuration and NLP model.

        Args:
            config: Configuration dictionary
            nlp: Loaded spaCy model
        """
        self.config = config
        self.nlp = nlp

        # Extract config values with defaults for quick access
        self.semantic_validation = config.get("text_processing", {}).get(
            "semantic_validation", False
        )
        self.similarity_threshold = config.get("text_processing", {}).get(
            "similarity_threshold", 0.85
        )

        # Extract negative keywords with case normalization
        self.negative_keywords = set()
        for kw in config.get("advanced", {}).get("negative_keywords", []):
            self.negative_keywords.add(kw.lower())

        # Extract allowed POS tags
        self.allowed_pos = set(
            config.get("whitelist", {}).get("fuzzy_matching", {}).get("allowed_pos", [])
            or config.get("text_processing", {}).get(
                "pos_filter", ["NOUN", "PROPN", "ADJ"]
            )
        )

        # Cache results for performance
        self._validation_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def validate_term(
        self, term: Union[str, Doc, Token], context_doc: Optional[Doc] = None
    ) -> bool:
        """
        Validate a term using POS, semantic, and negative keyword checks.

        Args:
            term: Term to validate (string, Doc, or Token)
            context_doc: Optional context document for semantic validation

        Returns:
            bool: True if term passes all validation checks
        """
        # Quick cache check for strings
        cache_key = None
        if isinstance(term, str):
            cache_key = f"{term}:{id(context_doc) if context_doc else 'None'}"
            if cache_key in self._validation_cache:
                self._cache_hits += 1
                return self._validation_cache[cache_key]
        self._cache_misses += 1

        # Convert term to appropriate types for each validation
        term_text, term_doc = self._prepare_term(term, context_doc)

        # Negative keywords check (quickest)
        if term_text.lower() in self.negative_keywords:
            if cache_key:
                self._validation_cache[cache_key] = False
            return False

        # POS validation (medium cost)
        if self.allowed_pos and not self._validate_pos(term_doc):
            if cache_key:
                self._validation_cache[cache_key] = False
            return False

        # Semantic validation (highest cost)
        if (
            self.semantic_validation
            and context_doc
            and not self._validate_semantics(term_text, context_doc)
        ):
            if cache_key:
                self._validation_cache[cache_key] = False
            return False

        # All checks passed
        if cache_key:
            self._validation_cache[cache_key] = True
        return True

    def _prepare_term(self, term, context_doc) -> tuple:
        """Prepare term for validation by converting to consistent types."""
        if isinstance(term, str):
            term_text = term
            term_doc = self.nlp(term)
        elif isinstance(term, Doc):
            term_text = term.text
            term_doc = term
        elif isinstance(term, Token):
            term_text = term.text
            term_doc = term.doc
        else:
            raise TypeError(f"Expected str, Doc, or Token but got {type(term)}")

        return term_text, term_doc

    def _validate_pos(self, doc: Doc) -> bool:
        """Check if document contains at least one token with allowed POS."""
        if not doc or len(doc) == 0:
            return False

        return any(token.pos_ in self.allowed_pos for token in doc)

    def _validate_semantics(self, term_text: str, context_doc: Doc) -> bool:
        """Check if term is semantically related to its context."""
        try:
            # Check if document has vectors
            if not hasattr(context_doc, "has_vector") or not context_doc.has_vector:
                return True  # Skip validation if no vectors available

            # Get term vector
            term_doc = self.nlp(term_text)
            if not term_doc.has_vector:
                return True  # Skip validation if term has no vector

            # Calculate cosine similarity
            similarity = term_doc.similarity(context_doc)
            return similarity >= self.similarity_threshold

        except Exception as e:
            logger.warning(f"Semantic validation error for '{term_text}': {str(e)}")
            return True  # Be lenient on errors

    def get_cache_stats(self) -> Dict:
        """Return statistics about validation cache performance."""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0

        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": hit_rate,
            "cache_size": len(self._validation_cache),
        }

    def clear_cache(self):
        """Clear the validation cache."""
        self._validation_cache.clear()

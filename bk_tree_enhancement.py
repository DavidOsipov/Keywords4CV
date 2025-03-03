"""
Enhanced BK-Tree implementation with optimized fuzzy matching and caching.
Can be imported by keywords4cv.py to replace the default BK-Tree implementation.
"""

import logging
from typing import List, Tuple, Set, Optional, Dict, Any
import pybktree
from Levenshtein import distance
from rapidfuzz import fuzz
from cachetools import LRUCache
import numpy as np

logger = logging.getLogger(__name__)


class EnhancedBKTree:
    """Enhanced BK-Tree with optimized fuzzy matching and caching support."""

    def __init__(self, items: List[str], cache_size: int = 1000):
        """Initialize the enhanced BK-Tree with the provided items."""
        self.bk_tree = pybktree.BKTree(distance, items) if items else None
        self.cache = LRUCache(maxsize=cache_size)
        self._query_count = 0
        self._hit_count = 0

    def find(
        self, query: str, threshold: int, limit: Optional[int] = None
    ) -> List[Tuple[int, str]]:
        """
        Find items within a certain Levenshtein distance threshold of the query string.

        Args:
            query: String to search for
            threshold: Maximum Levenshtein distance allowed
            limit: Maximum number of results to return (optional)

        Returns:
            List of (distance, item) pairs sorted by distance
        """
        if not self.bk_tree:
            return []

        self._query_count += 1
        cache_key = f"{query}_{threshold}_{limit}"

        # Check cache first
        if cache_key in self.cache:
            self._hit_count += 1
            return self.cache[cache_key]

        # Not in cache, perform the actual query
        try:
            # Get all matches within threshold
            matches = self.bk_tree.find(query, threshold)

            # Sort by distance (should already be sorted, but making sure)
            matches.sort(key=lambda x: x[0])

            # Apply limit if specified
            if limit is not None and limit > 0:
                matches = matches[:limit]

            # Cache the results
            self.cache[cache_key] = matches
            return matches

        except Exception as e:
            logger.error(f"Error in BK-Tree search for '{query}': {e}")
            return []

    def get_hit_rate(self) -> float:
        """Return the cache hit rate as a percentage."""
        if self._query_count == 0:
            return 0.0
        return (self._hit_count / self._query_count) * 100

    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about the tree and cache usage."""
        return {
            "tree_size": len(self.bk_tree.tree) if self.bk_tree else 0,
            "cache_size": len(self.cache),
            "cache_maxsize": self.cache.maxsize,
            "queries": self._query_count,
            "hits": self._hit_count,
            "hit_rate": self.get_hit_rate(),
        }

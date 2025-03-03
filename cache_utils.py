"""Caching utilities for the Keywords4CV application."""

import time
import json
import logging
from typing import Any, Dict, Optional, Tuple, Union, List
from abc import ABC, abstractmethod
from cachetools import LRUCache

logger = logging.getLogger(__name__)


class CacheBackend(ABC):
    """Abstract base class for cache backends."""

    @abstractmethod
    def get(self, key: str) -> Any:
        """Get a value from the cache."""
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a value in the cache with optional TTL."""
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete a value from the cache."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear the cache."""
        pass


class MemoryCacheBackend(CacheBackend):
    """In-memory cache backend using LRUCache."""

    def __init__(self, maxsize: int = 10000):
        """Initialize the memory cache backend."""
        self.cache = LRUCache(maxsize=maxsize)
        self.expiry = {}  # Separate dict for expiry times

    def get(self, key: str) -> Any:
        """
        Get a value from the cache, respecting TTL.

        Args:
            key: The cache key

        Returns:
            The cached value or None if not found or expired
        """
        # Check if key exists and not expired
        if key in self.expiry and self.expiry[key] is not None:
            if self.expiry[key] < time.time():
                # Expired, remove from cache
                self.delete(key)
                return None

        # Get from cache
        return self.cache.get(key)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set a value in the cache with optional TTL.

        Args:
            key: The cache key
            value: The value to cache
            ttl: Time-to-live in seconds, or None for no expiry
        """
        self.cache[key] = value

        # Set expiry time if TTL provided
        if ttl is not None:
            self.expiry[key] = time.time() + ttl
        else:
            self.expiry[key] = None

    def delete(self, key: str) -> None:
        """
        Delete a value from the cache.

        Args:
            key: The cache key to delete
        """
        if key in self.cache:
            del self.cache[key]
        if key in self.expiry:
            del self.expiry[key]

    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.expiry.clear()


class CacheManager:
    """Cache manager that can use different backend implementations."""

    def __init__(self, backend: CacheBackend = None, namespace: str = "default"):
        """
        Initialize the cache manager.

        Args:
            backend: The cache backend to use
            namespace: Namespace for keys to avoid collisions
        """
        self.backend = backend or MemoryCacheBackend()
        self.namespace = namespace

    def _make_key(self, key: str) -> str:
        """Create a namespaced key."""
        return f"{self.namespace}:{key}"

    def get(self, key: str) -> Any:
        """Get a value from the cache."""
        return self.backend.get(self._make_key(key))

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a value in the cache with optional TTL."""
        self.backend.set(self._make_key(key), value, ttl)

    def delete(self, key: str) -> None:
        """Delete a value from the cache."""
        self.backend.delete(self._make_key(key))

    def clear(self) -> None:
        """Clear the cache."""
        self.backend.clear()

    def get_or_compute(
        self, key: str, compute_func, ttl: Optional[int] = None, *args, **kwargs
    ):
        """
        Get a value from cache or compute it if not found.

        Args:
            key: The cache key
            compute_func: Function to compute the value if not in cache
            ttl: Time-to-live in seconds
            *args: Arguments to pass to compute_func
            **kwargs: Keyword arguments to pass to compute_func

        Returns:
            The cached or computed value
        """
        cached = self.get(key)
        if cached is not None:
            return cached

        value = compute_func(*args, **kwargs)
        self.set(key, value, ttl)
        return value

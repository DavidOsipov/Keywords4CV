"""
Cache management utilities for Keywords4CV.
"""

import json
import os
import xxhash
import psutil
from typing import Dict, Any, Optional
from cachetools import LRUCache

# Constants
DEFAULT_CACHE_SIZE = 5000
DEFAULT_CACHE_SALT = "default_secret_salt"
CACHE_VERSION = "1.0.0"


def get_cache_salt(config: Dict[str, Any]) -> str:
    """
    Retrieves the cache salt, prioritizing environment variables, then config, then a default.

    Args:
        config: The configuration dictionary

    Returns:
        str: The cache salt to use for hashing operations
    """
    return os.environ.get(
        "K4CV_CACHE_SALT",
        config.get("caching", {}).get("cache_salt", DEFAULT_CACHE_SALT),
    )


def calculate_optimal_cache_size(config: Dict[str, Any]) -> int:
    """
    Calculate the optimal cache size based on available memory and configuration.

    Args:
        config: The configuration dictionary

    Returns:
        int: The calculated optimal cache size
    """
    base_cache_size = config.get("caching", {}).get("cache_size", DEFAULT_CACHE_SIZE)
    scaling_factor = config.get("hardware_limits", {}).get("memory_scaling_factor", 0.3)

    if scaling_factor:
        available_mb = psutil.virtual_memory().available / (1024 * 1024)
        dynamic_size = int(available_mb / scaling_factor)
        return min(base_cache_size, dynamic_size)

    return base_cache_size


class ConfigHasher:
    """
    Handles configuration hashing with intelligent cache invalidation.
    """

    @staticmethod
    def hash_config(
        config: Dict[str, Any], salt: str, sections: Optional[list] = None
    ) -> str:
        """
        Create a hash of relevant configuration sections.

        Args:
            config: Configuration dictionary
            salt: Salt value for the hash
            sections: Specific sections to include (if None, includes commonly cached sections)

        Returns:
            str: Hexadecimal hash of the configuration
        """
        if sections is None:
            sections = [
                "stop_words",
                "stop_words_add",
                "stop_words_exclude",
                "text_processing",
                "caching",
                "validation",
                "keyword_categories",
            ]

        relevant_config = {}
        for section in sections:
            if section in config:
                relevant_config[section] = config.get(section)

        config_str = json.dumps(relevant_config, sort_keys=True)
        return xxhash.xxh3_64(f"{salt}_{config_str}".encode("utf-8")).hexdigest()

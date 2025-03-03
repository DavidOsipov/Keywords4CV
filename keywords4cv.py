# Version 0.26

# Standard library imports
import argparse
import concurrent.futures
import gc
import json
import logging
import os
import random
import re
import shutil
import sys
import time
import torch
from collections import OrderedDict, deque, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from itertools import product
from multiprocessing import TimeoutError as MPTimeoutError
from pathlib import Path
from typing import (
    Dict,
    List,
    Set,
    Tuple,
    NamedTuple,
    Optional,
    Generator,
    Union,
    Literal,
)

# Third-party imports
import nltk
import numpy as np
import pandas as pd
import psutil
import requests
import spacy
import srsly
import yaml
import xxhash
from cachetools import LRUCache
from nltk.corpus import wordnet as wn
from pydantic import BaseModel, Field, ValidationError, field_validator, conlist
from pyarrow import feather
import pyarrow as pa
import pyarrow.parquet as pq
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import pybktree
from Levenshtein import distance
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
import scipy.sparse as sparse

# First-party/local imports
from exceptions import (
    ConfigError,
    InputValidationError,
    DataIntegrityError,
    AuthenticationError,
    NetworkError,
    APIError,
)
import tempfile
import os
import structlog

# Add imports for our new modules
from cache_manager import CacheManager, MemoryCacheBackend
from circuit_breaker import SimpleCircuitBreaker, CircuitOpenError
from semantic_validator import SemanticValidator

# Configure structlog
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),  # Or another renderer
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# --- Sub-Models (for nested structures) ---


class ValidationConfig(BaseModel):
    allow_numeric_titles: bool = True
    empty_description_policy: Literal["warn", "error", "allow"] = "warn"
    title_min_length: int = Field(2, ge=1)
    title_max_length: int = Field(100, ge=1)
    min_desc_length: int = Field(60, ge=1)
    text_encoding: str = "utf-8"

    class Config:
        extra = "forbid"


class DatasetConfig(BaseModel):
    short_description_threshold: int = Field(25, ge=1)
    min_job_descriptions: int = Field(3, ge=1)
    max_job_descriptions: int = Field(120, ge=1)  # Added max_job_descriptions
    min_jobs: int = Field(3, ge=1)

    class Config:
        extra = "forbid"


class SpacyPipelineConfig(BaseModel):
    enabled_components: List[str] = Field(
        default_factory=lambda: [
            "tok2vec",
            "tagger",
            "lemmatizer",
            "entity_ruler",
            "sentencizer",
        ]
    )

    class Config:
        extra = "forbid"


class TextProcessingConfig(BaseModel):
    """Configuration for text processing."""

    spacy_model: str = "en_core_web_lg"
    spacy_pipeline: SpacyPipelineConfig = Field(default_factory=SpacyPipelineConfig)
    ngram_range: Tuple[int, int] = (1, 3)
    whitelist_ngram_range: Tuple[int, int] = (1, 2)
    pos_filter: List[str] = Field(default_factory=lambda: ["NOUN", "PROPN", "ADJ"])
    semantic_validation: bool = True
    similarity_threshold: float = 0.85
    pos_processing: str = "hybrid"
    # New fields for phrase-level synonym handling
    phrase_synonym_source: Literal["static", "api"] = "static"  # Default to static
    phrase_synonyms_path: Optional[str] = None
    api_endpoint: Optional[str] = None
    api_key: Optional[str] = None
    # Add context_window_size
    context_window_size: int = Field(
        1, ge=0
    )  # Default to 1 (one sentence before and after)
    # Add fuzzy_before_semantic option
    fuzzy_before_semantic: bool = (
        True  # Default to fuzzy matching before semantic validation
    )
    # Add wordnet_similarity_threshold for synonym filtering
    wordnet_similarity_threshold: int = Field(
        80, ge=0, le=100
    )  # Default to 80% similarity threshold

    @field_validator("ngram_range", "whitelist_ngram_range")
    @classmethod
    def check_ngram_ranges(cls, value):
        """Check that ngram ranges are valid."""
        if value[0] > value[1]:
            raise ValueError("ngram_range/whitelist_ngram_range start must be <= end")
        return value

    # Added validators for new fields
    @staticmethod
    @field_validator("phrase_synonyms_path")
    def _validate_phrase_synonyms_path(v, values):
        if values.get("phrase_synonym_source") == "static" and not v:
            raise ValueError(
                "phrase_synonyms_path must be provided when phrase_synonym_source is 'static'"
            )
        return v

    @staticmethod
    @field_validator("api_endpoint", "api_key")
    def _validate_api_settings(v, values, field):  # Added 'field' argument
        if values.get("phrase_synonym_source") == "api" and not v:
            raise ValueError(
                f"{field.name} must be provided when phrase_synonym_source is 'api'"
            )
        return v

    class Config:
        """Configuration for extra settings."""


# Update the SynonymEntry class with the source field
class SynonymEntry(BaseModel):
    """Model for validating synonym entries."""

    term: str = Field(..., min_length=1)
    synonyms: conlist(str, min_items=1)
    source: Optional[str] = None  # Add optional source field

    @field_validator("synonyms")
    @classmethod
    def validate_synonyms(cls, v):
        """Ensure each synonym is a non-empty string."""
        if not all(isinstance(s, str) and s.strip() for s in v):
            raise ValueError("All synonyms must be non-empty strings")
        return v

    class Config:
        extra = "forbid"


class CategorizationConfig(BaseModel):
    """Configuration for categorization."""

    default_category: str = "Uncategorized Skills"
    categorization_cache_size: int = 15000
    direct_match_threshold: float = 0.85

    class Config:
        """Configuration for extra settings."""

        extra = "forbid"


class FuzzyMatchingConfig(BaseModel):
    """Configuration for fuzzy matching."""

    enabled: bool = True
    max_candidates: int = 3
    allowed_pos: List[str] = Field(default_factory=lambda: ["NOUN", "PROPN"])
    min_similarity: int = 85
    algorithm: str = "WRatio"
    default_pos_filter: List[str] = Field(
        default_factory=lambda: ["NOUN", "PROPN", "VERB"]
    )
    max_candidates: int = Field(3, ge=1)
    allowed_pos: List[str] = Field(default_factory=lambda: ["NOUN", "PROPN"])
    min_similarity: int = Field(85, ge=0, le=100)
    algorithm: str = "WRatio"

    @field_validator("algorithm")
    @classmethod
    def fuzzy_algorithm_check(cls, v: str) -> str:
        """
        Validates if the provided fuzzy matching algorithm is supported.

        Args:
            cls: The class reference.
            v: The name of the fuzzy matching algorithm to check.

        Returns:
            str: The validated fuzzy matching algorithm name.

        Raises:
            ValueError: If the provided algorithm name is not in the list of supported fuzzy matchers.
        """
        if v not in AdvancedKeywordExtractor.FUZZY_MATCHERS:
            raise ValueError(
                f"Algorithm must be one of: {list(AdvancedKeywordExtractor.FUZZY_MATCHERS.keys())}"
            )
        return v

    class Config:
        extra = "forbid"


class WhitelistConfig(BaseModel):
    """Configuration for whitelist settings."""

    whitelist_recall_threshold: float = Field(0.72, ge=0.0, le=1.0)
    whitelist_cache: bool = True
    fuzzy_matching: FuzzyMatchingConfig = Field(default_factory=FuzzyMatchingConfig)

    class Config:
        extra = "forbid"


class WeightingConfig(BaseModel):
    """Configuration for weighting settings."""

    tfidf_weight: float = Field(0.65, ge=0.0)
    frequency_weight: float = Field(0.35, ge=0.0)
    whitelist_boost: float = Field(1.6, ge=0.0)
    section_weights: Dict[str, float] = Field(
        default_factory=lambda: {"education": 1.2}
    )

    class Config:
        extra = "forbid"


class HardwareLimitsConfig(BaseModel):
    """Configuration for hardware limits."""

    use_gpu: bool = True
    batch_size: int = Field(64, ge=1)
    auto_chunk_threshold: int = Field(100, ge=1)
    memory_threshold: int = Field(70, ge=0, le=100)
    max_ram_usage_percent: int = Field(80, ge=0, le=100)  # Renamed for clarity
    abort_on_oom: bool = True
    max_workers: int = Field(4, ge=1)
    memory_scaling_factor: float = Field(0.3, ge=0.0, le=1.0)

    class Config:
        extra = "forbid"


class CachingConfig(BaseModel):
    """Configuration for caching settings."""

    cache_size: int = Field(15000, ge=1)
    tfidf_max_features: int = Field(100000, ge=1)

    class Config:
        """Configuration for extra settings."""

        extra = "forbid"

    def get_cache_size(self) -> int:
        """Returns the cache size."""
        return self.cache_size

    def get_tfidf_max_features(self) -> int:
        """Returns the maximum number of TF-IDF features."""
        return self.tfidf_max_features


class IntermediateSaveConfig(BaseModel):
    """Configuration for intermediate saving."""

    enabled: bool = True
    save_interval: int = Field(15, ge=0)
    format_: str = Field("feather", alias="format")
    working_dir: str = "working_dir"
    cleanup: bool = True

    class Config:
        extra = "forbid"


class AdvancedConfig(BaseModel):
    """Configuration for advanced settings."""

    dask_enabled: bool = False
    success_rate_threshold: float = Field(0.7, ge=0.0, le=1.0)
    checksum_rtol: float = Field(0.001, ge=0.0)
    negative_keywords: List[str] = Field(
        default_factory=lambda: ["company", "team", "office"]
    )
    section_headings: List[str] = Field(default_factory=list)

    class Config:
        """Configuration for extra settings in AdvancedConfig."""

        extra = "forbid"

        def __repr__(self) -> str:
            """Return string representation of Config."""
            return f"Config(extra={self.extra})"

        def get_config(self) -> dict:
            """Return configuration as dictionary."""
            return {"extra": self.extra}

    def get_success_rate_threshold(self) -> float:
        """Returns the success rate threshold."""
        return self.success_rate_threshold

    def get_checksum_rtol(self) -> float:  # Fixed: Added 'self' parameter
        """Returns the checksum relative tolerance."""
        return self.checksum_rtol


class OptimizationConfig(BaseModel):
    """Configuration for optimization settings."""

    complexity_entity_weight: int = Field(10, ge=1)
    complexity_fallback_factor: float = Field(1.0, ge=0.0)
    trigram_cache_size: int = Field(1000, ge=1)
    trigram_warmup_size: int = Field(100, ge=1)
    q_table_decay: float = Field(0.99, ge=0.0, le=1.0)
    reward_weights: Dict[str, Union[int, float]] = {
        "recall": 2.0,
        "memory": 1.0,
        "time": 0.5,
    }
    reward_std_low: float = Field(0.05, ge=0.0)
    reward_std_high: float = Field(0.2, ge=0.0)
    memory_scale_factor: int = Field(100, ge=1)
    abort_on_oom: bool = True
    max_workers: int = Field(4, ge=1)
    complexity_factor: int = Field(10, ge=1)

    class Config:
        extra = "forbid"
        json_schema_extra = {
            "reward_weights": {
                "recall": "Weight for recall",
                "memory": "Weight for memory usage",
                "time": "Weight for processing time",
            }
        }

    @field_validator("reward_weights")
    @classmethod
    def reward_weights_keys(cls, v):
        """
        Validates that the reward_weights dictionary contains exactly the keys: 'recall', 'memory', 'time'.

        Args:
            cls: The class reference.
            v (dict): The reward_weights dictionary to validate.

        Returns:
            dict: The validated reward_weights dictionary.

        Raises:
            ValueError: If the reward_weights dictionary does not contain the required keys.
        """
        required_keys = {"recall", "memory", "time"}
        if not set(v.keys()) == required_keys:
            raise ValueError(
                f"reward_weights must contain exactly these keys: {required_keys}"
            )
        return v

    def should_abort_on_oom(self) -> bool:
        """Returns whether to abort on out-of-memory errors."""
        return self.abort_on_oom

    def get_max_workers(self) -> int:
        """Returns the maximum number of workers."""
        return self.max_workers


class Config(BaseModel):
    weighting: WeightingConfig = Field(default_factory=WeightingConfig)
    hardware_limits: HardwareLimitsConfig = Field(default_factory=HardwareLimitsConfig)
    caching: CachingConfig = Field(default_factory=CachingConfig)
    intermediate_save: IntermediateSaveConfig = Field(
        default_factory=IntermediateSaveConfig
    )
    advanced: AdvancedConfig = Field(default_factory=AdvancedConfig)
    stop_words: List[str] = Field(default_factory=list)
    stop_words_add: List[str] = Field(default_factory=list)
    stop_words_exclude: List[str] = Field(default_factory=list)
    keyword_categories: Dict[str, List[str]] = Field(default_factory=dict)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    text_processing: TextProcessingConfig = Field(default_factory=TextProcessingConfig)
    validation: ValidationConfig = Field(
        default_factory=ValidationConfig
    )  # Add validation field

    class Config:
        extra = "forbid"


# --- Loading Function ---


def load_config(config_path: str) -> Dict:
    """Loads and validates the configuration from the given path."""
    try:
        # Import here to avoid circular imports
        from config_validation import validate_config_file

        # Use the dedicated function from config_validation module
        return validate_config_file(config_path)

    except FileNotFoundError:
        logger.error("Config file not found: %s", config_path)
        sys.exit(78)  # Configuration error exit code
    except yaml.YAMLError as e:
        line_info = ""
        if hasattr(e, "problem_mark"):
            line_info = f" at line {e.problem_mark.line + 1}, column {e.problem_mark.column + 1}"
        logger.error(f"YAML syntax error in config file {config_path}{line_info}: {e}")
        sys.exit(78)
    except ValidationError as e:
        logger.error("Configuration validation error:\n%s", e)
        sys.exit(78)
    except (IOError, ValueError, KeyError, TypeError) as e:
        logger.exception("Unexpected error loading config: %s", e)
        sys.exit(78)


def get_cache_salt(config: Dict) -> str:
    """
    Retrieves the cache salt, prioritizing environment variables, then config, then a default.

    Args:
        config: The configuration dictionary

    Returns:
        str: The cache salt to use for hashing operations
    """
    return os.environ.get(
        "K4CV_CACHE_SALT",
        config.get("caching", {}).get("cache_salt", "default_secret_salt"),
    )


CACHE_VERSION = "1.0"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("Keywords4CV.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


NLTK_RESOURCES = [
    "corpora/wordnet",
    "corpora/averaged_perceptron_tagger",
    "tokenizers/punkt",
]


def ensure_nltk_resources():
    for resource in NLTK_RESOURCES:
        try:
            nltk.data.find(resource)
        except LookupError:
            logger.info("Downloading NLTK resource: %s", resource)
            nltk.download(resource.split("/")[1], quiet=True)


class ValidationResult(NamedTuple):
    valid: bool
    value: Optional[str]
    reason: Optional[str] = None


class EnhancedTextPreprocessor:
    def __init__(self, config: Dict, nlp):
        self.config = config
        self.nlp = nlp
        self.stop_words = self._load_stop_words()
        # Use more granular regex patterns for better control
        self.regex_patterns = {
            "url": re.compile(r"http\S+|www\.\S+"),
            "email": re.compile(r"\S+@\S+"),
            "special_chars": re.compile(r"[^\w\s'\-]"),
            "whitespace": re.compile(r"\s+"),
        }
        
        # Calculate dynamic cache size based on available memory
        base_cache_size = self.config["caching"].get("cache_size", 5000)
        scaling_factor = self.config["hardware_limits"].get("memory_scaling_factor", 0.3)
        
        if scaling_factor:
            available_mb = psutil.virtual_memory().available / (1024 * 1024)
            dynamic_size = int(available_mb / scaling_factor)
            self._CACHE_SIZE = min(base_cache_size, dynamic_size)
        else:
            self._CACHE_SIZE = base_cache_size
            
        # Use cachetools.LRUCache instead of OrderedDict for better performance
        self._cache = LRUCache(maxsize=self._CACHE_SIZE)
        self.cache_salt = get_cache_salt(config)
        self.config_hash = self._calculate_config_hash()
        self.pos_processing = self.config["text_processing"].get("pos_processing", "")

    def _calculate_config_hash(self) -> str:
        """Calculate a comprehensive hash of all relevant configuration parameters."""
        # Include more relevant configuration parameters
        relevant_config = {
            "stop_words": self.config.get("stop_words", []),
            "stop_words_add": self.config.get("stop_words_add", []),
            "stop_words_exclude": self.config.get("stop_words_exclude", []),
            "text_processing": self.config.get("text_processing", {}),
            "caching": self.config.get("caching", {}),
            "validation": self.config.get("validation", {})
        }
        config_str = json.dumps(relevant_config, sort_keys=True)
        # Use self.cache_salt to salt the hash
        return xxhash.xxh3_64(f"{self.cache_salt}_{config_str}".encode("utf-8")).hexdigest()

    def preprocess(self, text: str) -> str:
        """Preprocess text with instance-specific caching."""
        # Check if config has changed and invalidate cache if needed
        current_hash = self._calculate_config_hash()
        if (current_hash != self.config_hash):
            self._cache.clear()
            self.config_hash = current_hash
            
        # Create cache key with salt
        text_hash = f"{CACHE_VERSION}_{xxhash.xxh3_64((self.cache_salt + text).encode()).hexdigest()}"
        
        if text_hash in self._cache:
            return self._cache[text_hash]

        # Use separate regex patterns for more precise text cleaning
        cleaned = text.casefold()
        cleaned = self.regex_patterns["url"].sub(" ", cleaned)
        cleaned = self.regex_patterns["email"].sub(" ", cleaned)
        cleaned = self.regex_patterns["special_chars"].sub(" ", cleaned)
        cleaned = self.regex_patterns["whitespace"].sub(" ", cleaned).strip()
        return cleaned

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        return [self.preprocess(text) for text in texts]

    def _process_doc_tokens(self, doc):
        """
        Extract tokens from a spaCy document with ultra-optimized processing.
        
        Args:
            doc: spaCy Doc object to process
            
        Returns:
            List[str]: Unique processed tokens
        """
        # Pre-cache config parameters for maximum performance
        pos_filter = set(self.config["text_processing"].get("pos_filter", ["NOUN", "PROPN", "ADJ"]))
        min_len = max(2, self.config["validation"].get("title_min_length", 2))
        stop_words = self.preprocessor.stop_words
        
        # Use a set from the start for O(1) insertion and automatic deduplication
        token_set = set()
        
        # Phase 1: Extract entities first - use set for faster lookups in span checking
        skill_spans = {(ent.start, ent.end) for ent in doc.ents if ent.label_ == "SKILL"}
        
        # Add skill entities directly using set.update for better performance
        token_set.update(ent.text.lower() for ent in doc.ents if ent.label_ == "SKILL")
        
        # Phase 2: Process non-entity tokens with POS filtering
        for i, token in enumerate(doc):
            # Skip tokens that are part of skill entities - avoid duplicate processing
            if any(start <= i < end for start, end in skill_spans):
                continue
                
            # Skip tokens with unwanted POS - do this early to avoid unnecessary processing
            if token.pos_ not in pos_filter:
                continue
                
            # Get lowercase text once for reuse in multiple checks
            text_lower = token.text.lower()
                
            # Skip stop words - check both spaCy's is_stop and our custom stop_words
            if token.is_stop or text_lower in stop_words:
                continue
                
            # Apply minimum length filter before more expensive operations
            if len(token.text) < min_len:
                continue
            
            # Process hyphenated terms with improved validation
            if '-' in token.text:
                parts = token.text.split('-')
                for part in parts:
                    part_lower = part.lower()
                    # Only add parts that meet length and aren't stop words
                    if len(part) >= min_len and part_lower not in stop_words:
                        token_set.add(part_lower)
            else:
                # Add lemmatized token with length validation
                lemma = token.lemma_.lower()
                if len(lemma) >= min_len:
                    token_set.add(lemma)
        
        # Final validation to ensure all tokens meet minimum length
        return [token for token in token_set if len(token) >= min_len]

    def tokenize_batch(self, texts: List[str]) -> Generator[List[str], None, None]:
        # Add custom stop words from config
        custom_stop_words = self.config.get("stop_words", [])
        stop_words.update([w.lower() for w in custom_stop_words])
        
        # Add additional stop words specified in config
        additional_stop_words = self.config.get("stop_words_add", [])
        if additional_stop_words:
            stop_words.update([w.lower() for w in additional_stop_words])
        
        # Remove any stop words that are meant to be excluded
        excluded_stop_words = self.config.get("stop_words_exclude", [])
        if excluded_stop_words:
            stop_words = stop_words - set([w.lower() for w in excluded_stop_words])
        
        if len(stop_words) < 50:
            logger.warning(
                "Stop words list seems unusually small (less than 50 words). Consider adding more stop words to improve text preprocessing."
            )
            
        return stop_words


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    if vec1.shape != vec2.shape:
        logger.warning("Vector dimension mismatch: %s vs %s", vec1.shape, vec2.shape)
        return 0.0

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return np.dot(vec1, vec2) / (norm1 * norm2)


class AdvancedKeywordExtractor:
    FUZZY_MATCHERS = {
        "ratio": fuzz.ratio,
        "partial_ratio": fuzz.partial_ratio,
        "token_sort_ratio": fuzz.token_sort_ratio,
        "token_set_ratio": fuzz.token_set_ratio,
        "WRatio": fuzz.WRatio,
    }

    # Add POS mapping dictionary for alignment between spaCy and WordNet tags
    POS_MAPPING = {
        "NOUN": wn.NOUN,
        "PROPN": wn.NOUN,
        "VERB": wn.VERB,
        "ADJ": wn.ADJ,
        "ADV": wn.ADV,
    }

    def __init__(self, config: Dict, nlp, preprocessor=None):
        self.config = config
        self.nlp = nlp
        self.preprocessor = preprocessor or EnhancedTextPreprocessor(config, nlp)
        self.phrase_synonyms = self._load_phrase_synonyms()
        
        # Create primary cache manager with configurable size
        cache_size = config["caching"].get("cache_size", 5000)
        self.cache_manager = CacheManager(
            backend=MemoryCacheBackend(maxsize=cache_size),
            namespace="api_synonyms"  # Keep original namespace for compatibility
        )

        # Create specialized caches for specific use cases with appropriate sizing
        term_cache_size = config["caching"].get("term_cache_size", 5000)
        self.term_processing_cache = CacheManager(
            backend=MemoryCacheBackend(maxsize=term_cache_size),
            namespace="term_processing"
        )

        # Keep api_cache for backward compatibility
        self.api_cache = {}
        
        # Add cache stats collection for performance monitoring
        self.cache_stats = {"api_hits": 0, "api_misses": 0, "term_hits": 0, "term_misses": 0}
        
        # Initialize circuit breaker
        self.api_circuit_breaker = SimpleCircuitBreaker(
            failure_threshold=config.get("api", {}).get("failure_threshold", 5),
            recovery_timeout=config.get("api", {}).get("recovery_timeout", 60)
        )
        
        self.all_skills = self._load_and_process_all_skills()
        self.category_vectors = {}
        self.cache_size = self.config["caching"].get("cache_size", 5000)
        self.bk_tree = self._build_enhanced_bk_tree()  # Build the enhanced BK-tree
        
        # Initialize the validator
        self.validator = SemanticValidator(config, nlp)

    def _build_enhanced_bk_tree(self):
        """
        Builds memory-aware enhanced BK-tree with adaptive caching for efficient fuzzy matching.
        
        Returns:
            EnhancedBKTree: An enhanced BK-tree for fuzzy matching
        """
        # Import the EnhancedBKTree class
        from bk_tree_enhancement import EnhancedBKTree
        
        # Validate input before processing
        if not self.all_skills:
            logger.warning("Empty skills list for BK-tree initialization")
            return EnhancedBKTree([])
        
        try:
            # Get minimum length from config with fallback
            min_length = self.config.get("validation", {}).get("title_min_length", 2)
            
            # Normalize skills with casefold for better Unicode handling and filter by length
            normalized_skills = [
                skill.strip().casefold() 
                for skill in self.all_skills 
                if len(skill.strip()) >= min_length
            ]
            
            if not normalized_skills:
                logger.warning("No valid skills after normalization, using empty tree")
                return EnhancedBKTree([])
            
            # Calculate memory-aware cache size
            base_cache_size = self.config["caching"].get("bk_tree_cache_size", 1000)
            
            # Dynamic cache sizing based on available memory and skill count
            mem_available_mb = psutil.virtual_memory().available / (1024 * 1024)
            scaling_factor = self.config["hardware_limits"].get("memory_scaling_factor", 0.3)
            
            # Calculate safe cache size based on available memory and skills count
            if scaling_factor > 0:
                # Use at most 25% of available memory per MB for caching
                memory_based_size = int(mem_available_mb * scaling_factor)
                # Take the minimum of config setting, memory-based size, and actual skill count
                cache_size = min(base_cache_size, memory_based_size, len(normalized_skills))
            else:
                # If scaling factor is disabled, use configuration value
                cache_size = min(base_cache_size, len(normalized_skills))
            
            logger.info(f"Using BK-tree cache size of {cache_size} (from base: {base_cache_size})")
            
            # Create the enhanced tree with adaptive cache size
            tree = EnhancedBKTree(normalized_skills, cache_size=cache_size)
            
            # Log success stats
            logger.info(f"Built enhanced BK-tree with {len(normalized_skills)} skills")
            return tree
            
        except Exception as e:
            logger.error(f"Enhanced BK-tree initialization failed: {str(e)}")
            # Return an empty but valid tree to prevent NoneType errors
            return EnhancedBKTree([])

    def _validate_pos_for_term(self, processed_doc):
        """
        Validate if a processed term has the allowed POS tags.
        
        Args:
            processed_doc: spaCy Doc object to check
            
        Returns:
            bool: True if term has at least one allowed POS tag
        """
        fuzzy_config = self.config["whitelist"]["fuzzy_matching"]
        allowed_pos = fuzzy_config.get("allowed_pos", []) or fuzzy_config.get("default_pos_filter", [])
        
        # No POS filter defined, consider valid by default
        if not allowed_pos:
            return True
            
        # Check if any token has an allowed POS
        return any(token.pos_ in allowed_pos for token in processed_doc)

    def _apply_fuzzy_matching(self, keyword: str) -> List[str]:
        """
        Apply fuzzy matching to find similar terms in the whitelist.
        
        Args:
            keyword: The keyword to match
            
        Returns:
            List of matched terms
        """
        if not self.bk_tree:
            return []
            
        fuzzy_config = self.config["whitelist"]["fuzzy_matching"]
        threshold = fuzzy_config.get("min_similarity", 85)
        max_candidates = fuzzy_config.get("max_candidates", 3)
        
        try:
            keyword_lower = keyword.casefold()  # Use casefold for better Unicode handling
            
            # Get matches using the enhanced BK-tree
            matches = self.bk_tree.find(
                query=keyword_lower,
                threshold=threshold,
                limit=max_candidates
            )
            
            # Process and filter matches using the unified validation pipeline
            valid_matches = []
            for _, match in matches:
                # Skip exact duplicates
                if match == keyword_lower:
                    continue
                    
                # Use the new unified validation pipeline
                if self._validate_fuzzy_candidate(match):
                    valid_matches.append(match)
                    
            return valid_matches
            
        except Exception as e:
            logger.error(f"Fuzzy matching error for '{keyword}': {str(e)}")
            return []

    def _load_phrase_synonyms(self) -> Dict[str, List[str]]:
        """Loads and validates phrase synonyms from a JSON file."""
        synonyms_path = self.config["text_processing"].get("phrase_synonyms_path")
        if not synonyms_path:
            return {}
        try:
            with open(synonyms_path, "r", encoding="utf-8") as f:
                raw_synonyms = json.load(f)
            validated_synonyms = {}
            for term, syns in raw_synonyms.items():
                try:
                    entry = SynonymEntry(term=term, synonyms=syns)
                    # Use casefold() instead of lower() for more robust case-insensitivity
                    validated_synonyms[entry.term.casefold()] = [
                        s.casefold() for s in entry.synonyms
                    ]
                except ValidationError as e:
                    logger.warning(f"Invalid synonym entry for term '{term}': {e}")
            logger.info(
                f"Loaded {len(validated_synonyms)} valid phrase synonyms from {synonyms_path}"
            )
            return validated_synonyms
        except FileNotFoundError:
            logger.warning(f"Synonyms file not found: {synonyms_path}")
            return {}
        except Exception as e:
            logger.error(f"Error loading synonyms from {synonyms_path}: {e}")
            return {}

    def _load_and_process_all_skills(self) -> Set[str]:
        """Loads, preprocesses, and expands all skills from keyword_categories."""
        all_skills = set()
        for category_skills in self.config["keyword_categories"].values():
            all_skills.update(category_skills)

        # Preprocess and expand with synonyms and ngrams:
        processed_skills = set()
        for skill in all_skills:
            # 1. Preprocess (casefold, clean) - use casefold instead of lowercase
            cleaned_skill = self.preprocessor.preprocess(skill)

            # 2. Tokenize (using spaCy for consistency)
            doc = self.nlp(cleaned_skill)
            tokens = [
                token.lemma_.casefold()  # Use casefold instead of lower
                for token in doc
                if token.text.casefold() not in self.preprocessor.stop_words
                and len(token.text) > 1
            ]

            # 3. Generate ngrams
            for n in range(
                self.config["text_processing"]["whitelist_ngram_range"][0],
                self.config["text_processing"]["whitelist_ngram_range"][1] + 1,
            ):
                processed_skills.update(self._generate_ngrams(tokens, n))

            # 4. Add original (cleaned) skill  <-  CORRECTED: Add *before* synonym generation
            processed_skills.add(cleaned_skill)

        # 5. Generate and add synonyms (including API and static)
        all_synonyms = self._generate_synonyms(list(processed_skills))  # Use list
        processed_skills.update(all_synonyms)

        return processed_skills

    def _generate_synonyms(self, skills: List[str]) -> Set[str]:
        """Generates synonyms for skills, prioritizing phrase-level matches."""
        synonyms = set()
        # Get similarity threshold from config or use default
        min_similarity = self.config["text_processing"].get("wordnet_similarity_threshold", 80)
        
        for skill in skills:
            skill_cf = skill.casefold()  # Use casefold() for better Unicode handling
            doc = self.nlp(skill)
            
            # Handle phrase-level synonyms
            if self._add_phrase_synonyms(skill_cf, synonyms):
                continue  # Skip word-level processing if phrase synonyms found
            
            # Handle word-level synonyms with filtering
            self._add_wordnet_synonyms(doc, synonyms, min_similarity)
                
        return synonyms

    def _add_phrase_synonyms(self, skill: str, synonyms: Set[str]) -> bool:
        """Adds phrase-level synonyms from configured source. Returns True if any added."""
        source = self.config["text_processing"]["phrase_synonym_source"]
        
        if source == "static" and skill in self.phrase_synonyms:
            synonyms.update(s.casefold() for s in self.phrase_synonyms[skill])
            return True
        elif source == "api":
            api_synonyms = self._get_synonyms_from_api(skill)
            if api_synonyms:
                synonyms.update(s.casefold() for s in api_synonyms)
                return True
        return False

    def _add_wordnet_synonyms(self, doc, synonyms: Set[str], min_similarity: int):
        """Adds filtered WordNet synonyms with similarity check."""
        # Add normalized lemma form
        lemmatized = " ".join(token.lemma_ for token in doc).casefold()
        if lemmatized != doc.text.casefold():
            synonyms.add(lemmatized)
        
        # Add filtered synonyms per token
        for token in doc:
            if token.is_stop or len(token.text) < 3:
                continue
                
            wn_pos = self._convert_spacy_to_wordnet_pos(token.pos_)
            if not wn_pos:
                continue
                
            for syn in wn.synsets(token.text, pos=wn_pos):
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace("_", " ").casefold()
                    if synonym != token.text.casefold() and fuzz.ratio(token.text.casefold(), synonym) >= min_similarity:
                        synonyms.add(synonym)

    def _convert_spacy_to_wordnet_pos(self, spacy_pos: str) -> Optional[str]:
        mapping = {"NOUN": wn.NOUN, "ADJ": wn.ADJ, "ADV": wn.ADV, "VERB": wn.VERB}
        return mapping.get(spacy_pos)

    def _init_categories(self):
        self.categories = self.config["keyword_categories"]

        def calculate_category_vector(category, terms):
            vectors = [
                self._get_term_vector(t)
                for t in terms
                if self._get_term_vector(t).any()
            ]
            return category, {
                "centroid": np.mean(vectors, axis=0) if vectors else None,
                "terms": terms,
            }

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config["hardware_limits"].get("max_workers", 4)
        ) as executor:
            future_to_category = {
                executor.submit(calculate_category_vector, cat, terms): cat
                for cat, terms in self.categories.items()
            }
            for future in concurrent.futures.as_completed(future_to_category):
                category, data = future.result()
                self.category_vectors[category] = data

    @lru_cache(maxsize=5000)
    def _get_term_vector(self, term: str) -> np.ndarray:
        try:
            doc = self.nlp(term)
            if not doc.has_vector:
                return np.array([])
            return doc.vector
        except (AttributeError, ValueError, TypeError) as e:
            logger.warning(
                f"AttributeError during vectorization for '{term}': {str(e)}"
            )
            return np.array([])
        except Exception as e:
            logger.warning(
                f"Unexpected error during vectorization for '{term}': {str(e)}"
            )
            return np.array([])

    def _semantic_categorization(self, term: str) -> str:
        term_vec = self._get_term_vector(term)
        if not term_vec.any():
            return self.default_category
        best_score = self.config["text_processing"].get("similarity_threshold", 0.6)
        best_category = self.default_category
        valid_categories = 0
        for category, data in self.category_vectors.items():
            if data["centroid"] is not None:
                similarity = cosine_similarity(term_vec, data["centroid"])
                if (similarity > best_score):
                    best_score = similarity
                    best_category = category
                valid_categories += 1
            else:
                logger.warning(
                    f"Skipping category '{category}' - no valid centroid available"
                )
        if valid_categories == 0:
            logger.warning(
                f"No valid categories available for semantic categorization of term '{term}'"
            )
        return best_category

    def extract_keywords(
        self, texts: List[str]
    ) -> Generator[Tuple[List[str], List[str]], None, None]:
        """Extracts keywords, yielding (original_tokens, filtered_keywords) for each text."""
        docs = list(self.nlp.pipe(texts))
        for doc, text in zip(docs, texts):
            try:
                # Extract entity keywords (skills directly recognized by spaCy)
                entity_keywords = [
                    ent.text for ent in doc.ents if ent.label_ == "SKILL"
                ]

                # Track skill spans to avoid processing them twice
                skill_spans = [
                    (ent.start, ent.end) for ent in doc.ents if ent.label_ == "SKILL"
                ]
                
                # Extract non-entity tokens (avoiding tokens already in skill entities)
                non_entity_tokens = []
                for i, token in enumerate(doc):
                    if any(start <= i < end for start, end in skill_spans):
                        continue
                    if not token.is_stop and len(token.text) > 1:
                        non_entity_tokens.append(token.text)

                # Preprocess the non-entity text
                preprocessed_text = self.preprocessor.preprocess(" ".join(non_entity_tokens))
                
                # Create original tokens list (unique, case-folded)
                original_tokens = list(set(
                    [t.casefold() for t in entity_keywords] + 
                    [t for t in preprocessed_text.split() 
                     if t not in self.preprocessor.stop_words and len(t) > 1]
                ))
                
                # Initialize keywords set with entity keywords
                keywords = set(entity_keywords)
                
                # Generate n-grams from non-entity tokens using lemmatization for better matching
                non_entity_keywords = set()
                tokens = [token.lemma_.lower() for token in doc 
                          if not token.is_stop and len(token.text) > 1 and
                          not any(start <= token.i < end for start, end in skill_spans)]
                
                for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
                    n_grams = self._generate_ngrams(tokens, n)
                    non_entity_keywords.update(n_grams)
                
                # Combine all keywords
                keywords.update(non_entity_keywords)
                
                # Basic filtering
                filtered_keywords = [
                    kw for kw in keywords
                    if len(kw.strip()) > 1
                    and not any(len(w.strip()) <= 1 for w in kw.split())
                    and not all(w in self.preprocessor.stop_words for w in kw.split())
                ]
                
                # Apply staged filtering for efficiency
                if filtered_keywords:
                    # Stage 1: First separate whitelist keywords (direct matches)
                    whitelisted = []
                    nonwhitelisted = []
                    
                    for kw in filtered_keywords:
                        if kw.lower() in self.all_skills:
                            whitelisted.append(kw)
                        else:
                            nonwhitelisted.append(kw)
                    
                    # Stage 2: Apply fuzzy matching only to non-whitelisted keywords
                    fuzzy_matched = []
                    if nonwhitelisted:
                        # Use existing fuzzy matching but only on non-whitelisted terms
                        matched_list = self._apply_fuzzy_matching_and_pos_filter([nonwhitelisted])
                        if matched_list:
                            fuzzy_matched = matched_list[0]
                    
                    # Stage 3: Apply semantic filtering only if configured
                    if self.config["text_processing"].get("semantic_validation", False):
                        # Combine results and filter semantically
                        combined = whitelisted + fuzzy_matched
                        
                        # Apply semantic filtering
                        if combined:
                            # Apply semantic filter only to fuzzy-matched terms to save processing time
                            # but avoid filtering whitelisted terms which are already validated
                            negative_set = set(self.config.get("negative_keywords", []))
                            filtered_keywords = [
                                kw for kw in whitelisted
                                if kw.lower() not in negative_set
                            ] + [
                                kw for kw in fuzzy_matched
                                if kw.lower() not in negative_set and self._is_in_context(kw, doc)
                            ]
                        else:
                            filtered_keywords = []
                    else:
                        # Skip semantic filtering if disabled
                        filtered_keywords = whitelisted + fuzzy_matched

                yield (original_tokens, filtered_keywords)

            except Exception as e:
                logger.error(f"Error processing text '{text[:50]}...': {e}")
                yield ([], [])  # Yield empty lists on error

    def _apply_fuzzy_matching_and_pos_filter(
        self, keyword_lists: List[List[str]]
    ) -> List[List[str]]:
        """Applies fuzzy matching and POS filtering."""
        fuzzy_config = self.config["whitelist"]["fuzzy_matching"]
        allowed_pos = fuzzy_config.get("allowed_pos") or fuzzy_config.get(
            "default_pos_filter"
        )
        scorer = self.FUZZY_MATCHERS.get(
            fuzzy_config.get("algorithm", "WRatio"), fuzz.WRatio
        )
        score_cutoff = fuzzy_config.get("min_similarity", 85)

        if self.config.get("whitelist_cache", True):
            cached_process_term = lru_cache(maxsize=self.cache_size)(self._process_term)
        else:
            cached_process_term = self._process_term

        filtered_keyword_lists = []
        for keywords in keyword_lists:
            filtered_keywords = []
            threshold = self.config["whitelist"]["fuzzy_matching"].get(
                "min_similarity", 85
            )

            for keyword in keywords:
                keyword_lower = keyword.lower()
                if keyword_lower in self.all_skills:
                    filtered_keywords.append(keyword)
                    continue

                matches = self.bk_tree.find(keyword_lower, threshold)
                if matches:
                    best_match = min(matches, key=lambda x: x[0])[1]
                    term_doc = self._process_term(best_match)
                    if allowed_pos and any(
                        token.pos_ in allowed_pos for token in term_doc
                    ):
                        filtered_keywords.append(best_match)

            filtered_keyword_lists.append(filtered_keywords)
        return filtered_keyword_lists

    def _semantic_filter(
        self, keyword_lists: List[List[str]], docs: List
    ) -> List[List[str]]:
        negative_set = set(self.config.get("negative_keywords", []))
        return [
            [
                kw
                for kw in kws
                if kw not in negative_set and self._is_in_context(kw, doc)
            ]
            for kws, doc in zip(keyword_lists, docs)
        ]

    def _is_in_context(self, keyword: str, doc) -> bool:
        # Extract context window for semantic validation
        context_window = self._get_context_window(
            self._extract_sentences(doc.text), keyword
        )
        
        if not context_window:
            return False
            
        # Create context doc
        context_doc = self.nlp(context_window)
        
        # Use validator for semantic validation
        return self.validator.validate_term(keyword, context_doc)

    # Added method for extracting sentences with custom rules
    def _extract_sentences(self, text: str) -> List[str]:
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]

        # Add custom rules *after* spaCy's segmentation (optional and configurable)
        if self.config["text_processing"].get("custom_sentence_rules", False):
            # Example: Split sentences after bullet points (•)
            new_sentences = []
            for sentence in sentences:
                parts = sentence.split("•")  # Simple split, might need refinement
                new_sentences.extend(part.strip() for part in parts if part.strip())
            sentences = new_sentences

        return sentences

    # Replace the existing _get_context_window method
    def _get_context_window(self, sentences: List[str], keyword: str) -> str:
        """Extracts a context window, respecting paragraph breaks."""
        window_size = self.config["text_processing"].get("context_window_size", 1)
        paragraphs = [
            p.strip()
            for p in re.split(r"\n{2,}|\r\n{2,}", "\n".join(sentences))
            if p.strip()
        ]  # More robust
        for para in paragraphs:
            para_sents = [sent.text.strip() for sent in self.nlp(para).sents]
            for i, sent in enumerate(para_sents):
                if keyword.lower() in sent.lower():
                    start = max(0, i - window_size)
                    end = min(len(para_sents), i + window_size + 1)
                    return " ".join(para_sents[start:end])
        return ""

    def _extract_sections(self, text: str) -> Dict[str, str]:
        """
        Extract sections from text by identifying section headings.
        Returns a dictionary mapping section names to their content.
        """
        sections = {}
        
        # Find all section headings and their positions
        headings_with_positions = []
        for match in self._section_heading_re.finditer(text):
            heading = match.group(0).strip().rstrip(":").lower()
            headings_with_positions.append((heading, match.start(), match.end()))
        
        # If no headings found, put everything in "General"
        if not headings_with_positions:
            sections["General"] = text
            return sections
        
        # Handle text before the first heading
        if headings_with_positions[0][1] > 0:
            sections["General"] = text[:headings_with_positions[0][1]].strip()
        
        # Handle text between headings and after the last heading
        for i, (heading, _, end_pos) in enumerate(headings_with_positions):
            if i < len(headings_with_positions) - 1:
                next_start = headings_with_positions[i+1][1]
                sections[heading] = text[end_pos:next_start].strip()
            else:
                # Last heading - grab text to the end
                sections[heading] = text[end_pos:].strip()
        
        return sections

    def save_analysis_results(self, df: pd.DataFrame, path: str) -> None:
        records = df.reset_index().to_dict(orient="records")
        srsly.write_jsonl(path, (r for r in records))

    def _calc_metrics(self, chunk_results: Tuple[pd.DataFrame, pd.DataFrame]) -> Dict:
        start_time = time.time()
        summary, _ = chunk_results

        # Calculate recall against the *original* set of skills (before expansion)
        original_skills = set()
        for category_skills in self.config["keyword_categories"].values():
            original_skills.update(s.lower() for s in category_skills)

        # Calculate recall against the *expanded* set of skills
        expanded_skills = set(s.lower() for s in self.keyword_extractor.all_skills)

        # Extract lowercased keywords from summary
        extracted_keywords = set(summary.index.str.lower())

        # Calculate metrics
        original_recall = (
            len(extracted_keywords & original_skills) / len(original_skills)
            if original_skills
            else 0
        )

        expanded_recall = (
        )

        # Prevent division by zero by checking both emptiness and length
        time_per_job = (
            (time.time() - start_time) / len(summary)
            if not summary.empty and len(summary) > 0
            else 0.5
        )

        return {
            "original_recall": original_recall,
            "expanded_recall": expanded_recall,  # This will be our primary recall metric
            "precision": precision,
            "f1_score": f1_score,
            "memory": psutil.virtual_memory().percent,
            "time_per_job": time_per_job,
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type(
            (requests.exceptions.RequestException, json.JSONDecodeError),
        ),
    )
    def _get_synonyms_from_api(self, phrase: str) -> List[str]:
        """Fetches synonyms from the API with caching, circuit breaker and retries."""
        phrase_lower = phrase.lower()
        
        # Check cache first with TTL support
        cache_entry = self.cache_manager.get(phrase_lower)
        if cache_entry is not None:
            return cache_entry
        
        # Circuit breaker pattern via decorator
        if self.api_circuit_breaker.state == "OPEN":
            logger.warning(f"Circuit breaker open for API, using empty synonyms for '{phrase}'")
            return []

        endpoint = self.config["text_processing"]["api_endpoint"]
        api_key = self.config["text_processing"]["api_key"]
        headers = {"Authorization": f"Bearer {api_key}"}
        params = {"phrase": phrase}

        try:
            response = requests.get(endpoint, headers=headers, params=params, timeout=5)
            
            # Enhanced status code handling with security considerations
            if response.status_code == 401:
                logger.error(f"API authentication failed for phrase: {phrase}")
                # Don't cache authentication failures permanently
                self.cache_manager.set(phrase_lower, [], time.time() + 300)  # Short TTL for auth failures
                raise AuthenticationError("Invalid API credentials")
            elif response.status_code == 403:
                logger.warning(f"API authorization failed for phrase: {phrase}")
                self.cache_manager.set(phrase_lower, [], time.time() + 300)  # Short TTL
                raise AuthenticationError("Insufficient API privileges")
            elif response.status_code == 429:
                logger.warning("API rate limit exceeded. Backing off...")
                # Don't cache rate limit errors
                raise NetworkError("Rate limit exceeded")
            elif 500 <= response.status_code < 600:
                logger.warning(f"Server error: {response.status_code}")
                # Don't cache server errors
                raise APIError(f"Server error: {response.status_code}")
                    
            response.raise_for_status()  # Raises HTTPError for other bad requests

            data = response.json()
            if "synonyms" not in data:
                logger.warning(f"API response missing 'synonyms' key for phrase: {phrase}")
                synonyms = []
            else:
                synonyms = data["synonyms"]
            
            # Cache with TTL based on response headers
            cache_ttl = None
            cache_control = response.headers.get("Cache-Control", "")
            if "max-age" in cache_control:
                try:
                    ttl = int(re.search(r'max-age=(\d+)', cache_control).group(1))
                    cache_ttl = time.time() + ttl
                except (AttributeError, ValueError):
                    # Default TTL if parsing fails
                    cache_ttl = time.time() + 3600  # 1 hour default
            else:
                # Use default TTL if no cache-control
                cache_ttl = time.time() + 3600  # 1 hour default
                
            # Store in cache with TTL
            self.cache_manager.set(phrase_lower, synonyms, cache_ttl - time.time())
            return synonyms

        except requests.exceptions.Timeout:
            logger.warning(f"API timeout for phrase: {phrase}")
            # Don't cache timeouts
            raise NetworkError("API request timed out")
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Connection error to API for phrase: {phrase} - {e}")
            # Don't cache connection errors
            raise NetworkError(f"Connection error: {e}")
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON response for phrase: {phrase} - {e}")
            # Cache invalid responses with short TTL
            self.cache_manager.set(phrase_lower, [], 300)  # 5 min TTL
            raise APIError(f"Invalid JSON response: {e}")
        except requests.exceptions.HTTPError as e:
            # Already logged specific status codes above
            # Don't cache HTTP errors
            raise APIError(f"HTTP error: {e}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"API request error for phrase: {phrase} - {e}")
            # Don't cache general request errors
            raise NetworkError(f"Request error: {e}")

    def _generate_ngrams(self, tokens: List[str], n: int) -> Set[str]:
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer")  # Robustness check

        filtered_tokens = [
            token
            for token in tokens
            if len(token.strip()) > 1 and token not in self.preprocessor.stop_words
        ]
        if len(filtered_tokens) < n:
            return set()

        ngrams = {
            " ".join(filtered_tokens[i : i + n])
            for i in range(len(filtered_tokens) - (n - 1))
            if all(len(t) > 1 for t in filtered_tokens[i : i + n])
        }
        return ngrams

    def _detect_keyword_section(self, keyword: str, text: str) -> str:
        # Pre-compute casefolded versions for better performance
        keyword_cf = keyword.casefold()
        text_cf = text.casefold()
        
        match = re.search(
            rf"\b{re.escape(keyword_cf)}\b", text_cf
        )  # Use pre-computed casefold
        
        if match:
            match_start = match.start()
            sections = {}
            current_section = "General"
            sections[current_section] = ""

            # Find section headings *before* the keyword match
            for heading_match in self._section_heading_re.finditer(text_cf):
                if heading_match.start() < match_start:
                    current_section = heading_match.group(0).strip().rstrip(":").casefold()
                else:
                    break  # Stop searching after the keyword - add this optimization
            return current_section

        return "default"  # Add explicit return for no-match case

    def _process_term(self, term: str):
        """
        Process a term through spaCy for POS tagging and other linguistic analysis.

        Args:
            term: The text term to process

        Returns:
            The processed spaCy Doc object
        """
        try:
            return self.nlp(term)
        except Exception as e:
            logger.warning(f"Error processing term '{term}': {e}")
            # Return empty doc as fallback
            return self.nlp("")

    def _validate_fuzzy_candidate(self, candidate: str) -> bool:
        """
        Unified validation pipeline for fuzzy match candidates.
        
        Args:
            candidate: String candidate to validate
            
        Returns:
            bool: True if candidate passes all validation checks
        """
        # Skip empty candidates
        if not candidate or len(candidate.strip()) < 2:
            return False
            
        # Process the term to get a spaCy doc for POS validation
        processed_doc = self._process_term_cached(candidate)
        if not processed_doc or not len(processed_doc):
            return False
        
        # Apply POS filtering using existing method
        if not self._validate_pos_for_term(processed_doc):
            return False
        
        # Apply semantic validation if enabled
        if self.config["text_processing"].get("semantic_validation", False):
            # Need context for semantic validation, use a minimal context
            # This is a compromise - ideally we'd have the actual context
            minimal_context = self.nlp(f"The term {candidate} is relevant.")
            return self.validator.validate_term(candidate, minimal_context)
        
        return True


from multiprocess_helpers import init_worker, process_chunk
import multiprocessing

class ParallelProcessor:
    def __init__(self, config: Dict, keyword_extractor, nlp):  # Accept nlp parameter
        self.config = config
        self.keyword_extractor = keyword_extractor
        self.nlp = nlp  # Store the passed-in nlp model
        self.disabled_pipes = self.config["text_processing"]["spacy_pipeline"].get(
            "disabled_components", []
        )
        # Initialize complexity_cache
        self.complexity_cache = LRUCache(maxsize=1000)  # Add a reasonable size

    def get_optimal_workers(self, texts: List[str]) -> int:
        sample_size = max(
            10, min(100, int(len(texts) * 0.1))
        )  # 10% of texts, clamped between 10 and 100
        # Ensure sample size does not exceed population size:
        sample_size = min(sample_size, len(texts))
        sample_texts = random.sample(texts, sample_size)
        complexities = []

        for text in sample_texts:
            if (score := self.complexity_cache.get(text)) is None:
                try:
                    doc = self.nlp(text)
                    score = len(text) + len(doc.ents) * self.config["optimization"].get(
                        "complexity_entity_weight", 10
                    )
                except Exception as e:
                    logger.error(
                        f"Complexity calc failed for '{text[:50]}...': {str(e)}. Using fallback."
                    )
                    score = len(text) * self.config["optimization"].get(
                        "complexity_fallback_factor", 1.0
                    )
            self.complexity_cache[text] = score
            complexities.append(score)  # Correct placement

        if len(self.complexity_cache) >= self.complexity_cache.maxsize:
            self.complexity_cache.popitem()

        if sample_size >= 10:  # Log only if sample is meaningful
            logger.info(
                f"Complexity stats (sample size={sample_size}): mean={np.mean(complexities):.1f}, max={max(complexities)}"
            )

        avg_complexity = np.mean(complexities) if complexities else 1
        mem_available = psutil.virtual_memory().available / (1024**3)

        # --- Enhanced GPU Memory Check with multi-level fallback ---
        if self.config["hardware_limits"].get("use_gpu", False):
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    free_mem, total_mem = torch.cuda.mem_get_info()
                    
                    # Get threshold from config with safety defaults
                    gpu_mem_threshold = self.config["hardware_limits"].get(
                        "gpu_memory_threshold_gb", 2.0
                    ) * 1e9  # Convert GB to bytes

                    # Calculate memory requirements per worker dynamically
                    avg_ent_count = np.mean([c for c in complexities if c > 0]) if complexities else 1
                    mem_per_worker = (self.config["optimization"].get("memory_per_entity", 0.1) 
                                    * avg_ent_count * 1e6)  # MB

                    if free_mem < gpu_mem_threshold:
                        # Calculate safe worker count using multiple constraints
                        max_possible = min(
                            int(free_mem / (mem_per_worker * 1e6)),  # Absolute memory limit
                            self.config["hardware_limits"].get("max_workers", 4),  # Config cap
                            int(psutil.cpu_count() * 0.75)  # CPU core limit
                        )
                        safe_workers = max(1, max_possible)
                        
                        logger.warning(
                            f"GPU memory constrained: {free_mem/1e9:.1f}GB free. "
                            f"Reducing workers to {safe_workers} (mem/worker: {mem_per_worker:.1f}MB)"
                        )
                        return safe_workers

            except Exception as e:
                logger.error(f"GPU check failed: {str(e)}. Falling back to CPU-only calculation")
                return max(1, self.config["hardware_limits"].get("max_workers", 4) // 2)

        # Existing CPU-based calculation
        return min(
            self.config["hardware_limits"].get("max_workers", 4),
            max(
                1,
                int(
                    (mem_available * avg_complexity)
                    / self.config["optimization"]["complexity_factor"]
                ),
            ),
        )

    def extract_keywords(self, texts: List[str]) -> List[List[str]]:
        """Extracts keywords from texts using an optimized parallel processing approach."""
        workers = self.get_optimal_workers(texts)
        chunk_size = max(1, len(texts) // workers)
        chunks = self._chunk_texts(texts, chunk_size)
        
        # Check if we should use GPU
        use_gpu = self.config["hardware_limits"].get("use_gpu", False)
        
        # Use Pool with initializer to avoid transmitting the large model
        with multiprocessing.Pool(
            processes=workers,
            initializer=init_worker,
            initargs=(self.config, use_gpu),
            maxtasksperchild=self.config.get("hardware_limits", {}).get("maxtasksperchild", None)
        ) as pool:
            results = pool.map(process_chunk, chunks)
            
        # Flatten results as in original code
        return [kw for chunk_result in results for kw in chunk_result]

    def _process_text_chunk(
        self, texts: List[str]
    ) -> Generator[Tuple[List[str], List[str]], None, None]:
        # Use the already loaded nlp model instead of loading it again
        # Create a keyword extractor with our existing model
        keyword_extractor = AdvancedKeywordExtractor(self.config, self.nlp)

        # Return the generator directly without materializing as a list
        return keyword_extractor.extract_keywords(texts)

    def _chunk_texts(self, texts: List[str], chunk_size: int) -> List[List[str]]:
        return [texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)]


class TrigramOptimizer:
    def __init__(
        self, config: Dict, all_skills: List[str], keyword_extractor
    ):  
        self.config = config
        
        # Add validation to ensure keyword_extractor has valid nlp attribute
        if not hasattr(keyword_extractor, 'nlp'):
            logger.error("keyword_extractor missing nlp attribute")
            raise ConfigError("keyword_extractor must provide an nlp model")
        
        self.nlp = keyword_extractor.nlp
        self.cache = LRUCache(
            maxsize=config["optimization"].get("trigram_cache_size", 1000)
        )
        self.hit_rates = deque(maxlen=100)
        self.hit_rates.append(0)

        # Use the existing preprocessor from keyword_extractor
        self.keyword_extractor = keyword_extractor
        self.preprocessor = self.keyword_extractor.preprocessor

        # Dynamic cache sizing based on available memory and configuration
        warmup_size = min(
            config["optimization"].get("trigram_warmup_size", 100),
            len(all_skills),
            int(psutil.virtual_memory().available / (1024 * 1024 * 0.1)),
        )
        
        # Log empty skills warning
        if not all_skills:
            logger.warning("No skills loaded from categories for TrigramOptimizer")

        # --- MODIFIED: Warmup using preprocessed tokens ---
        successful_warmups = 0
        for skill in all_skills[:warmup_size]:
            try:
                cleaned_skill = self.preprocessor.preprocess(skill)
                doc = self.nlp(cleaned_skill)
                tokens = [
                    token.lemma_.lower()
                    for token in doc
                    if token.text.lower() not in self.preprocessor.stop_words
                    and len(token.text) > 1
                ]
                # Use the tokens for warmup
                for n in range(1, 3 + 1):  # Generate up to trigrams
                    for ngram in self._generate_ngrams(tokens, n):
                        self.get_candidates(ngram)  # Add to cache
                successful_warmups += 1
            except (ValueError, TypeError, AttributeError) as e:
                logger.warning(f"Error during warmup for '{skill[:50]}...': {e}")
        
        # Enhanced logging with success rate
        success_rate = successful_warmups / warmup_size if warmup_size > 0 else 0
        logger.info(f"Warmed up trigram cache with {successful_warmups}/{warmup_size} terms ({success_rate:.1%} success rate)")

    @lru_cache(maxsize=1024)  # Use a reasonable maxsize
    def _generate_ngrams(self, tokens: List[str], n: int) -> Set[str]:
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer")  # Robustness check

        filtered_tokens = [
            token
            for token in tokens
            if len(token.strip()) > 1 and token not in self.preprocessor.stop_words
        ]
        if len(filtered_tokens) < n:
            return set()

        ngrams = {
            " ".join(tokens[i : i + 3])
            for i in range(len(tokens) - 2)
            if all(len(t) > 1 for t in tokens[i : i + 3])
        }
        return ngrams

    def get_candidates(self, text: str) -> List[str]:
        text_hash = xxhash.xxh3_64(text.encode()).hexdigest()
        if text_hash in self.cache:
            self.hit_rates.append(1)
            return self.cache[text_hash]

        try:
            doc = self.nlp(text)
            tokens = [
                token.text.lower()
                for token in doc
                if not token.is_stop and len(token.text) > 1
            ]
            trigrams = {
                " ".join(tokens[i : i + 3])
                for i in range(len(tokens) - 2)
                if all(len(t) > 1 for t in tokens[i : i + 3])
            }
            candidates = list(trigrams) if trigrams else [text]
            self._adjust_cache_size()
            self.cache[text_hash] = candidates
            self.hit_rates.append(0)
            return candidates
        except Exception as e:
            if random.random() < 0.1:
                logger.error(f"Trigram error in '{text[:50]}...': {str(e)}")
            return []

    def _adjust_cache_size(self):
        hit_rate = sum(self.hit_rates) / len(self.hit_rates) if self.hit_rates else 0
        new_size = min(
            self.config["optimization"]["max_trigram_cache_size"],
            int(self.config["optimization"]["trigram_cache_size"] * (1 + hit_rate)),
        )
        while len(self.cache) > new_size:
            self.cache.popitem()


class SmartChunker:
    def __init__(self, config: Dict):
        self.config = config
        self.q_table = LRUCache(maxsize=10000)
        self.timestamps = defaultdict(float)
        self.decay_factor = config["optimization"].get("q_table_decay", 0.99)
        self.base_learning_rate = config["optimization"].get("chunk_learning_rate", 0.1)
        self.learning_rate = self.base_learning_rate
        self.reward_history = deque(maxlen=10)
        self.state_history = []  # Changed from deque to a regular list
        self.state_history_maxlen = 100  # Store the maxlen as a separate variable
        self.last_reward = None

    def get_chunk_size(self, dataset_stats: Dict) -> int:
        state = (
            int(dataset_stats["avg_length"] / 100),
            int(dataset_stats["num_texts"] / 1000),
            int(psutil.virtual_memory().percent / 10),
        )
        self.state_history.append(state)
        # Manually manage the list size to mimic deque's maxlen behavior
        if len(self.state_history) > self.state_history_maxlen:
            self.state_history = self.state_history[-self.state_history_maxlen:]
            
        for key in list(self.q_table.keys()):
            self.config["dataset"].get("min_chunk_size", 10),
            min(
                self.config["dataset"].get("max_chunk_size", 200),
                int(
                    self.q_table.get(
                        state, self.config["dataset"]["default_chunk_size"]
                    )
                ),
            ),
        )

    def update_model(self, reward: float, chunk_size: Optional[int] = None):
        self.reward_history.append(reward)
        if (chunk_size is not None):
            self.config["dataset"]["default_chunk_size"] = chunk_size
        if len(self.reward_history) >= 10:
            reward_std = np.std(list(self.reward_history))
            low, high = (
                self.config["optimization"].get("reward_std_low", 0.05),
                self.config["optimization"].get("reward_std_high", 0.2),
            )
            if reward_std < low:
                self.learning_rate = max(
                    self.base_learning_rate * 0.5, self.learning_rate * 0.95
                )
                logger.debug(
                    f"Learning rate reduced to {self.learning_rate:.3f} due to plateau"
                )
            elif reward_std > high:
                self.learning_rate = min(
                    self.base_learning_rate, self.learning_rate * 1.1
                )
                logger.debug(
                    f"Learning rate increased to {self.learning_rate:.3f} for adaptation"
                )
        if self.learning_rate < self.base_learning_rate * 0.1:
            self.learning_rate = self.base_learning_rate
            logger.info("Learning rate reset to base value")
        for state in self.state_history:
            self.q_table[state] += self.learning_rate * (
                reward + 0.9 * max(self.q_table.values()) - self.q_table[state]
            )
            self.timestamps[state] = time.time()
        self.last_reward = reward


class AutoTuner:
    def __init__(self, config: Dict):
        self.config = config

    def tune_parameters(self, metrics: Dict, trigram_hit_rate: float) -> Dict:
        new_params = {"chunk_size": self._adjust_chunk_size(metrics["memory"])}
        if metrics["recall"] < 0.7:
            new_params["pos_processing"] = (
                "original" if trigram_hit_rate < 0.5 else "hybrid"
            )
        elif trigram_hit_rate > 0.8 and metrics["memory"] < 60:
            new_params["pos_processing"] = "noun_chunks"
        new_params["chunk_size"] = np.clip(
            new_params["chunk_size"],
            self.config["dataset"].get("min_chunk_size", 10),
            self.config["dataset"].get("max_chunk_size", 200),
        )
        return new_params  # Return the new parameters


class OptimizedATS:
    def __init__(self, config_path: str = "config.yaml"):
        # 1. Load config FIRST
        self.config = load_config(config_path)
        if not self.config.get("keyword_categories"):
            raise ConfigError(
                "The 'keyword_categories' section must be defined in the config.yaml file and contain at least one category."
            )
        
        # 2. Validate basic config structure
        self._validate_config()
        
        # 3. Initialize core NLP component FIRST
        self.nlp = self._load_and_configure_spacy_model()
        self._add_entity_ruler(self.nlp)
        
        # 4. Initialize preprocessing SECOND
        self.preprocessor = EnhancedTextPreprocessor(self.config, self.nlp)
        
        # 5. Initialize keyword extraction THIRD
        self.keyword_extractor = AdvancedKeywordExtractor(
            config=self.config,
            nlp=self.nlp,
            preprocessor=self.preprocessor  # Explicit dependency
        )
        
        # 6. Initialize keyword canonicalizer - NEW
        self.keyword_canonicalizer = KeywordCanonicalizer(
            nlp=self.nlp,
            config=self.config
        )
        
        # 7. Initialize parallel processing FOURTH
        self.processor = ParallelProcessor(
            config=self.config,
            keyword_extractor=self.keyword_extractor,
            nlp=self.nlp
        )
        
        # 8. Initialize optimization components LAST
        self.trigram_optim = TrigramOptimizer(
            config=self.config,
            all_skills=self.keyword_extractor.all_skills,
            keyword_extractor=self.keyword_extractor  # Pass keyword_extractor directly
        )
        
        # 9. Initialize remaining components
        self.chunker = SmartChunker(self.config)
        self.tuner = AutoTuner(self.config)
        
        # 10. Set up working directory (from existing code)
        self.working_dir = Path(
            self.config["intermediate_save"].get("working_dir", "working_dir")
        )
        self.working_dir.mkdir(exist_ok=True)
        self.run_id = xxhash.xxh3_64(
            f"{time.time()}_{random.randint(0, 1000)}".encode()
        ).hexdigest()
        
        # 11. Initialize categories
        self._init_categories()
        self.checksum_manifest_path = (
            self.working_dir / f"{self.run_id}_checksums.jsonl"
        )

    def sanitize_input(self, jobs: Dict) -> Dict:
        """Sanitizes job descriptions based on config (numeric titles, empty descriptions)."""
        cleaned = {}
        allow_numeric_titles = self.config["validation"].get(
            "allow_numeric_titles", False
        )  # Get defaults
        empty_description_policy = self.config["validation"].get(
            "empty_description_policy", "warn"
        )

        for title, desc in jobs.items():
            if not isinstance(title, str):
                if allow_numeric_titles:
                    title = str(title)
                else:
                    logger.warning(f"Discarding non-string title: {title}")
                    if self.config.get("strict_mode", False):
                        raise InputValidationError(f"Non-string title: {title}")
                    continue
            if not isinstance(desc, str) or not desc.strip():
                if empty_description_policy == "error":
                    logger.error(f"Invalid description for {title}")
                    if self.config.get("strict_mode", False):
                        raise InputValidationError(f"Invalid description for {title}")
                    continue

            cleaned[title] = desc.strip()
        return cleaned

    def analyze_jobs(self, job_descriptions: Dict):
        if not job_descriptions:
            logger.warning("Empty input: No job descriptions to process")
            return

        # Sanitize input data first
        job_descriptions = self.sanitize_input(job_descriptions)
        if not job_descriptions:
            logger.warning("No valid job descriptions after sanitization")
            return

        dataset_stats = self._calc_dataset_stats(job_descriptions)
        chunk_size = self.chunker.get_chunk_size(dataset_stats)
        results = []
        batch_idx = 0
        save_interval = self.config["intermediate_save"].get("save_interval", 0)

        try:
            for i, chunk in enumerate(self._create_chunks(job_descriptions)):
                if not chunk:
                    logger.warning("Empty chunk encountered. Skipping.")
                    continue

                # Process the chunk without converting generator to list
                chunk_results = self._process_chunk(chunk)
                if chunk_results:
                    results.append(chunk_results)

                if save_interval > 0 and (i + 1) % save_interval == 0:
                    self._save_intermediate(
                        batch_idx, [r[0] for r in results], [r[1] for r in results]
                    )
                    batch_idx += 1
                    results = []

                metrics = self._calc_metrics(chunk_results)
                hit_rate = (
                    np.mean(list(self.trigram_optim.hit_rates))
                    if self.trigram_optim.hit_rates
                    else 0
                )
                new_params = self.tuner.tune_parameters(metrics, hit_rate)
                # Update relevant components directly
                self.chunker.update_model(
                    self._calc_reward(metrics), new_params.get("chunk_size")
                )

            gc.collect()

            if results:
                self._save_intermediate(
                    batch_idx, [r[0] for r in results], [r[1] for r in results]
                )

        except (MemoryError, MPTimeoutError, Exception) as e:
            logger.exception("Error during analysis: %s", e)
            gc.collect()
            raise

    def _calc_dataset_stats(self, job_descriptions: Dict) -> Dict:
        lengths = [len(desc) for desc in job_descriptions.values()]
        return {
            "avg_length": np.nanmean(lengths) if lengths else 0,  # Use np.nanmean instead of np.mean
            "num_texts": len(job_descriptions),
        }

    def _create_chunks(self, job_descriptions: Dict) -> List[Dict]:
        """
        Split the job_descriptions dictionary into chunks based on the current configuration.
        """
        # Get the chunk size from the configuration or use a default
        chunk_size = self.config.get("dataset", {}).get("default_chunk_size", 50)
        items = list(job_descriptions.items())
        return [
            dict(items[i : i + chunk_size])
            for i in range(0, len(job_descriptions), chunk_size)
        ]

    def _process_chunk(
        self, chunk: Dict
    ) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
        if not chunk:
            return None
        try:
            # Create keyword generator directly from chunk values
            keywords_generator = self.processor._process_text_chunk(
                list(chunk.values())
            )

            # NEW: Collect and canonicalize keywords before processing
            all_keywords = []
            job_titles = list(chunk.keys())
            original_keywords_by_job = {}
            
            # First pass to extract keywords
            for i, (original_tokens, expanded_keywords) in enumerate(keywords_generator):
                job_title = job_titles[i] if i < len(job_titles) else f"Job_{i}"
                combined_keywords = list(set(original_tokens + expanded_keywords))
                all_keywords.extend(combined_keywords)
                original_keywords_by_job[job_title] = combined_keywords
            
            # Canonicalize all keywords
            all_skills = self.keyword_extractor.all_skills if hasattr(self.keyword_extractor, 'all_skills') else None
            canonical_keywords = self.keyword_canonicalizer.canonicalize_keywords(all_keywords, all_skills)
            
            # Map original keywords to canonical forms
            keyword_to_canonical = {}
            for original in all_keywords:
                # Find the best canonical form for each original keyword
                best_match = self._find_best_canonical_match(original, canonical_keywords)
                keyword_to_canonical[original] = best_match
                
            # Create modified keywords_generator using canonicalized forms
            modified_keywords = []
            for job_title, original_kws in original_keywords_by_job.items():
                canonical_kws = [keyword_to_canonical.get(kw, kw) for kw in original_kws]
                # Remove duplicates that might have been introduced through canonicalization
                canonical_kws = list(dict.fromkeys(canonical_kws))
                modified_keywords.append((original_kws, canonical_kws))
                
            # Now create TF-IDF matrix using canonicalized keywords
            dtm, features = self._create_tfidf_matrix(modified_keywords)

            # Calculate scores using streaming approach with canonicalized keywords
            results = list(
                self._calculate_scores(dtm, features, modified_keywords, chunk)
            )

            df = pd.DataFrame(results)
            if df.empty:
                return pd.DataFrame(), pd.DataFrame()
            summary_chunk = df.groupby("Keyword").agg(
                {"Score": ["sum", "mean"], "Job Title": "nunique"}
            )
            summary_chunk.columns = ["Total_Score", "Avg_Score", "Job_Count"]
            details_chunk = df
            return summary_chunk, details_chunk
        except Exception as e:
            logger.exception(f"Error processing chunk: {e}")
            raise  # Always raise the exception
    
    def _find_best_canonical_match(self, original: str, canonical_keywords: List[str]) -> str:
        """
        Find the best matching canonical form for an original keyword.
        
        Args:
            original: The original keyword
            canonical_keywords: List of canonical keyword forms
            
        Returns:
            The best matching canonical form or the original if no match found
        """
        # If the original is already in the canonical list, use it
        if original in canonical_keywords:
            return original
            
        original_doc = self.nlp(original)
        if not original_doc.has_vector:
            return original
            
        # Compare embeddings to find the most similar canonical form
        best_similarity = -1
        best_match = original
        
        for canonical in canonical_keywords:
            canonical_doc = self.nlp(canonical)
            if canonical_doc.has_vector:
                similarity = original_doc.similarity(canonical_doc)
                if similarity > best_similarity and similarity >= self.config.get("canonicalization", {}).get("similarity_threshold", 0.85):
                    best_similarity = similarity
                    best_match = canonical
                    
        return best_match

    def _calc_metrics(self, chunk_results: Tuple[pd.DataFrame, pd.DataFrame]) -> Dict:
        start_time = time.time()
        summary, _ = chunk_results

        # Calculate recall against the *original* set of skills (before expansion)
        original_skills = set()
        for category_skills in self.config["keyword_categories"].values():
            original_skills.update(s.lower() for s in category_skills)

        # Calculate recall against the *expanded* set of skills
        expanded_skills = set(s.lower() for s in self.keyword_extractor.all_skills)

        # Extract lowercased keywords from summary
        extracted_keywords = set(summary.index.str.lower())

        # Calculate metrics
        original_recall = (
            len(extracted_keywords & original_skills) / len(original_skills)
            if original_skills
            else 0
        )

        expanded_recall = (
            len(extracted_keywords & expanded_skills) / len(expanded_skills)
            if expanded_skills
            else 0
        )

        # Assuming we're working with the expanded set for relevance judgment
        # For precision: what percentage of extracted keywords are in our expanded set
        precision = (
            len(extracted_keywords & expanded_skills) / len(extracted_keywords)
            if extracted_keywords
            else 0
        )

        # Calculate F1 score (harmonic mean of precision and expanded_recall)
        f1_score = (
            2 * precision * expanded_recall / (precision + expanded_recall)
            if precision + expanded_recall > 0
            else 0
        )

        # Prevent division by zero by checking both emptiness and length
        time_per_job = (
            (time.time() - start_time) / len(summary)
            if not summary.empty and len(summary) > 0
            else 0.5
        )

        return {
            "original_recall": original_recall,
            "expanded_recall": expanded_recall,  # This will be our primary recall metric
            "precision": precision,
            "f1_score": f1_score,
            "memory": psutil.virtual_memory().percent,
            "time_per_job": time_per_job,
        }

    def _calc_reward(self, metrics: Dict) -> float:
        weights = self.config["optimization"]["reward_weights"]
        scale = self.config["optimization"].get("memory_scale_factor", 100)

        # Use expanded_recall instead of recall in reward calculation
        return (
            metrics["expanded_recall"] * weights["recall"]
            - metrics["memory"] / scale * weights["memory"]
            - metrics["time_per_job"] * weights["time"]
        )

    def _aggregate_results(
        self, results: Union[List, Generator]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Aggregates results incrementally, handling generators and lists."""
        summary_agg = defaultdict(lambda: {"total": 0.0, "count": 0, "jobs": set()})
        details_list = []

        for summary_chunk, detail_chunk in results:  # Works directly with generator
            if summary_chunk.empty or detail_chunk.empty:
                logger.warning("Empty chunk encountered. Skipping.")
                continue

            for keyword, row in summary_chunk.iterrows():
                try:  # Error handling within the loop
                    total_score = float(row["Total_Score"])
                    job_count = int(row["Job_Count"])
                except (ValueError, TypeError) as e:
                    logger.error(f"Invalid data for {keyword}: {e}. Skipping row.")
                    continue

                summary_agg[keyword]["total"] += total_score
                summary_agg[keyword]["count"] += job_count
                summary_agg[keyword]["jobs"].update(
                    detail_chunk.loc[
                        detail_chunk["Keyword"] == keyword, "Job Title"
                    ].tolist()
                )

            details_list.extend(detail_chunk.to_dict("records"))
            del summary_chunk, detail_chunk  # Free memory

        # Create DataFrames at the end
        summary_df = pd.DataFrame.from_dict(
            {
                k: {
                    "Total_Score": v["total"],
                    "Avg_Score": v["total"] / v["count"] if v["count"] > 0 else 0,
                    "Job_Count": len(v["jobs"]),
                }
                for k, v in summary_agg.items()
            },
            orient="index",
        ).sort_values("Total_Score", ascending=False)

        details_df = pd.DataFrame(details_list)
        return summary_df, details_df

    def _validate_config(self):
        """
        Validates the configuration using the new validation utility.
        """
        # Import here to avoid circular imports
        from config_validation import validate_config

        try:
            validate_config(self.config)
        except ConfigError as e:
            logger.error(f"Configuration error: {e}")
            sys.exit(78)  # Configuration error exit code

    def _try_load_model(self, model_name, disabled_components):
        try:
            nlp = spacy.load(model_name, disable=disabled_components)
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
            if "lemmatizer" not in nlp.pipe_names:
                nlp.add_pipe("lemmatizer")
            return nlp
        except OSError:
            return None

    def _download_model(self, model_name):
        try:
            spacy.cli.download(model_name)
            return True
        except Exception as e:
            logger.warning(f"Download failed: {e}")
            return False

    def _load_and_configure_spacy_model(self):
        """Loads spaCy model with improved error handling and validation."""
        model_name = self.config["text_processing"]["spacy_model"]
        enabled_components = self.config["text_processing"]["spacy_pipeline"].get(
            "enabled_components", []
        )
        
        # Check GPU configuration
        use_gpu = self.config["hardware_limits"].get("use_gpu", True)
        if use_gpu:
            try:
                # Only try GPU if available
                if torch.cuda.is_available():
                    logger.info("GPU detected, enabling for spaCy if possible")
                    # Explicit CUDA memory check
                    free_mem = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
                    if free_mem < 1e9:  # Less than 1GB free
                        logger.warning(f"Low GPU memory ({free_mem/1e9:.2f}GB free), using CPU instead")
                        spacy.require_cpu()
                        use_gpu = False
                else:
                    logger.info("No GPU detected, using CPU")
                    use_gpu = False
                    spacy.require_cpu()
            except Exception as e:
                logger.warning(f"GPU detection failed: {e}. Falling back to CPU")
                use_gpu = False
                spacy.require_cpu()
        
        # Validate and update required components
        required_components = {"tok2vec", "tagger", "lemmatizer"}
        missing_components = required_components - set(enabled_components)
        if missing_components:
            logger.warning(
                f"Adding missing required components to pipeline: {missing_components}"
            )
            enabled_components.extend(missing_components)
        
        self.config["text_processing"]["spacy_pipeline"]["enabled_components"] = enabled_components
        
        # Try to load the model with retries
        try:
            nlp = self._load_model_with_retries(model_name, enabled_components, use_gpu)
            
            # Validate the loaded pipeline
            self._validate_spacy_pipeline(nlp, model_name)
            
            # Add any missing critical components
            self._add_essential_components(nlp, enabled_components)
            
            logger.info(
                f"Loaded spaCy model: {model_name} with pipeline: {nlp.pipe_names}, "
                f"GPU enabled: {use_gpu}"
            )
            return nlp
            
        except Exception as e:
            logger.critical(f"SpaCy model initialization failed: {e}")
            raise ConfigError(f"SpaCy initialization failed: {e}") from e
    
    def _load_model_with_retries(self, model_name, enabled_components, use_gpu):
        """Load model with retries and fallback to CPU if GPU fails."""
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                disabled_components = self._get_disabled_components(model_name, enabled_components)
                
                # Handle GPU-specific configuration before loading
                if use_gpu and attempt == 0:
                    # Set GPU allocator if available
                    if hasattr(spacy, "prefer_gpu") and callable(spacy.prefer_gpu):
                        spacy.prefer_gpu()
                
                # Try to load the model
                nlp = spacy.load(model_name, disable=disabled_components)
                
                # Successfully loaded
                return nlp
                
            except OSError as e:
                # Handle different error scenarios
                if "CUDA out of memory" in str(e) or "CUDNN_STATUS_NOT_INITIALIZED" in str(e):
                    if attempt < max_retries - 1:
                        logger.warning(f"GPU memory error, retrying with CPU (attempt {attempt + 1})")
                        spacy.require_cpu()
                        use_gpu = False
                        torch.cuda.empty_cache()
                        time.sleep(retry_delay)
                    else:
                        logger.error("GPU memory issues persisted after retries")
                        raise
                elif "E050" in str(e) and attempt == 0:
                    # Missing model error, try to download
                    logger.info(f"Model {model_name} not found, attempting download")
                    if self._download_model(model_name):
                        continue  # Try loading again
                    else:
                        raise ConfigError(f"Failed to download model '{model_name}'")
                elif attempt < max_retries - 1:
                    # Other errors, general retry
                    logger.warning(f"Error loading model (attempt {attempt + 1}): {e}")
                    time.sleep(retry_delay)
                else:
                    # Last attempt failed
                    logger.error(f"Failed to load model after {max_retries} attempts")
                    raise
    
    def _get_disabled_components(self, model_name, enabled_components):
        """Get components to disable based on enabled components and model info."""
        try:
            # Get model info to determine available components
            all_components = set()
            
            # Try to use spacy.info if available (spaCy v3+)
            try:
                model_info = spacy.info(model_name)
                all_components = set(model_info.get("pipeline", []))
            except (AttributeError, KeyError):
                # Fallback for older spaCy or if info not available
                logger.warning("Could not determine model pipeline components")
                
            # Calculate components to disable
            disabled = list(all_components - set(enabled_components))
            logger.debug(f"Disabling components: {disabled}")
            return disabled
            
        except Exception as e:
            logger.warning(f"Error determining components to disable: {e}")
            return []  # Safe fallback: disable nothing, let spaCy handle it
    
    def _validate_spacy_pipeline(self, nlp, model_name):
        """Validate the loaded spaCy pipeline against requirements."""
        # 1. Check for critical components
        essential_pipes = {"tagger", "lemmatizer"}
        missing_pipes = essential_pipes - set(nlp.pipe_names)
        
        if missing_pipes:
            logger.warning(
                f"Model {model_name} missing critical components: {missing_pipes}. "
                "Some functionality may be limited."
            )
        
        # 2. Validate POS tags if the model has a tagger
        if "tagger" in nlp.pipe_names and self.config["text_processing"].get("pos_filter"):
            try:
                model_tagger = nlp.get_pipe("tagger")
                configured_pos_tags = set(self.config["text_processing"].get("pos_filter", []))
                
                # Different ways to get tagger labels depending on spaCy version
                if hasattr(model_tagger, "labels"):
                    model_pos_tags = set(model_tagger.labels)
                else:
                    # Fallback for older spaCy
                    # Create a simple doc to extract POS tags
                    doc = nlp("The quick brown fox jumps over the lazy dog.")
                    model_pos_tags = {token.pos_ for token in doc}
                
                unsupported_tags = configured_pos_tags - model_pos_tags
                if unsupported_tags:
                    logger.warning(
                        f"Model {model_name} may not support these POS tags: {unsupported_tags}. "
                        f"Available tags: {model_pos_tags}. "
                        "POS filtering may not work as expected."
                    )
            except Exception as e:
                logger.warning(f"Could not validate POS tags: {e}")
    
    def _add_essential_components(self, nlp, enabled_components):
        """Add essential components and resolve dependencies."""
        # Define component dependencies
        component_deps = {
            'lemmatizer': ['tagger'],  
            'entity_ruler': ['tok2vec'],
            'sentencizer': [],  # No dependencies
            'ner': ['tok2vec'],
        }
        
        # Add sentencizer if needed and not present
        if "sentencizer" not in nlp.pipe_names and "sentencizer" in enabled_components:
            try:
                nlp.add_pipe("sentencizer")
                logger.info("Added sentencizer to pipeline")
            except Exception as e:
                logger.warning(f"Could not add sentencizer: {e}")
        
        # Add entity_ruler if needed and not present
        if "entity_ruler" not in nlp.pipe_names and "entity_ruler" in enabled_components:
            try:
                config = {"phrase_matcher_attr": "LOWER", "validate": True}
                # Check if 'ner' exists to decide where to add entity_ruler
                position = "before" if "ner" in nlp.pipe_names else "last"
                target = "ner" if position == "before" else None
                
                nlp.add_pipe("entity_ruler", config=config, before=target if position == "before" else None)
                logger.info("Added entity_ruler to pipeline")
            except Exception as e:
                logger.warning(f"Could not add entity_ruler: {e}")
        
        # Add lemmatizer if needed
        if "lemmatizer" not in nlp.pipe_names and "lemmatizer" in enabled_components:
            try:
                if "tagger" in nlp.pipe_names:  # Check dependency
                    nlp.add_pipe("lemmatizer")
                    logger.info("Added lemmatizer to pipeline")
                else:
                    logger.warning("Cannot add lemmatizer: missing dependency 'tagger'")
            except Exception as e:
                logger.warning(f"Could not add lemmatizer: {e}")

    def _download_model(self, model_name):
        """Download a spaCy model."""
        try:
            logger.info(f"Downloading spaCy model: {model_name}")
            spacy.cli.download(model_name)
            return True
        except Exception as e:
            logger.error(f"Error downloading model {model_name}: {e}")
            return False

    def _add_entity_ruler(self, nlp):
        """Add entity ruler with skill patterns and section headings to the pipeline."""
        if "entity_ruler" not in nlp.pipe_names:
            # Use configuration parameters if available
            config = {"phrase_matcher_attr": "LOWER", "validate": True}
            
            # Determine best position for the ruler
            if "ner" in nlp.pipe_names:
                ruler = nlp.add_pipe("entity_ruler", config=config, before="ner")
                logger.info("Added entity_ruler before NER component")
            else:
                ruler = nlp.add_pipe("entity_ruler", config=config)
                logger.info("Added entity_ruler to pipeline (NER not present)")
        else:
            ruler = nlp.get_pipe("entity_ruler")
            logger.info("Using existing entity_ruler component")
        
        # Create section patterns
        section_patterns = [
            {"label": "SECTION", "pattern": [{"LOWER": heading.lower()}]}
            for heading in self.config["advanced"].get("section_headings", [])
        ]
        
        # Create skill patterns
        unique_skills = set()
        for terms in self.config["keyword_categories"].values():
            unique_skills.update(terms)
        
        # Create and validate skill patterns
        skill_patterns = [
            {"label": "SKILL", "pattern": [{"LOWER": skill.lower()}]}
            for skill in unique_skills
        ]
        
        # Validate patterns
        valid_section_patterns = [p for p in section_patterns if self._validate_pattern(p)]
        valid_skill_patterns = [p for p in skill_patterns if self._validate_pattern(p)]
        
        # Log validation results
        skipped_section_count = len(section_patterns) - len(valid_section_patterns)
        skipped_skill_count = len(skill_patterns) - len(valid_skill_patterns)
        
        if skipped_section_count > 0:
            logger.warning(f"Skipped {skipped_section_count} invalid section patterns")
        
        if skipped_skill_count > 0:
            if skipped_skill_count < 10 or skipped_skill_count < len(skill_patterns) * 0.05:
                logger.warning(f"Skipped {skipped_skill_count} invalid skill patterns")
            else:
                # More significant warning for larger numbers of invalid patterns
                logger.error(
                    f"High number of invalid skill patterns: {skipped_skill_count} "
                    f"({skipped_skill_count/len(skill_patterns):.1%}). Check skill data source."
                )
        
        # Add patterns to ruler
        ruler.add_patterns(valid_section_patterns + valid_skill_patterns)
        
        # Diagnostic logging
        logger.info(
            f"Added {len(valid_section_patterns)} section patterns and "
            f"{len(valid_skill_patterns)} skill patterns to entity ruler"
        )
        
        # Check ruler position for optimal performance
        if "ner" in nlp.pipe_names:
            try:
                ruler_idx = nlp.pipe_names.index("entity_ruler")
                ner_idx = nlp.pipe_names.index("ner")
                if ruler_idx > ner_idx:
                    logger.warning(
                        "Entity ruler positioned after NER component. "
                        "For best results, entity_ruler should be before NER. "
                        "Consider reinitializing with correct pipeline order."
                    )
            except ValueError:
                # This should not happen as we already checked for presence
                pass

    def _validate_pattern(self, pattern: dict) -> bool:
        """
        Validate a spaCy entity ruler pattern structure.
        
        Args:
            pattern: The pattern dictionary to validate
            
        Returns:
            bool: True if pattern is valid, False otherwise
        """
        try:
            # Check basic structure
            if not isinstance(pattern, dict):
                return False
            
            # Check required keys
            if not all(key in pattern for key in ["label", "pattern"]):
                return False
                
            # Validate label
            if not isinstance(pattern["label"], str) or not pattern["label"].strip():
                return False
                
            # Validate pattern
            if not isinstance(pattern["pattern"], list) or not pattern["pattern"]:
                return False
                
            # Validate each token dict in the pattern
            for token in pattern["pattern"]:
                # Each token must be a non-empty dict with at least one attribute
                if not isinstance(token, dict) or not token:
                    return False
                    
                # At least one valid matcher attribute should be present
                valid_attrs = {"TEXT", "LOWER", "LEMMA", "POS", "TAG", "DEP", "OP", "IS_SENT_START"}
                if not any(attr in token for attr in valid_attrs):
                    return False
                    
            return True
            
        except (KeyError, TypeError, AttributeError) as e:
            logger.debug(f"Pattern validation error: {e}")
            return False

    def _init_categories(self):
        self.keyword_extractor._init_categories()

    def _save_intermediate(
        self,
        batch_idx: int,
        summary_chunks: List[pd.DataFrame],
        details_chunks: List[pd.DataFrame],
    ):
        logger.info("Starting intermediate save", batch_idx=batch_idx)
        if not self.config["intermediate_save"]["enabled"]:
            return

        format_type = self.config["intermediate_save"]["format"]
        suffix = {"feather": ".feather", "jsonl": ".jsonl", "json": ".json"}.get(
            format_type, ".json"
        )
        summary_path = (
            self.working_dir / f"{self.run_id}_chunk_summary_{batch_idx}{suffix}"
        )
        details_path = (
            self.working_dir / f"{self.run_id}_chunk_details_{batch_idx}{suffix}"
        )

        checksums = {}  # Dictionary to store checksums

        def save_and_verify(path, data, save_func, append=False, max_retries=3):
            for attempt in range(max_retries):
                try:
                    save_func(path, data, append)
                    checksum = self._calculate_file_checksum(path)
                    if self._verify_single_checksum(path, checksum):
                        return checksum
                    logger.warning(
                        f"Checksum failed for {path} (attempt {attempt + 1}/{max_retries})"
                    )
                except Exception as e:
                    logger.error(f"Save failed for {path}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
            raise DataIntegrityError(
                f"Failed to save and verify {path} after {max_retries} attempts"
            )

        try:
            if format_type == "feather":
                for i, (summary_chunk, details_chunk) in enumerate(
                    zip(summary_chunks, details_chunks)
                ):
                    summary_chunk = summary_chunk.reset_index()
                    details_chunk = details_chunk.reset_index()

                    if i == 0:  # First chunk: create the file
                        checksums[str(summary_path)] = save_and_verify(
                            summary_path,
                            summary_chunk,
                            lambda p, d, a: feather.write_feather(d, p),
                            append=False,
                        )
                        checksums[str(details_path)] = save_and_verify(
                            details_path,
                            details_chunk,
                            lambda p, d, a: feather.write_feather(d, p),
                            append=False,
                        )
                    else:  # Subsequent chunks: append, with fallback
                        def append_to_feather(path, df, append):
                            try:
                                if append:
                                    # Read existing data, concat, then overwrite
                                    existing_df = pd.read_feather(path)
                                    new_df = pd.concat([existing_df, df], ignore_index=True)
                                    feather.write_feather(new_df, path)
                                else:
                                     feather.write_feather(df, path)

                            except Exception as e:
                                logger.error(f"Error appending to Feather file {path}: {e}")
                                # Fallback: Create a new file with .part{i} suffix
                                fallback_path = Path(str(path) + f".part{i}")
                                logger.warning(
                                    f"Appending to {path} failed. Creating fallback file at {fallback_path}"
                                )
                                feather.write_feather(df, fallback_path)
                                return xxhash.xxh3_64(str(fallback_path).encode()).hexdigest()  # Unique ID
                            return self._calculate_file_checksum(path)
                        checksums[str(summary_path)] = save_and_verify(
                            summary_path,
                            summary_chunk,
                            lambda p, d, a: append_to_feather(p, d, a),
                            append=True
                        )

                        checksums[str(details_path)] = save_and_verify(
                            details_path,
                            details_chunk,
                            lambda p, d, a: append_to_feather(p, d, a),
                            append=True
                        )

            elif format_type == "jsonl":
                for i, (summary_chunk, details_chunk) in enumerate(
                    zip(summary_chunks, details_chunks)
                ):
                    append = i > 0
                    checksums[str(summary_path)] = save_and_verify(
                        summary_path,
                        (r.to_dict() for _, r in summary_chunk.iterrows()),
                        lambda p, d, a: srsly.write_jsonl(p, d, append=a),
                        append=append,
                    )
                    checksums[str(details_path)] = save_and_verify(
                        details_path,
                        (r.to_dict() for _, r in details_chunk.iterrows()),
                        lambda p, d, a: srsly.write_jsonl(p, d, append=a),
                        append=append,
                    )
            else:  # json format
                combined_summary = pd.concat(summary_chunks).reset_index()
                combined_details = pd.concat(details_chunks).reset_index()

                checksums[str(summary_path)] = save_and_verify(
                    summary_path,
                    combined_summary.to_dict(),
                    lambda p, d, a: srsly.write_json(p, d),
                    append=False,
                )
                checksums[str(details_path)] = save_and_verify(
                    details_path,
                    combined_details.to_dict(),
                    lambda p, d, a: srsly.write_json(p, d),
                    append=False,
                )

            logger.info(
                "Intermediate results saved",
                batch_idx=batch_idx,
                working_dir=str(self.working_dir),
            )

            # Save the checksums to a manifest file
            self._save_checksum_manifest(checksums)

        except Exception as e:
            logger.error(f"Failed to save intermediate results: {e}")

    def _verify_single_checksum(self, file_path: Path, expected_checksum: str) -> bool:
        if not file_path.exists():
            logger.error("Intermediate file not found", file_path=str(file_path))
            return False
        calculated_hash = self._calculate_file_checksum(file_path)
        if calculated_hash != expected_checksum:
            logger.error(
                "Checksum mismatch",
                file_path=str(file_path),
                expected_checksum=expected_checksum,
                calculated_checksum=calculated_hash,
            )
            return False
        logger.info("Checksum verified", file_path=str(file_path))
        return True

    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculates the XXH3_128 hash of a file."""
        hasher = xxhash.xxh3_128()
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(8192)  # Read in chunks for efficiency
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()

    def _save_checksum_manifest(self, checksums: Dict[str, str]):
        """Saves the checksum manifest to a JSONL file."""
        try:
            with open(self.checksum_manifest_path, "a") as f:  # Append mode
                for file_path, checksum in checksums.items():
                    srsly.write_jsonl(
                        f, [{"file": file_path, "checksum": checksum}]
                    )  # Write line
        except Exception as e:
            logger.error(f"Failed to save checksum manifest: {e}")
            # Consider raising a custom exception or handling differently

    def _verify_intermediate_checksums(self) -> bool:
        """Verifies the checksums of all intermediate files."""
        if not self.checksum_manifest_path.exists():
            logger.warning("Checksum manifest file not found. Skipping verification.")
            return True  # Or False, depending on whether you want to proceed

        try:
            stored_checksums = {}  # Use a dictionary
            for line in srsly.read_jsonl(self.checksum_manifest_path):
                stored_checksums[line["file"]] = line["checksum"]

        except Exception as e:
            logger.error(f"Failed to load checksum manifest: {e}")
            return False  # Or raise an exception

        for file_path_str, stored_hash in stored_checksums.items():
            file_path = Path(file_path_str)
            if not file_path.exists():
                logger.error(f"Intermediate file not found: {file_path}")
                raise DataIntegrityError(f"Intermediate file not found: {file_path}")

            calculated_hash = self._calculate_file_checksum(file_path)
            if calculated_hash != stored_hash:
                logger.error(
                    f"Checksum mismatch for {file_path}: "
                    f"expected {stored_hash}, got {calculated_hash}"
                )
                raise DataIntegrityError(
                    f"Checksum mismatch for {file_path}: "
                    f"expected {stored_hash}, got {calculated_hash}"
                )
            else:
                logger.info(f"Checksum verified for: {file_path}")

        logger.info("All intermediate file checksums verified.")
        return True

    def _load_all_intermediate(
        self, batch_count: int
    ) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
        logger.info("Loading all intermediate results", batch_count=batch_count)
        """Loads intermediate results, yielding DataFrames for each batch."""
        format_type = self.config["intermediate_save"]["format"]
        suffix = {"feather": ".feather", "jsonl": ".jsonl", "json": ".json"}.get(
            format_type, ".json"
        )
        for i in range(batch_count):
            try:
                summary_path = (
                    self.working_dir / f"{self.run_id}_chunk_summary_{i}{suffix}"
                )
                details_path = (
                    self.working_dir / f"{self.run_id}_chunk_details_{i}{suffix}"
                )

                if summary_path.exists() and details_path.exists():
                    # Check file sizes:
                    if (
                        summary_path.stat().st_size == 0
                        or details_path.stat().st_size == 0
                    ):
                        logger.warning(
                            f"Empty intermediate file(s) found for batch {i}: {summary_path}, {details_path}"
                        )

                        # In strict mode, we might want to raise an error instead
                        if self.config.get("strict_mode", False):
                            raise DataIntegrityError(
                                f"Empty intermediate file(s) detected for batch {i}"
                            )

                        # Otherwise, yield empty DataFrames
                        yield pd.DataFrame(), pd.DataFrame()
                        continue

                    try:
                        if format_type == "feather":
                            summary = pd.read_feather(summary_path)
                            details = pd.read_feather(details_path)
                        elif format_type == "jsonl":
                            summary = pd.DataFrame(list(srsly.read_jsonl(summary_path)))
                            details = pd.DataFrame(list(srsly.read_jsonl(details_path)))
                        else:  # json
                            summary = pd.DataFrame(srsly.read_json(summary_path))
                            details = pd.DataFrame(srsly.read_json(details_path))

                        # Verify dataframes aren't empty after loading
                        if summary.empty or details.empty:
                            logger.warning(f"Loaded empty DataFrame(s) from batch {i}")

                        # Apply consistent data types to columns
                        if not summary.empty and "Total_Score" in summary:
                            summary["Total_Score"] = summary["Total_Score"].astype(
                                float
                            )
                        if not summary.empty and "Job_Count" in summary:
                            summary["Job_Count"] = summary["Job_Count"].astype(int)

                        logger.info("Loaded intermediate results", batch_idx=i)
                        yield summary, details
                    except (
                        FileNotFoundError,
                        IOError,
                        pa.ArrowInvalid,
                    ) as e:  # Specific exceptions
                        logger.error(f"File error in batch {i}: {e}")
                        if self.config.get("strict_mode", False):
                            raise
                        yield pd.DataFrame(), pd.DataFrame()  # Yield empty DataFrames
                else:
                    # If we can't find both files, yield empty DataFrames
                    logger.warning(
                        f"Missing intermediate files for batch {i}: {summary_path}, {details_path}"
                    )
                    if self.config.get("strict_mode", False):
                        raise FileNotFoundError(
                            f"Missing intermediate files for batch {i}: {summary_path}, {details_path}"
                        )
                    yield pd.DataFrame(), pd.DataFrame()
            except Exception as e:
                # Top-level exception handler to catch any other errors
                logger.exception(f"Unexpected error processing batch {i}: {e}")
                if self.config.get("strict_mode", False):
                    raise  # Re-raise exception in strict mode
                yield pd.DataFrame(), pd.DataFrame()  # Keep the generator running

    def _aggregate_results(
        self, results: Union[List, Generator]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Aggregates results incrementally, handling generators and lists."""
        summary_agg = defaultdict(lambda: {"total": 0.0, "count": 0, "jobs": set()})
        details_list = []

        for summary_chunk, detail_chunk in results:  # Works directly with generator
            if summary_chunk.empty or detail_chunk.empty:
                logger.warning("Empty chunk encountered. Skipping.")
                continue

            for keyword, row in summary_chunk.iterrows():
                try:  # Error handling within the loop
                    total_score = float(row["Total_Score"])
                    job_count = int(row["Job_Count"])
                except (ValueError, TypeError) as e:
                    logger.error(f"Invalid data for {keyword}: {e}. Skipping row.")
                    continue

                summary_agg[keyword]["total"] += total_score
                summary_agg[keyword]["count"] += job_count
                summary_agg[keyword]["jobs"].update(
                    detail_chunk.loc[
                        detail_chunk["Keyword"] == keyword, "Job Title"
                    ].tolist()
                )

            details_list.extend(detail_chunk.to_dict("records"))
            del summary_chunk, detail_chunk  # Free memory

        # Create DataFrames at the end
        summary_df = pd.DataFrame.from_dict(
            {
                k: {
                    "Total_Score": v["total"],
                    "Avg_Score": v["total"] / v["count"] if v["count"] > 0 else 0,
                    "Job_Count": len(v["jobs"]),
                }
                for k, v in summary_agg.items()
            },
            orient="index",
        ).sort_values("Total_Score", ascending=False)

        details_df = pd.DataFrame(details_list)
        return summary_df, details_df

    def _cleanup_intermediate(self):
        if self.config["intermediate_save"]["cleanup"]:
            try:
                shutil.rmtree(self.working_dir)
                logger.info(
                    "Cleaned up intermediate directory",
                    working_dir=str(self.working_dir),
                )
            except (OSError, IOError, PermissionError) as e:
                logger.error("Failed to cleanup intermediate directory", error=str(e))

    def _create_tfidf_matrix(self, keyword_sets_tuples_generator):
        """Creates TF-IDF matrix using memory-efficient HashingVectorizer. Accepts a generator."""
        max_features = self.config["caching"].get("tfidf_max_features", 10000)

        # Process the generator safely
        keyword_sets_list = []
        indices = []
        
        try:
            for i, (original_tokens, expanded_keywords) in enumerate(keyword_sets_tuples_generator):
                combined = list(set(original_tokens + expanded_keywords))
                if combined:  # Only add non-empty keyword sets
                    keyword_sets_list.append(combined)
                    indices.append(i)

            if not keyword_sets_list:
                logger.warning("Empty keyword sets for HashingVectorizer")
                return None, []

            # Check memory before proceeding
            mem_usage = psutil.virtual_memory().percent
            memory_threshold = self.config["hardware_limits"].get("memory_threshold", 70)
            if mem_usage > memory_threshold:
                logger.warning(f"Memory usage high ({mem_usage}%) before HashingVectorizer processing")
                gc.collect()  # Attempt to free memory

            # Initialize HashingVectorizer if not already done
            if not hasattr(self, "hashing_vectorizer"):
                self.hashing_vectorizer = HashingVectorizer(
                    ngram_range=self.keyword_extractor.ngram_range,
                    n_features=max_features,
                    dtype=np.float32,
                    lowercase=False,  # Already lowercased during preprocessing
                    tokenizer=lambda x: x,  # Pass tokens directly
                    preprocessor=lambda x: x,  # No additional preprocessing
                    norm='l2',  # Ensure consistent normalization
                    alternate_sign=False,  # Prevent feature cancellation
                )

            # Process all at once for efficiency
            dtm = self.hashing_vectorizer.transform(keyword_sets_list)
            
            # Return matrix and indices (feature names not available with HashingVectorizer)
            return dtm, indices

        except MemoryError as e:
            logger.error(f"Memory error in HashingVectorizer: {e}")
            gc.collect()
            return None, []
        except Exception as e:
            logger.exception(f"Unexpected error in HashingVectorizer processing: {e}")
            return None, []

    def _calculate_scores(self, dtm, feature_names, keyword_sets_generator, job_descriptions):
        """Calculates scores with hashing vectorizer support."""
        if dtm is None:
            logger.error("DTM is None, cannot calculate scores")
            return
            
        try:
            # Convert to COO format for efficient iteration
            dtm_coo = dtm.tocoo()
            job_descriptions_list = list(job_descriptions.items())
            weighting = self.config.get(
                "weighting",
                {"tfidf_weight": 0.7, "frequency_weight": 0.3, "whitelist_boost": 1.5},
            )

            # Build lookup dictionaries for efficient processing
            keyword_sets_dict, index_to_keywords = self._build_keyword_lookups(keyword_sets_generator)
            
            # Track processed terms to avoid duplicates
            processed_terms_by_job = defaultdict(set)
            
            # Process the sparse matrix entries
            for row, col, value in zip(dtm_coo.row, dtm_coo.col, dtm_coo.data):
                try:
                    job_index = row
                    
                    # Skip if job index is out of bounds
                    if job_index >= len(job_descriptions_list):
                        continue
                        
                    job_title, job_text = job_descriptions_list[job_index]
                    
                    # Skip if we don't have keywords for this job
                    if job_index not in keyword_sets_dict:
                        continue

                    original_tokens, expanded_keywords = keyword_sets_dict[job_index]
                    
                    # Process each term from this job's keywords
                    for term in self._get_job_terms(original_tokens, expanded_keywords):
                        # Skip duplicates within the same job
                        if term in processed_terms_by_job[job_index]:
                            continue
                            
                        processed_terms_by_job[job_index].add(term)
                        
                        # Generate result for this term and yield it
                        result = self._generate_term_result(
                            term, job_title, job_text, original_tokens, value, weighting
                        )
                        yield result

                except Exception as e:
                    logger.error(f"Error calculating score for row={row}, col={col}: {e}")
                    continue

        except Exception as e:
            logger.exception(f"Critical error in calculate_scores: {e}")
            return

    def _build_keyword_lookups(self, keyword_sets_generator):
        """Builds lookup dictionaries for keywords."""
        keyword_sets_dict = {}
        index_to_keywords = {}
        
        for i, (original_tokens, expanded_keywords) in enumerate(keyword_sets_generator):
            combined = list(set(original_tokens + expanded_keywords))
            for term in combined:
                # Store each term with its job index
                if term not in index_to_keywords:
                    index_to_keywords[term] = []
                index_to_keywords[term].append(i)
            keyword_sets_dict[i] = (original_tokens, expanded_keywords)
            
        return keyword_sets_dict, index_to_keywords
        
    def _get_job_terms(self, original_tokens, expanded_keywords):
        """Get combined list of unique terms for a job."""
        return list(set(original_tokens + expanded_keywords))

    def _generate_term_result(self, term, job_title, job_text, original_tokens, tfidf_value, weighting):
        """Generate result dictionary for a term."""
        # Calculate base score
        presence = 1 if term in original_tokens else 0
        score = self._compute_base_score(tfidf_value, presence, weighting)
        
        # Apply whitelist boost
        score = self._apply_whitelist_boost(term, score, weighting)
        
        # Apply section weights
        score = self._apply_section_weight(term, job_text, score)
        
        # Create and return the result dictionary
        return {
            "Keyword": term,
            "Job Title": job_title,
            "Score": score,
            "TF-IDF": tfidf_value,
            "Frequency": presence,
            "Category": self._categorize_term(term),
            "In Whitelist": term.lower() in self.keyword_extractor.all_skills,
        }
        
    def _compute_base_score(self, tfidf_value, presence, weighting):
        """Compute base score from TF-IDF and presence."""
        return (
            weighting.get("tfidf_weight", 0.7) * tfidf_value
            + weighting.get("frequency_weight", 0.3) * presence
        )
        
    def _apply_whitelist_boost(self, term, score, weighting):
        """Apply whitelist boost if term is in whitelist."""
        term_lower = term.lower()
        if term_lower in self.keyword_extractor.all_skills:
            return score * weighting.get("whitelist_boost", 1.5)
        return score
        
    def _apply_section_weight(self, term, job_text, score):
        """Apply section-specific weight modifier to score."""
        section_weights = self.config["weighting"].get("section_weights", {})
        section = self.keyword_extractor._detect_keyword_section(term, job_text)
        return score * section_weights.get(
            section,
            section_weights.get("default", 1.0),
        )
        
    def _categorize_term(self, term):
        """Categorize a term according to the defined categories."""
        return self.keyword_extractor._semantic_categorization(term)

def parse_arguments():
    parser = argparse.ArgumentParser(description="ATS Keyword Optimizer")
    parser.add_argument(
        "-i", "--input", default="job_descriptions.json", help="Input JSON file"
    )
    parser.add_argument("-c", "--config", default="config.yaml", help="Config file")
    parser.add_argument("-o", "--output", default="results.xlsx", help="Output file")
    return parser.parse_args()  # Fixed: changed parser.args() to parser.parse_args()


def initialize_analyzer(config_path: str):
    ensure_nltk_resources()
    analyzer = OptimizedATS(config_path)
    return analyzer


def save_results(summary: pd.DataFrame, details: pd.DataFrame, output_file: str):
    try:
        working_dir = Path(output_file).parent
        if shutil.disk_usage(working_dir).free < 1000000000:
            logger.warning("Low disk space in working directory")

        # Use a temporary file for atomic write
        with tempfile.NamedTemporaryFile(
            mode="w+b", suffix=".tmp", dir=working_dir, delete=False
        ) as temp_file:
            temp_filepath = Path(temp_file.name)
            try:
                with pd.ExcelWriter(temp_filepath) as writer:
                    summary.to_excel(writer, sheet_name="Summary")
                    details.to_excel(writer, sheet_name="Detailed Scores")
                # Atomically rename the temporary file to the final file
                os.replace(temp_filepath, output_file)  # Use os.replace for atomicity
                logger.info("Analysis complete. Results saved to %s", output_file)

            except OSError as e:
                if e.errno == errno.ENOSPC:  # No space left on device
                    logger.error(f"Out of disk space while writing to {temp_filepath}")
                    # Consider deleting the temporary file here
                    temp_filepath.unlink(missing_ok=True)
                    raise  # Re-raise the exception
                else:
                    logger.exception(
                        "Failed to save results to temporary file %s: %s",
                        temp_filepath,
                        e,
                    )
                    raise
    except Exception as e:
        logger.exception("Failed to save results to %s: %s", output_file, e)
        raise


def load_job_data(input_file: str) -> Dict:
    """
    Load job description data from a JSON file.

    Args:
        input_file: Path to the JSON file containing job descriptions.

    Returns:
        Dict: Dictionary containing job description data.

    Raises:
        FileNotFoundError: If the input file doesn't exist.
        json.JSONDecodeError: If the input file contains invalid JSON.
        Exception: For other unexpected errors.
    """
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error("Input file not found: %s", input_file)
        raise
    except json.JSONDecodeError as e:
        logger.error(
            "Invalid JSON in %s (line %s, column %s): %s",
            input_file,
            e.lineno,
            e.colno,
            e.msg,
        )
        raise
    except Exception as e:
        logger.exception("Unexpected error loading %s: %s", input_file, e)
        raise


def run_analysis(args):
    analyzer = initialize_analyzer(args.config)
    jobs = load_job_data(args.input)
    try:
        # 1. Analyze jobs (this will save intermediate files and checksums if enabled)
        analyzer.analyze_jobs(jobs)

        # 2. Determine the number of batches
        batch_count = 0
        if analyzer.config["intermediate_save"]["enabled"]:
            format_type = analyzer.config["intermediate_save"]["format"]
            suffix = {"feather": ".feather", "jsonl": ".jsonl", "json": ".json"}.get(
                format_type, ".json"
            )
            while (
                analyzer.working_dir
                / f"{analyzer.run_id}_chunk_summary_{batch_count}{suffix}"
            ).exists():
                batch_count += 1

            # 3. Verify checksums BEFORE loading
            analyzer._verify_intermediate_checksums()

        # 4. Load all intermediate results as a generator
        loaded_results_generator = analyzer._load_all_intermediate(batch_count)

        # 5. Aggregate the results using streaming aggregation
        final_summary, final_details = analyzer._aggregate_results(
            loaded_results_generator
        )

        # 6. Save to Excel
        save_results(final_summary, final_details, args.output)

    except DataIntegrityError as e:
        logger.error("Data integrity error: %s", e)
        sys.exit(75)  # Use a specific exit code for data integrity issues
    finally:
        analyzer._cleanup_intermediate()


def main():
    """
    Entry point function for the program.

    Validates Python version, parses command line arguments, and runs the analysis.
    Handles various exceptions with appropriate error messages and exit codes:
    - ConfigError (exit: 78): Configuration-related errors
    - InputValidationError (exit: 77): Input validation failures
    - MemoryError (exit: 70): Memory allocation failures
    - MPTimeoutError (exit: 73): Timeout errors during processing
    - Other exceptions (exit: 1): Unhandled exceptions

    Requires Python 3.8+

    Returns:
        None

    Exits with status codes:
        1: General error or Python version below 3.8
        70: Memory error
        73: Timeout error
        77: Input validation error
        78: Configuration error
    """
    if sys.version_info < (3, 8):
        logger.error("Requires Python 3.8+")
        sys.exit(1)
    args = parse_arguments()
    try:
        run_analysis(args)
    except ConfigError as e:
        logger.error("Configuration error: %s", e)
        sys.exit(78)
    except InputValidationError as e:
        logger.error("Input validation error: %s", e)
        sys.exit(77)
    except MemoryError as e:
        logger.error("Memory error: %s", e)
        sys.exit(70)
    except MPTimeoutError as e:
        logger.error("Timeout error: %s", e)
        sys.exit(73)
    except Exception as e:
        logger.exception("Unhandled exception: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
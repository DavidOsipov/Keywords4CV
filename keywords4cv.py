# Version 0.24

import argparse
import json
import logging
import re
import sys
import time
import random
import gc
import shutil
from collections import OrderedDict, deque, defaultdict
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

# Import the new exceptions module
from exceptions import (
    ConfigError,
    InputValidationError,
    CriticalFailureError,
    AggregationError,
    DataIntegrityError,
)
from pathlib import Path
from functools import lru_cache
from multiprocessing import TimeoutError as MPTimeoutError, current_process
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor

import nltk
import pandas as pd
import spacy
import yaml
import numpy as np
import psutil
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from rapidfuzz import process, fuzz
import srsly
import xxhash
from nltk.corpus import wordnet as wn
from itertools import product
from cachetools import LRUCache
from pydantic import BaseModel, Field, ValidationError, field_validator, conlist
import pyarrow.feather as feather  # Added import for pyarrow.feather
import pyarrow as pa
import pyarrow.parquet as pq
import os  # Ensure os is imported for environment variable access

logger = logging.getLogger(__name__)

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
        config_path = Path(config_path)
        with open(config_path, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)

        # Validate and convert to a dictionary
        config = Config(**raw_config)
        return config.dict(by_alias=True)  # Use alias for serialization

    except FileNotFoundError:
        logger.error("Config file not found: %s", config_path)
        sys.exit(78)  # Configuration error exit code
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
        self.regex_patterns = {
            "url": re.compile(r"http\S+|www\.\S+"),
            "email": re.compile(r"\S+@\S+"),
            "special_chars": re.compile(r"[^\w\s'\-]"),
            "whitespace": re.compile(r"\s+"),
        }
        self._cache = OrderedDict()
        base_cache_size = self.config["caching"].get("cache_size", 5000)

        # Fix: Consistently access memory_scaling_factor from hardware_limits
        scaling_factor = self.config["hardware_limits"].get(
            "memory_scaling_factor", 0.3
        )

        if scaling_factor:
            available_mb = psutil.virtual_memory().available / (1024 * 1024)
            dynamic_size = int(available_mb / scaling_factor)
            self._CACHE_SIZE = min(base_cache_size, dynamic_size)
        else:
            self._CACHE_SIZE = base_cache_size
        self.config_hash = None
        self._update_config_hash()
        self.pos_processing = self.config["text_processing"].get("pos_processing", "")
        self.cache_salt = get_cache_salt(
            self.config
        )  # Get cache salt with proper priority

    def _update_config_hash(self):
        new_hash = self._calculate_config_hash()
        if new_hash != self.config_hash:
            self.config_hash = new_hash
            self._cache.clear()

    def _calculate_config_hash(self) -> str:
        relevant_config = {
            "stop_words": self.config.get("stop_words", []),
            "stop_words_add": self.config.get("stop_words_add", []),
            "stop_words_exclude": self.config.get("stop_words_exclude", []),
            "keyword_categories": self.config.get(
                "keyword_categories", {}
            ),  # Add keyword categories
        }
        config_str = json.dumps(relevant_config, sort_keys=True)
        # Use self.cache_salt to salt the hash
        return xxhash.xxh3_64(
            f"{self.cache_salt}_{config_str}".encode("utf-8")
        ).hexdigest()

    def _load_stop_words(self) -> Set[str]:
        stop_words = set(self.config.get("stop_words", []))
        stop_words.update(self.config.get("stop_words_add", []))
        stop_words.difference_update(self.config.get("stop_words_exclude", []))

        if len(stop_words) < 50:
            logger.warning(
                "Stop words list seems unusually small (less than 50 words). Consider adding more stop words to improve text preprocessing."
            )
        return stop_words

    def preprocess(self, text: str) -> str:
        self._update_config_hash()
        # Use self.cache_salt to salt the text hash
        text_hash = f"{CACHE_VERSION}_{xxhash.xxh3_64((self.cache_salt + text).encode()).hexdigest()}"
        if text_hash in self._cache:
            self._cache.move_to_end(text_hash)
            return self._cache[text_hash]

        cleaned = text.lower()
        cleaned = self.regex_patterns["url"].sub("", cleaned)
        cleaned = self.regex_patterns["email"].sub("", cleaned)
        cleaned = self.regex_patterns["special_chars"].sub(" ", cleaned)
        cleaned = self.regex_patterns["whitespace"].sub(" ", cleaned).strip()

        while len(self._cache) >= self._CACHE_SIZE:
            self._cache.popitem(last=False)

        self._cache[text_hash] = cleaned
        return cleaned

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        return [self.preprocess(text) for text in texts]

    def _process_doc_tokens(self, doc):
        tokens = [ent.text.lower() for ent in doc.ents if ent.label_ == "SKILL"]
        skill_spans = [
            (ent.start, ent.end) for ent in doc.ents if ent.label_ == "SKILL"
        ]
        for i, token in enumerate(doc):
            if any(start <= i < end for start, end in skill_spans):
                continue
            if len(token.strip()) > 1 and token not in self.preprocessor.stop_words:
                tokens.append(token.lemma_.lower())
        return list(set(tokens))  # Deduplicate for efficiency

    def tokenize_batch(self, texts: List[str]) -> Generator[List[str], None, None]:
        """Tokenizes a batch of texts using spaCy.

        Uses nlp.pipe for efficient batch processing, and dynamically enables
        components. Employs a generator for memory efficiency.

        Args:
            texts: A list of text strings.

        Yields:
            A list of tokens for each input text.
        """
        batch_size = self.config["hardware_limits"].get("batch_size", 64)
        # Use configured max_workers, but ensure it's at least 1
        max_workers = max(1, self.config["hardware_limits"].get("max_workers", 1))

        enabled_pipes = self.config["text_processing"]["spacy_pipeline"].get(
            "enabled_components", []
        )
        if not enabled_pipes:
            enabled_pipes = ["tok2vec", "tagger", "lemmatizer", "entity_ruler"]

        try:
            with self.nlp.select_pipes(enable=enabled_pipes):
                for i, doc in enumerate(
                    self.nlp.pipe(texts, batch_size=batch_size, n_process=max_workers)
                ):
                    try:
                        yield self._process_doc_tokens(doc)
                        del doc  # Explicitly release memory after processing EACH doc
                    except (ValueError, AttributeError, RuntimeError, TypeError) as e:
                        logger.error("Error tokenizing text #%s: %s", i, e)
                        yield []  # Return empty token list as fallback

        except (ValueError, IOError, RuntimeError, OSError) as e:
            logger.exception("Error during tokenization batch process: %s", e)
            for _ in texts:  # Still yield something for each input text
                yield []


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

    def __init__(self, config: Dict, nlp):
        self.config = config
        self.nlp = nlp
        self.phrase_synonyms = self._load_phrase_synonyms()
        self.api_cache: Dict[str, List[str]] = {}
        self.all_skills = self._load_and_process_all_skills()
        self.preprocessor = EnhancedTextPreprocessor(config, nlp)
        self.category_vectors = {}
        self.ngram_range = self.config["text_processing"].get("ngram_range", [1, 3])
        self._section_heading_re = re.compile(
            r"^(?:"
            + "|".join(
                re.escape(heading)
                for heading in self.config["advanced"].get("section_headings", [])
            )
            + r")(?:\s*:)?",
            re.MULTILINE | re.IGNORECASE,
        )
        self.default_category = config["categorization"].get(
            "default_category", "Other"
        )
        self.category_vectors = {}
        self.cache_size = self.config["caching"].get("cache_size", 5000)

    @lru_cache(maxsize=1)  # Cache the validated synonyms
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
                    validated_synonyms[entry.term] = [
                        s.lower() for s in entry.synonyms
                    ]  # Lowercase
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
            # 1. Preprocess (lowercase, clean)
            cleaned_skill = self.preprocessor.preprocess(skill)

            # 2. Tokenize (using spaCy for consistency)
            doc = self.nlp(cleaned_skill)
            tokens = [
                token.lemma_.lower()
                for token in doc
                if token.text.lower() not in self.preprocessor.stop_words
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
        synonyms = set()
        for skill in skills:
            doc = self.nlp(skill)
            lemmatized = " ".join([token.lemma_ for token in doc]).lower()
            if lemmatized != skill.lower():  # Consistent lowercasing
                synonyms.add(lemmatized)

            for token in doc:
                pos_tag = token.pos_
                wn_pos = self._convert_spacy_to_wordnet_pos(pos_tag)
                if wn_pos is None:
                    continue

                for syn in wn.synsets(token.text, pos=wn_pos):
                    for lemma in syn.lemmas():
                        synonym = (
                            lemma.name().replace("_", " ").lower()
                        )  # Consistent lowercasing
                        if synonym not in (token.text.lower(), lemmatized):
                            synonyms.add(synonym)

            # Add phrase-level synonyms
            source = self.config["text_processing"]["phrase_synonym_source"]
            skill_lower = skill.lower()  # Use lowercased skill
            if source == "static":
                if skill_lower in self.phrase_synonyms:
                    # Ensure case consistency for static synonyms
                    synonyms.update(
                        s.lower() for s in self.phrase_synonyms[skill_lower]
                    )
            elif source == "api":
                api_synonyms = self._get_synonyms_from_api(skill)
                # Ensure case consistency for API synonyms
                synonyms.update(
                    s.lower() for s in api_synonyms
                )  # Consistent lowercasing

        return synonyms

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

    @lru_cache(maxsize=10000)
    def _categorize_term(self, term: str) -> str:
        for category, data in self.category_vectors.items():
            if data["terms"] is not None:
                if any(keyword.lower() in term.lower() for keyword in data["terms"]):
                    return category
        return self._semantic_categorization(term)

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
                if similarity > best_score:
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
            try:  # Error handling WITHIN the generator
                entity_keywords = [
                    ent.text for ent in doc.ents if ent.label_ == "SKILL"
                ]

                skill_spans = [
                    (ent.start, ent.end) for ent in doc.ents if ent.label_ == "SKILL"
                ]
                non_entity_tokens = []
                for i, token in enumerate(doc):
                    if any(start <= i < end for start, end in skill_spans):
                        continue
                    non_entity_tokens.append(token.text)

                preprocessed_text = self.preprocessor.preprocess(
                    " ".join(non_entity_tokens)
                )
                token_list = preprocessed_text.split()

                original_tokens = list(
                    set(
                        [t.lower() for t in entity_keywords]
                        + [
                            t
                            for t in token_list
                            if t not in self.preprocessor.stop_words and len(t) > 1
                        ]
                    )
                )

                non_entity_keywords = set()

                for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
                    non_entity_keywords.update(self._generate_ngrams(token_list, n))

                keywords = set(entity_keywords) | non_entity_keywords
                filtered_keywords = [
                    kw
                    for kw in keywords
                    if len(kw.strip()) > 1
                    and not any(len(w.strip()) <= 1 for w in kw.split())
                    and not all(w in self.preprocessor.stop_words for w in kw.split())
                ]

                # Configurable processing order (Fuzzy before/after Semantic)
                if self.config["text_processing"].get("fuzzy_before_semantic", True):
                    filtered_keywords_list = self._apply_fuzzy_matching_and_pos_filter(
                        [filtered_keywords]
                    )
                    if filtered_keywords_list:
                        filtered_keywords = filtered_keywords_list[0]
                    else:
                        filtered_keywords = []
                    if self.config.get("semantic_validation", False):
                        filtered_keywords = self._semantic_filter(
                            [filtered_keywords], [doc]
                        )[0]
                else:  # Semantic before Fuzzy
                    if self.config.get("semantic_validation", False):
                        filtered_keywords = self._semantic_filter(
                            [filtered_keywords], [doc]
                        )[0]
                    filtered_keywords_list = self._apply_fuzzy_matching_and_pos_filter(
                        [filtered_keywords]
                    )
                    if filtered_keywords_list:
                        filtered_keywords = filtered_keywords_list[0]
                    else:
                        filtered_keywords = []

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
            for keyword in keywords:
                if keyword.lower() in self.all_skills:
                    filtered_keywords.append(keyword)
                    continue

                best_match, score = process.extractOne(
                    keyword.lower(),
                    self.all_skills,
                    scorer=scorer,
                    score_cutoff=score_cutoff,
                )
                if best_match and score >= score_cutoff:
                    term_doc = cached_process_term(best_match)
                    # Use the allowed_pos from the config
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
        if not self.config.get("semantic_validation", False):
            return True

        if keyword in self.config.get("negative_keywords", []):
            return False

        keyword_doc = self.nlp(keyword)
        if hasattr(keyword_doc._, "trf_data") and keyword_doc._.trf_data.tensors:
            keyword_embedding = keyword_doc._.trf_data.tensors[0].mean(axis=0)
        else:
            keyword_embedding = keyword_doc.vector

        if keyword_embedding.size == 0:
            logger.debug(f"Empty embedding for '{keyword}' - skipping semantic check")
            return True

        sentences = self._extract_sentences(doc.text)  # Use the new sentence extractor
        context_window = self._get_context_window(sentences, keyword)

        if not context_window:
            return False

        context_doc = self.nlp(context_window)
        if hasattr(context_doc._, "trf_data") and context_doc._.trf_data.tensors:
            context_embedding = context_doc._.trf_data.tensors[0].mean(axis=0)
        else:
            context_embedding = context_doc.vector

        if context_embedding.size == 0:
            logger.debug(f"Empty embedding for context window - skipping")
            return True

        similarity = cosine_similarity(keyword_embedding, context_embedding)
        return similarity > self.config["text_processing"]["similarity_threshold"]

    # Added method for extracting sentences with custom rules
    def _extract_sentences(self, text: str) -> List[str]:
        # Custom rules to split sentences based on bullet points and numbered lists
        text = re.sub(r"(\s*•\s*|-\s+|\s*[0-9]+\.\s+)", r"\n", text)
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents]

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
        doc = self.nlp(text)
        sections = {}
        current_section = "General"
        sections[current_section] = ""
        for sent in doc.sents:
            match = self._section_heading_re.match(sent.text)
            if match:
                current_section = match.group(0).strip().rstrip(":")
                sections[current_section] = ""
            sections[current_section] += " " + sent.text.strip()
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

        recall = (
            len(set(summary.index.str.lower()) & original_skills) / len(original_skills)
            if original_skills
            else 0
        )
        # Prevent division by zero by checking both emptiness and length
        time_per_job = (
            (time.time() - start_time) / len(summary)
            if not summary.empty and len(summary) > 0
            else 0.5
        )
        return {
            "recall": recall,
            "memory": psutil.virtual_memory().percent,
            "time_per_job": time_per_job,
        }

    def _get_synonyms_from_api(self, phrase: str) -> List[str]:
        """Fetches synonyms from the API with caching, timeout, and exponential backoff retries."""
        phrase_lower = phrase.lower()
        if phrase_lower in self.api_cache:
            return self.api_cache[phrase_lower]

        endpoint = self.config["text_processing"]["api_endpoint"]
        api_key = self.config["text_processing"]["api_key"]
        headers = {"Authorization": f"Bearer {api_key}"}
        params = {"phrase": phrase}

        max_retries = 3
        base_delay = 1  # seconds

        for attempt in range(max_retries):
            try:
                response = requests.get(
                    endpoint, headers=headers, params=params, timeout=5
                )
                response.raise_for_status()  # This will raise for 4xx and 5xx errors

                try:
                    data = response.json()
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"API JSON decoding error for phrase: {phrase} - {e}"
                    )
                    self.api_cache[phrase_lower] = []
                    return []

                if "synonyms" not in data:
                    logger.warning(
                        f"API response missing 'synonyms' key for phrase: {phrase}"
                    )
                    self.api_cache[phrase_lower] = []
                    return []
                synonyms = data["synonyms"]
                self.api_cache[phrase_lower] = synonyms
                return synonyms
            except requests.Timeout as e:
                logger.warning(
                    f"API timeout (attempt {attempt + 1}/{max_retries}): {e}"
                )
            except requests.RequestException as e:  # Catch *all* request exceptions
                logger.warning(f"API error (attempt {attempt + 1}/{max_retries}): {e}")
                # Log the full response if possible for debugging
                if hasattr(e, "response") and e.response is not None:
                    logger.warning(f"API Response: {e.response.text}")
            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt)  # Exponential backoff
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logger.error(
                    f"API request failed after {max_retries} retries for phrase: {phrase}"
                )
                self.api_cache[phrase_lower] = []
                return []

    @lru_cache(maxsize=None)  # Or a reasonable maxsize, e.g., 1024*10
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
        keyword_lower = keyword.lower()
        match = re.search(
            rf"(?i)\b{re.escape(keyword_lower)}\b", text
        )  # Case-insensitive, whole word match
        if match:
            match_start = match.start()
            sections = {}
            current_section = "General"
            sections[current_section] = ""

            # Find section headings *before* the keyword match
            for heading_match in self._section_heading_re.finditer(text):
                if heading_match.start() < match_start:
                    current_section = heading_match.group(0).strip().rstrip(":").lower()
                else:
                    break  # Stop searching after the keyword
            return current_section

        return "default"

    # Add missing _process_term method
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


class ParallelProcessor:
    def __init__(self, config: Dict, nlp, keyword_extractor):
        self.config = config
        self.nlp = nlp
        try:
            self.complexity_cache = LRUCache(
                maxsize=config["caching"].get("cache_size", 1000)
            )
        except Exception as e:
            logger.error(f"Failed to initialize LRUCache: {e}")
            self.complexity_cache = {}
        self.keyword_extractor = keyword_extractor

    def get_optimal_workers(self, texts: List[str]) -> int:
        import torch  # Local import

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

        # --- GPU Memory Check (Optional) ---
        if self.config["hardware_limits"].get("use_gpu", False) and self.config.get(
            "check_gpu_memory", False
        ):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                free_mem = torch.cuda.mem_get_info()[0]
                if free_mem < 2e9:  # 2GB threshold
                    logger.warning(
                        "GPU memory low, disabling GPU usage (or reducing workers)"
                    )
                    # Option 1: Disable GPU (less preferred)
                    # self.config["hardware_limits"]["use_gpu"] = False
                    # Option 2: Reduce workers (more robust)
                    return max(1, self.config["hardware_limits"]["max_workers"] // 2)

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
        workers = self.get_optimal_workers(texts)
        chunk_size = max(1, len(texts) // workers)
        chunks = self._chunk_texts(texts, chunk_size)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(self._process_text_chunk, chunks))
        return [kw for chunk_result in results for kw in chunk_result]

    def _process_text_chunk(self, texts: List[str]) -> List[List[str]]:
        return self.keyword_extractor.extract_keywords(texts)

    def _chunk_texts(self, texts: List[str], chunk_size: int) -> List[List[str]]:
        return [texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)]


class TrigramOptimizer:
    def __init__(
        self, config: Dict, all_skills: List[str], nlp, keyword_extractor
    ):  # Add keyword_extractor parameter
        self.config = config
        self.nlp = nlp
        self.cache = LRUCache(
            maxsize=config["optimization"].get("trigram_cache_size", 1000)
        )
        self.hit_rates = deque(maxlen=100)
        self.hit_rates.append(0)

        # Use the existing preprocessor from keyword_extractor
        self.keyword_extractor = keyword_extractor  # Store the keyword_extractor
        self.preprocessor = (
            self.keyword_extractor.preprocessor
        )  # Use existing preprocessor

        warmup_size = min(
            config["optimization"].get("trigram_warmup_size", 100),
            len(all_skills),
            int(psutil.virtual_memory().available / (1024 * 1024 * 0.1)),
        )
        if not all_skills:
            logger.warning("No skills loaded from categories for TrigramOptimizer")

        # --- MODIFIED: Warmup using preprocessed tokens ---
        for skill in all_skills[:warmup_size]:
            try:
                cleaned_skill = self.preprocessor.preprocess(
                    skill
                )  # Use self.preprocessor
                doc = self.nlp(cleaned_skill)
                tokens = [
                    token.lemma_.lower()
                    for token in doc
                    if token.text.lower() not in self.preprocessor.stop_words
                    and len(token.text) > 1
                ]
                # Use the tokens for warmup
                for n in range(1, 3 + 1):  # Generate up to trigrams
                    for ngram in self._generate_ngrams(
                        tokens, n
                    ):  # Use cached _generate_ngrams
                        self.get_candidates(ngram)  # Add to cache

            except (ValueError, TypeError, AttributeError) as e:
                logger.warning(f"Error during warmup for '{skill[:50]}...': {e}")
        logger.info(f"Warmed up trigram cache with {warmup_size} category terms")

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
        self.state_history = deque(maxlen=100)
        self.last_reward = None

    def get_chunk_size(self, dataset_stats: Dict) -> int:
        state = (
            int(dataset_stats["avg_length"] / 100),
            int(dataset_stats["num_texts"] / 1000),
            int(psutil.virtual_memory().percent / 10),
        )
        self.state_history.append(state)
        for key in list(self.q_table.keys()):
            self.q_table[key] *= self.decay_factor
            if self.q_table[key] < 0.01:
                del self.q_table[key]

        return max(
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

    def update_model(self, reward: float):
        self.reward_history.append(reward)
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
        return new_params

    def _adjust_chunk_size(self, mem_trend: float) -> int:
        current = self.config["dataset"].get(
            "chunk_size", self.config["dataset"]["default_chunk_size"]
        )
        if mem_trend > 80:
            return current // 2
        if mem_trend < 60:
            return current * 2
        return current


class OptimizedATS:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        if not self.config.get(
            "keyword_categories"
        ):  # Check for existence and non-emptiness
            raise ConfigError(
                "The 'keyword_categories' section must be defined in the config.yaml file and contain at least one category."
            )
        self.nlp = self._load_and_configure_spacy_model()
        self.keyword_extractor = AdvancedKeywordExtractor(self.config, self.nlp)
        self.processor = ParallelProcessor(
            self.config, self.nlp, self.keyword_extractor
        )
        self.trigram_optim = TrigramOptimizer(
            self.config,
            self.keyword_extractor.all_skills,
            self.nlp,
            self.keyword_extractor,  # Pass keyword_extractor
        )
        self.chunker = SmartChunker(self.config)
        self.tuner = AutoTuner(self.config)
        self.working_dir = Path(
            self.config["intermediate_save"].get("working_dir", "working_dir")
        )
        self.working_dir.mkdir(exist_ok=True)
        self.run_id = xxhash.xxh3_64(
            f"{time.time()}_{random.randint(0, 1000)}".encode()
        ).hexdigest()
        self._validate_config()
        self._add_entity_ruler(self.nlp)  # Pass nlp
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

                texts = list(chunk.values())
                # Use the generator but collect results immediately for this batch
                enhanced_keywords_with_original = list(
                    self.processor.keyword_extractor.extract_keywords(texts)
                )

                chunk_results = self._process_chunk(
                    dict(zip(chunk.keys(), enhanced_keywords_with_original)), chunk
                )
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
                self.config.update(new_params)
                self.chunker.update_model(self._calc_reward(metrics))

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
            "avg_length": np.mean(lengths) if lengths else 0,
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
        self, keywords: Dict, chunk: Dict
    ) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
        if not chunk:
            return None
        try:
            # Pass keywords directly, no need to extract values or texts
            dtm, features = self._create_tfidf_matrix(
                list(keywords.values())  # Only pass the keyword tuples
            )

            # Convert generator to list for DataFrame creation
            results = list(self._calculate_scores(dtm, features, keywords, chunk))

            df = pd.DataFrame(results)
            if df.empty:
                return pd.DataFrame(), pd.DataFrame()
            summary_chunk = df.groupby("Keyword").agg(
                {"Score": ["sum", "mean"], "Job Title": "nunique"}
            )
            summary_chunk.columns = ["Total_Score", "Avg_Score", "Job_Count"]
            details_chunk = df
            return summary_chunk, details_chunk  # Return the DataFrames
        except Exception as e:
            logger.exception(f"Error processing chunk: {e}")
            raise  # Always raise the exception

    def _calc_metrics(self, chunk_results: Tuple[pd.DataFrame, pd.DataFrame]) -> Dict:
        start_time = time.time()
        summary, _ = chunk_results
        # Calculate recall against the *original* set of skills (before expansion)
        original_skills = set()
        for category_skills in self.config["keyword_categories"].values():
            original_skills.update(s.lower() for s in category_skills)

        recall = (
            len(set(summary.index.str.lower()) & original_skills) / len(original_skills)
            if original_skills
            else 0
        )
        # Prevent division by zero by checking both emptiness and length
        time_per_job = (
            (time.time() - start_time) / len(summary)
            if not summary.empty and len(summary) > 0
            else 0.5
        )
        return {
            "recall": recall,
            "memory": psutil.virtual_memory().percent,
            "time_per_job": time_per_job,
        }

    def _calc_reward(self, metrics: Dict) -> float:
        weights = self.config["optimization"]["reward_weights"]
        scale = self.config["optimization"].get("memory_scale_factor", 100)
        return (
            metrics["recall"] * weights["recall"]
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
                nlp.add_pipe("lemmatizer", config={"mode": "rule"})
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
        """Loads spaCy model, disabling unnecessary components dynamically."""
        model_name = self.config["text_processing"]["spacy_model"]
        enabled_components = self.config["text_processing"]["spacy_pipeline"].get(
            "enabled_components", []
        )
        enabled = set(enabled_components)

        # Dynamically determine required components
        required = {"tok2vec", "tagger"}
        if "lemmatizer" in enabled:
            required.update(
                spacy.info(model_name).get("dependencies", {}).get("lemmatizer", [])
            )

        all_pipes = set(spacy.info(model_name)["pipelines"])
        disabled = list(all_pipes - enabled - required)

        max_retries = 3
        retry_delay = 5

        for attempt in range(max_retries):
            try:
                nlp = spacy.load(model_name, disable=disabled)
                if "sentencizer" not in nlp.pipe_names:
                    nlp.add_pipe("sentencizer")
                if "lemmatizer" not in nlp.pipe_names and "lemmatizer" in enabled:
                    nlp.add_pipe("lemmatizer", config={"mode": "rule"})
                logger.info(
                    f"Loaded spaCy model: {model_name} with pipeline: {nlp.pipe_names}"
                )
                return nlp
            except OSError as e:
                logger.warning(
                    f"Failed to load spaCy model '{model_name}' (attempt {attempt + 1}/{max_retries}): {e}"
                )
                logger.info(f"Attempting to download spaCy model: {model_name}")
                if self._download_model(model_name):
                    if not spacy.util.is_package(model_name):  # Verify installation
                        logger.error(f"Downloaded model '{model_name}' appears invalid")
                        continue
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error(
                        f"Failed to load/download '{model_name}' after {max_retries} retries"
                    )
                    raise

    def _add_entity_ruler(self, nlp):
        if "entity_ruler" not in nlp.pipe_names:
            ruler = nlp.add_pipe(
                "entity_ruler",
                config={"phrase_matcher_attr": "LOWER", "validate": True},
                before="ner",
            )
        else:
            ruler = nlp.get_pipe("entity_ruler")

        section_patterns = [
            {"label": "SECTION", "pattern": [{"LOWER": heading.lower()}]}
            for heading in self.config["advanced"]["section_headings"]
        ]
        unique_skills = set()
        for terms in self.config["keyword_categories"].values():
            unique_skills.update(terms)
        skill_patterns = [
            {"label": "SKILL", "pattern": [{"LOWER": skill.lower()}]}
            for skill in unique_skills
        ]
        ruler.add_patterns(section_patterns + skill_patterns)
        logger.info(f"Added {len(unique_skills)} unique skill patterns to entity ruler")

    def _init_categories(self):
        self.keyword_extractor._init_categories()

    def _save_intermediate(
        self,
        batch_idx: int,
        summary_chunks: List[pd.DataFrame],
        details_chunks: List[pd.DataFrame],
    ):
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

        checksums = {}  # Dictionary to store checksums, will be written line by line

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
                summary_schema = None
                details_schema = None

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

                        # Store schema for future appends (correctly using PyArrow API)
                        try:
                            summary_schema = pq.ParquetFile(summary_path).schema_arrow
                            details_schema = pq.ParquetFile(details_path).schema_arrow
                        except Exception as e:
                            logger.warning(
                                f"Could not read schema from initial files: {e}"
                            )
                            # Continue without schema - will use schema inference

                    else:  # Subsequent chunks: append using ParquetWriter

                        def append_to_parquet(path, df, append, schema=None):
                            if append:
                                try:
                                    # Create table from DataFrame
                                    table = pa.Table.from_pandas(
                                        df, preserve_index=False
                                    )

                                    # If schema provided, validate compatibility
                                    if schema is not None:
                                        # Check compatibility - basic field count check
                                        if len(table.schema) != len(schema):
                                            logger.warning(
                                                f"Schema field count mismatch. Expected {len(schema)}, got {len(table.schema)}. Using DataFrame schema."
                                            )
                                        else:
                                            # Use the stored schema for consistency
                                            writer_schema = schema
                                    else:
                                        # Fall back to table's schema if no stored schema
                                        writer_schema = table.schema

                                    # Write with appropriate compression based on available options
                                    compression = (
                                        "ZSTD"
                                        if "ZSTD" in pq.compression_codecs
                                        else "SNAPPY"
                                    )
                                    with pq.ParquetWriter(
                                        path,
                                        writer_schema,
                                        append=True,
                                        compression=compression,
                                    ) as writer:
                                        writer.write_table(table)

                                except Exception as e:
                                    logger.error(
                                        f"Error appending to Parquet file {path}: {e}"
                                    )
                                    # Fall back to creating a new file with a modified name if append fails
                                    fallback_path = Path(str(path) + f".part{i}")
                                    logger.warning(
                                        f"Creating fallback file at {fallback_path}"
                                    )
                                    feather.write_feather(df, fallback_path)
                                    return xxhash.xxh3_64(
                                        str(fallback_path).encode()
                                    ).hexdigest()
                            else:
                                feather.write_feather(df, path)

                            return self._calculate_file_checksum(path)

                        # Pass schema to append function
                        checksums[str(summary_path)] = save_and_verify(
                            summary_path,
                            summary_chunk,
                            lambda p, d, a: append_to_parquet(p, d, a, summary_schema),
                            append=True,
                        )
                        checksums[str(details_path)] = save_and_verify(
                            details_path,
                            details_chunk,
                            lambda p, d, a: append_to_parquet(p, d, a, details_schema),
                            append=True,
                        )
            elif format_type == "jsonl":
                for i, (summary_chunk, details_chunk) in enumerate(
                    zip(summary_chunks, details_chunks)
                ):
                    append = i > 0
                    if i == 0:
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
                    else:
                        # Correctly pass append parameter to srsly.write_jsonl
                        save_and_verify(
                            summary_path,
                            (r.to_dict() for _, r in summary_chunk.iterrows()),
                            lambda p, d, a: srsly.write_jsonl(p, d, append=a),
                            append=append,
                        )
                        save_and_verify(
                            details_path,
                            (r.to_dict() for _, r in details_chunk.iterrows()),
                            lambda p, d, a: srsly.write_jsonl(p, d, append=a),
                            append=append,
                        )
            else:  # json format
                # For JSON format, we append by combining and rewriting
                combined_summary = pd.concat(
                    [chunk for chunk in summary_chunks]
                ).reset_index()
                combined_details = pd.concat(
                    [chunk for chunk in details_chunks]
                ).reset_index()

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

            logger.info(f"Intermediate results saved to {self.working_dir}")

            # Save the checksums to a manifest file
            self._save_checksum_manifest(checksums)

        except Exception as e:
            logger.error(f"Failed to save intermediate results: {e}")

    def _verify_single_checksum(self, file_path: Path, expected_checksum: str) -> bool:
        """Verifies the checksum of a single file."""
        if not file_path.exists():
            logger.error(f"Intermediate file not found: {file_path}")
            return False
        calculated_hash = self._calculate_file_checksum(file_path)
        if calculated_hash != expected_checksum:
            logger.error(
                f"Checksum mismatch for {file_path}: expected {expected_checksum}, got {calculated_hash}"
            )
            return False
        logger.info(f"Checksum verified for: {file_path}")
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
                logger.info("Cleaned up intermediate directory: %s", self.working_dir)
            except (OSError, IOError, PermissionError) as e:
                logger.error("Failed to cleanup intermediate directory: %s", e)

    def _create_tfidf_matrix(self, keyword_sets_tuples):
        """Creates TF-IDF matrix, fitting only once (with optional sampling)."""
        max_features = self.config["caching"].get("tfidf_max_features", 10000)
        combined_sets = [list(set(o + e)) for o, e in keyword_sets_tuples]
        all_keywords = [kw for subset in combined_sets for kw in subset]

        if not hasattr(self, "tfidf_vectorizer"):
            # Sample for large datasets
            if len(all_keywords) > 100000:
                logger.info(
                    f"Sampling {100000} keywords for TF-IDF vocabulary (original size: {len(all_keywords)})"
                )
                all_keywords = random.sample(all_keywords, 100000)  # Sample 100k

            self.tfidf_vectorizer = TfidfVectorizer(
                ngram_range=self.keyword_extractor.ngram_range,
                max_features=max_features,
                dtype=np.float32,
                lowercase=False,
                tokenizer=lambda x: x,
                preprocessor=lambda x: x,
            )
            self.tfidf_vectorizer.fit(all_keywords)  # Fit on sampled keywords
        dtm = self.tfidf_vectorizer.transform(combined_sets)
        return dtm, self.tfidf_vectorizer.get_feature_names_out()

    def _calculate_scores(self, dtm, feature_names, keyword_sets, job_descriptions):
        """Calculates scores, yielding result dictionaries for each keyword."""
        dtm_coo = dtm.tocoo()
        job_descriptions_list = list(job_descriptions.items())
        weighting = self.config.get(
            "weighting",
            {"tfidf_weight": 0.7, "frequency_weight": 0.3, "whitelist_boost": 1.5},
        )

        for row, col, value in zip(dtm_coo.row, dtm_coo.col, dtm_coo.data):
            try:
                job_index = row
                term_index = col

                # Check for index out of bounds errors
                if job_index >= len(job_descriptions_list) or term_index >= len(
                    feature_names
                ):
                    logger.warning(
                        f"Index out of bounds: job_index={job_index}, term_index={term_index}"
                    )
                    continue

                title = job_descriptions_list[job_index][0]
                term = feature_names[term_index]

                # Check if keyword_sets has the expected structure
                if job_index >= len(keyword_sets):
                    logger.warning(f"Keyword set index out of bounds: {job_index}")
                    continue

                original_tokens, expanded_keywords = keyword_sets[job_index]

                presence = (
                    1
                    if isinstance(original_tokens, list) and term in original_tokens
                    else 0
                )  # Use original tokens
                score = (
                    weighting.get("tfidf_weight", 0.7) * value
                    + weighting.get("frequency_weight", 0.3) * presence
                )
                term_lower = term.lower()

                # Apply whitelist boost if the term is in the *expanded* set of skills:
                if term_lower in self.keyword_extractor.all_skills:
                    score *= weighting.get("whitelist_boost", 1.5)
                section_weights = self.config["weighting"].get("section_weights", {})
                job_text = job_descriptions_list[job_index][1]

                section = self.keyword_extractor._detect_keyword_section(term, job_text)
                score *= section_weights.get(
                    section,
                    section_weights.get("default", 1.0),
                )

                result = {
                    "Keyword": term,
                    "Job Title": title,
                    "Score": score,
                    "TF-IDF": value,
                    "Frequency": presence,
                    "Category": self.keyword_extractor._categorize_term(term),
                    "In Whitelist": term_lower in self.keyword_extractor.all_skills,
                }
                yield result

            except Exception as e:
                logger.error(f"Error calculating score for row={row}, col={col}: {e}")
                # No yield here - just skip this item and continue with the next one


def parse_arguments():
    parser = argparse.ArgumentParser(description="ATS Keyword Optimizer")
    parser.add_argument(
        "-i", "--input", default="job_descriptions.json", help="Input JSON file"
    )
    parser.add_argument("-c", "--config", default="config.yaml", help="Config file")
    parser.add_argument("-o", "--output", default="results.xlsx", help="Output file")
    return parser.parse_args()


def initialize_analyzer(config_path: str):
    ensure_nltk_resources()
    analyzer = OptimizedATS(config_path)
    return analyzer


def save_results(summary: pd.DataFrame, details: pd.DataFrame, output_file: str):
    try:
        working_dir = Path(output_file).parent
        if shutil.disk_usage(working_dir).free < 1000000000:
            logger.warning("Low disk space in working directory")

        with pd.ExcelWriter(output_file) as writer:
            summary.to_excel(writer, sheet_name="Summary")
            details.to_excel(writer, sheet_name="Detailed Scores")
        logger.info("Analysis complete. Results saved to %s", output_file)
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
        logger.error("Invalid JSON in %s: %s", input_file, e)
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

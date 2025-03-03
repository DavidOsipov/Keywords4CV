"""
Configuration validation utilities for Keywords4CV.

This module uses both Schema (for initial YAML structure validation)
and Pydantic (for runtime validation and type coercion).
"""

import logging
from typing import Dict, Any, List, Literal, Optional, Tuple, Set
from urllib.parse import urlparse

try:
    from pydantic import BaseModel, Field, ValidationError, field_validator, ConfigDict
except ImportError:
    from pydantic.v1 import BaseModel, Field, ValidationError
    from pydantic.v1 import validator as field_validator

    ConfigDict = dict

try:
    from schema import Schema, SchemaError, Optional as SchemaOptional, And, Or
except ImportError as exc:
    # If schema is not installed, install it with pip install schema
    raise ImportError(
        "The 'schema' package is required. Install it with 'pip install schema'"
    ) from exc

import yaml
import sys
from pathlib import Path


class ConfigError(Exception):
    """Custom exception for configuration errors."""


class ValidationConfig(BaseModel):
    """Configuration model for validation settings."""

    allow_numeric_titles: bool = True
    empty_description_policy: str = "warn"
    title_min_length: int = Field(2, ge=1)
    title_max_length: int = Field(100, ge=1)
    min_desc_length: int = Field(60, ge=1)
    text_encoding: str = "utf-8"
    model_config = ConfigDict(extra="forbid")


class DatasetConfig(BaseModel):
    """Configuration model for dataset parameters."""

    short_description_threshold: int = Field(25, ge=1)
    min_job_descriptions: int = Field(3, ge=1)
    max_job_descriptions: int = Field(120, ge=1)
    min_jobs: int = Field(3, ge=1)

    model_config = ConfigDict(extra="forbid")


class SpacyPipelineConfig(BaseModel):
    """Configuration model for spaCy NLP pipeline components."""

    enabled_components: List[str] = Field(
        default_factory=lambda: [
            "tok2vec",
            "tagger",
            "lemmatizer",
            "entity_ruler",
            "sentencizer",
        ]
    )

    model_config = ConfigDict(extra="forbid")


class TextProcessingConfig(BaseModel):
    """Configuration model for text processing settings."""

    spacy_model: str = "en_core_web_lg"
    spacy_pipeline: SpacyPipelineConfig = Field(default_factory=SpacyPipelineConfig)
    ngram_range: Tuple[int, int] = Field((1, 3))
    whitelist_ngram_range: Tuple[int, int] = Field((1, 2))
    pos_filter: List[str] = Field(default_factory=lambda: ["NOUN", "PROPN", "ADJ"])
    semantic_validation: bool = True
    similarity_threshold: float = 0.85
    pos_processing: str = "hybrid"
    phrase_synonym_source: Literal["static", "api"] = "static"
    phrase_synonyms_path: Optional[str] = None
    api_endpoint: Optional[str] = None
    api_key: Optional[str] = None
    context_window_size: int = Field(1, ge=0)
    fuzzy_before_semantic: bool = True

    @classmethod
    @field_validator("ngram_range", "whitelist_ngram_range")
    def check_ngram_ranges(cls, value):
        """Validate that ngram range start is less than or equal to end."""
        if value[0] > value[1]:
            raise ValueError("ngram_range/whitelist_ngram_range start must be <= end")
        return value

    @classmethod
    @field_validator("phrase_synonyms_path")
    def validate_phrase_synonyms_path(cls, value, values):
        """Validate that phrase_synonyms_path is provided when using static source."""
        if values.get("phrase_synonym_source") == "static" and not value:
            raise ValueError(
                "phrase_synonyms_path must be provided when phrase_synonym_source is 'static'"
            )
        return value

    @classmethod
    @field_validator("api_endpoint", "api_key")
    def validate_api_settings(cls, value, values, field):
        """Validate API settings when API source is selected."""
        if values.get("phrase_synonym_source") == "api" and not value:
            raise ValueError(
                f"{field.name} must be provided when phrase_synonym_source is 'api'"
            )
        return value

    @classmethod
    @field_validator("api_endpoint")
    def validate_api_url(cls, value, _):
        """Validate that API endpoint is a valid URL."""
        if value:
            pass  # Add indented block here.  Replace with actual logic if needed.


class CategorizationConfig(BaseModel):
    """Configuration model for keyword categorization."""

    default_category: str = "Uncategorized Skills"
    categorization_cache_size: int = 15000
    direct_match_threshold: float = 0.85

    model_config = ConfigDict(extra="forbid")


class FuzzyMatchingConfig(BaseModel):
    """Configuration model for fuzzy matching settings."""

    enabled: bool = True
    max_candidates: int = Field(3, ge=1)
    allowed_pos: List[str] = Field(default_factory=lambda: ["NOUN", "PROPN"])
    min_similarity: int = Field(85, ge=0, le=100)
    algorithm: str = "WRatio"
    default_pos_filter: List[str] = Field(
        default_factory=lambda: ["NOUN", "PROPN", "VERB"]
    )

    model_config = ConfigDict(extra="forbid")


class WhitelistConfig(BaseModel):
    """Configuration model for whitelist settings.
    This class defines the structure and validation rules for whitelist configuration
    parameters used in the application.
    Attributes:
        whitelist_recall_threshold (float): The minimum recall threshold for whitelist matches.
            Must be between 0.0 and 1.0. Defaults to 0.72.
        whitelist_cache (bool): Whether to cache whitelist results. Defaults to True.
        cache_validation (str): Method of cache validation to use. Defaults to "strict".
        fuzzy_matching (FuzzyMatchingConfig): Configuration for fuzzy matching algorithms.
            Defaults to a new instance of FuzzyMatchingConfig.
    Note:
        This model forbids extra attributes not defined in the schema.
    """

    whitelist_recall_threshold: float = Field(0.72, ge=0.0, le=1.0)
    whitelist_cache: bool = True
    cache_validation: str = "strict"
    fuzzy_matching: FuzzyMatchingConfig = Field(default_factory=FuzzyMatchingConfig)
    model_config = ConfigDict(extra="forbid")

    class Config:
        pass  # Add indented block


class WeightingConfig(BaseModel):
    """Configuration model for keyword weighting parameters."""

    tfidf_weight: float = Field(0.65, ge=0.0)
    frequency_weight: float = Field(0.35, ge=0.0)
    whitelist_boost: float = Field(1.6, ge=0.0)
    section_weights: Dict[str, float] = Field(
        default_factory=lambda: {"education": 1.2}
    )
    default_pos_filter: List[str] = Field(
        default_factory=lambda: ["NOUN", "PROPN", "VERB"]
    )  # Corrected location
    model_config = ConfigDict(extra="forbid")


class HardwareLimitsConfig(BaseModel):
    """Configuration model for hardware resource limits."""

    use_gpu: bool = True
    batch_size: int = Field(64, ge=1)
    auto_chunk_threshold: int = Field(100, ge=1)
    memory_threshold: int = Field(70, ge=0, le=100)
    max_ram_usage_percent: int = Field(80, ge=0, le=100)
    abort_on_oom: bool = True
    max_workers: int = Field(4, ge=1)
    memory_scaling_factor: float = Field(0.3, ge=0.0, le=1.0)

    model_config = ConfigDict(extra="forbid")

    class Config:
        pass  # Add indented block


class CachingConfig(BaseModel):
    """Configuration model for caching settings."""

    cache_size: int = Field(15000, ge=1)
    tfidf_max_features: int = Field(100000, ge=1)
    cache_salt: str = "default_secret_salt"

    model_config = ConfigDict(extra="forbid")


class IntermediateSaveConfig(BaseModel):
    """Configuration model for intermediate save settings."""

    enabled: bool = True
    save_interval: int = Field(15, ge=0)
    format: str = Field("feather", alias="format_")
    working_dir: str = "working_dir"
    cleanup: bool = True

    model_config = ConfigDict(extra="forbid")


class AdvancedConfig(BaseModel):
    """Configuration model for advanced settings."""

    dask_enabled: bool = False
    success_rate_threshold: float = Field(0.7, ge=0.0, le=1.0)
    checksum_rtol: float = Field(0.001, ge=0.0)
    negative_keywords: List[str] = Field(
        default_factory=lambda: ["company", "team", "office"]
    )
    section_headings: List[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class OptimizationConfig(BaseModel):
    """Configuration model for optimization settings."""

    complexity_entity_weight: int = Field(10, ge=1)
    complexity_fallback_factor: float = Field(1.0, ge=0.0)
    trigram_cache_size: int = Field(1000, ge=1)
    trigram_warmup_size: int = Field(100, ge=1)
    q_table_decay: float = Field(0.99, ge=0.0, le=1.0)
    reward_weights: Dict[str, float] = {  # Corrected dictionary
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

    # Define the required keys for reward_weights *inside* the class, but *outside* any method
    _required_reward_keys: Set[str] = {"recall", "memory", "time"}

    @classmethod  # Decorators *before* the method
    @field_validator("reward_weights")
    def reward_weights_keys(cls, v):
        """Validate that reward weights contain required keys."""
        if not set(v.keys()) == cls._required_reward_keys:
            raise ValueError(
                f"reward_weights must contain exactly these keys: {cls._required_reward_keys}"
            )
        return v  # Add return


class Config(BaseModel):
    """Main configuration model for Keywords4CV."""

    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    text_processing: TextProcessingConfig = Field(default_factory=TextProcessingConfig)
    categorization: CategorizationConfig = Field(default_factory=CategorizationConfig)
    whitelist: WhitelistConfig = Field(default_factory=WhitelistConfig)
    weighting: WeightingConfig = Field(default_factory=WeightingConfig)
    hardware_limits: HardwareLimitsConfig = Field(default_factory=HardwareLimitsConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    caching: CachingConfig = Field(default_factory=CachingConfig)
    intermediate_save: IntermediateSaveConfig = Field(
        default_factory=IntermediateSaveConfig
    )
    advanced: AdvancedConfig = Field(default_factory=AdvancedConfig)
    stop_words: List[str] = Field(default_factory=list)
    stop_words_add: List[str] = Field(default_factory=list)
    stop_words_exclude: List[str] = Field(default_factory=list)
    keyword_categories: Dict[str, List[str]] = Field(default_factory=dict)

    model_config = ConfigDict(
        extra="forbid"
    )  # CRITICAL: Prevent extra fields at the top level
    max_workers: int = Field(4, ge=1)


config_schema = Schema(
    {
        "validation": {
            "allow_numeric_titles": bool,
            "empty_description_policy": And(
                str, Or("warn", "error", "allow")
            ),  # Allowed values
            "title_min_length": And(int, lambda n: n >= 1),
            "title_max_length": And(int, lambda n: n >= 1),
            "min_desc_length": And(int, lambda n: n >= 1),
            "text_encoding": str,
        },
        "dataset": {
            "short_description_threshold": And(int, lambda n: n >= 1),
            "min_job_descriptions": And(int, lambda n: n >= 1),
            "max_job_descriptions": And(int, lambda n: n >= 1),
            "min_jobs": And(int, lambda n: n >= 1),
        },
        "text_processing": {
            "spacy_model": str,
            "spacy_pipeline": {"enabled_components": [str]},
            # Corrected: Use (int, int) for tuples
            "ngram_range": And((int, int), lambda n: n[0] <= n[1]),
            "whitelist_ngram_range": And((int, int), lambda n: n[0] <= n[1]),
            "pos_filter": [str],
            "semantic_validation": bool,
            "similarity_threshold": And(float, lambda n: 0.0 <= n <= 1.0),
            "pos_processing": str,
            "phrase_synonym_source": Or("static", "api"),
            # Corrected: Use SchemaOptional for optional keys
            SchemaOptional("phrase_synonyms_path"): str,
            SchemaOptional("api_endpoint"): And(str, urlparse),
            SchemaOptional("api_key"): str,
            "context_window_size": And(int, lambda n: n >= 0),
            "fuzzy_before_semantic": bool,
        },
        "categorization": {
            "default_category": str,
            "categorization_cache_size": And(int, lambda n: n >= 1),
            "direct_match_threshold": And(float, lambda n: 0.0 <= n <= 1.0),
        },
        "whitelist": {
            "whitelist_recall_threshold": And(float, lambda n: 0.0 <= n <= 1.0),
            "whitelist_cache": bool,
            "cache_validation": str,
            "fuzzy_matching": {
                "enabled": bool,
                "max_candidates": And(int, lambda n: n >= 1),
                "allowed_pos": [str],
                "min_similarity": And(int, lambda n: 0 <= n <= 100),
                "algorithm": str,
                "default_pos_filter": [str],
            },
        },
        "weighting": {
            "tfidf_weight": And(float, lambda n: n >= 0.0),
            "frequency_weight": And(float, lambda n: n >= 0.0),
            "whitelist_boost": And(float, lambda n: n >= 0.0),
            "section_weights": {str: And(float, lambda n: n >= 0.0)},
        },
        "hardware_limits": {
            "use_gpu": bool,
            "batch_size": And(int, lambda n: n >= 1),
            "auto_chunk_threshold": And(int, lambda n: n >= 1),
            "memory_threshold": And(int, lambda n: 0 <= n <= 100),
            "max_ram_usage_percent": And(int, lambda n: 0 <= n <= 100),
            "abort_on_oom": bool,
            "max_workers": And(int, lambda n: n >= 1),
            "memory_scaling_factor": And(float, lambda n: 0.0 <= n <= 1.0),
        },
        "optimization": {
            "complexity_entity_weight": And(int, lambda n: n >= 1),
            "complexity_fallback_factor": And(float, lambda n: n >= 0.0),
            "trigram_cache_size": And(int, lambda n: n >= 1),
            "trigram_warmup_size": And(int, lambda n: n >= 1),
            "q_table_decay": And(float, lambda n: 0.0 <= n <= 1.0),
            "reward_weights": {
                "recall": Or(int, float),
                "memory": Or(int, float),
                "time": Or(int, float),
            },
            "reward_std_low": And(float, lambda n: n >= 0.0),
            "reward_std_high": And(float, lambda n: n >= 0.0),
            "memory_scale_factor": And(int, lambda n: n >= 1),
            "abort_on_oom": bool,
            "max_workers": And(int, lambda n: n >= 1),
            "complexity_factor": And(int, lambda n: n >= 1),
        },
        "caching": {
            "cache_size": And(int, lambda n: n >= 1),
            "tfidf_max_features": And(int, lambda n: n >= 1),
            "cache_salt": str,
        },
        "intermediate_save": {
            "enabled": bool,
            "save_interval": And(int, lambda n: n >= 0),
            "format": str,
            "working_dir": str,
            "cleanup": bool,
        },
        "advanced": {
            "dask_enabled": bool,
            "success_rate_threshold": And(float, lambda n: 0.0 <= n <= 1.0),
            "checksum_rtol": And(float, lambda n: 0.001),
            "negative_keywords": [str],
            "section_headings": [str],
        },
        SchemaOptional("keyword_categories"): {str: [str]},
        SchemaOptional("stop_words"): [str],
        SchemaOptional("stop_words_add"): [str],
        SchemaOptional("stop_words_exclude"): [str],
        SchemaOptional("max_workers"): And(int, lambda n: n >= 1),
    }
)

logger = logging.getLogger(__name__)


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validates the configuration dictionary.  First validates the YAML
    structure using Schema, then performs runtime validation and type
    coercion using Pydantic.

    Args:
        config: The configuration dictionary to validate.

    Raises:
        ConfigError: If the configuration is invalid.  Contains details.
    """
    try:
        # First, validate the YAML structure using Schema
        validated_config = config_schema.validate(config)

        # Then, perform runtime validation and type coercion with Pydantic
        Config(**validated_config)

        logger.info("Configuration validation successful.")

    except SchemaError as e:
        logger.error("Schema validation error:\n%s", e)
        raise ConfigError(
            f"Schema validation failed:\n{e.autos}"
        ) from e  # Show detailed errors
    except ValidationError as e:
        logger.error("Pydantic validation error:\n%s", e)
        raise ConfigError(f"Pydantic validation failed:\n{e}") from e
    except Exception as e:
        logger.exception("Unexpected error during validation: %s", e)
        raise ConfigError(f"Unexpected validation error: {e}") from e

    # Top-level checks (after Schema and Pydantic)
    if not config.get("keyword_categories"):
        raise ConfigError("The keyword_categories section must not be empty")

    has_keywords = any(
        isinstance(keywords, list) and keywords
        for keywords in config["keyword_categories"].values()
    )
    if not has_keywords:
        raise ConfigError("At least one category must contain keywords")


def validate_config_file(config_path: str) -> Dict:
    """
    Loads and validates a configuration file from the given path.

    This function handles the complete loading and validation workflow:
    1. Reads the YAML configuration file
    2. Validates its structure using Schema
    3. Performs runtime validation using Pydantic

    Args:
        config_path: Path to the configuration YAML file.

    Returns:
        Dict: Validated configuration dictionary.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        yaml.YAMLError: If the YAML syntax is invalid.
        ConfigError: If the configuration content is invalid.
        ValidationError: If Pydantic validation fails.
    """
    try:
        config_path = Path(config_path)
        with open(config_path, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)

        if not isinstance(raw_config, dict):
            raise ConfigError(
                f"Configuration file must contain a YAML dictionary/object, found {type(raw_config).__name__}"
            )

        # Validate the configuration
        validate_config(raw_config)

        # Convert to Pydantic model and back to dict
        config = Config(**raw_config)
        return config.dict(by_alias=True)  # Use alias for serialization

    except FileNotFoundError:
        logger.error("Config file not found: %s", config_path)
        raise
    except yaml.YAMLError as e:
        # Detailed YAML error information
        line_info = ""
        if hasattr(e, "problem_mark"):
            line_info = f" at line {e.problem_mark.line + 1}, column {e.problem_mark.column + 1}"
        logger.error(f"YAML syntax error in config file {config_path}{line_info}: {e}")
        raise
    except (SchemaError, ValidationError) as e:
        # Schema and Pydantic validation errors are already handled in validate_config
        raise ConfigError(f"Configuration validation failed: {e}") from e
    except Exception as e:
        logger.exception("Unexpected error loading config from %s: %s", config_path, e)
        raise ConfigError(f"Unexpected error loading configuration: {e}") from e

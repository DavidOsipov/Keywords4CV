import argparse  # For command-line argument parsing
import hashlib  # For generating hashes of configuration for caching
import json  # For handling JSON data (input and config hashing)
import logging  # For logging messages (info, warnings, errors)
import re  # For regular expressions (text cleaning)
import sys  # For system-specific parameters and functions (exiting)
from collections import OrderedDict  # For creating an LRU cache
from typing import Dict, List, Set, Tuple, NamedTuple, Optional  # For type hinting
from multiprocessing import Pool, TimeoutError  # For parallel processing
import platform  # For getting platform information (in error messages)

import nltk  # Natural Language Toolkit (for WordNet and other resources)
import numpy as np  # For numerical operations (especially with TF-IDF vectors)
import pandas as pd  # For data manipulation and analysis (DataFrames)
import spacy  # For natural language processing (tokenization, lemmatization, etc.)
import yaml  # For reading the configuration file (YAML format)
from nltk.corpus import wordnet  # For accessing WordNet lexical database
from sklearn.feature_extraction.text import TfidfVectorizer  # For TF-IDF calculations
from functools import lru_cache  # For caching function results (term vectors)
import psutil  # For monitoring system resources (memory, CPU)
import gc  # For garbage collection


# --- Custom Exceptions ---
class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass

class InputValidationError(Exception):
    """Custom exception for input validation errors."""
    pass

# --- Configure Logging ---
# Logs to both a file (ats_optimizer.log) and the console (stdout).
logging.basicConfig(
    level=logging.INFO,  # Log level: INFO, WARNING, ERROR, CRITICAL
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log message format
    handlers=[logging.FileHandler("ats_optimizer.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)  # Get the root logger


# --- NLTK Resource Management ---
# List of NLTK resources that need to be downloaded.
NLTK_RESOURCES = [
    "corpora/wordnet",  # WordNet lexical database
    "corpora/averaged_perceptron_tagger",  # POS tagger
    "tokenizers/punkt",  # Sentence tokenizer
]

def ensure_nltk_resources():
    """Ensure required NLTK resources are downloaded.  If not, download them."""
    for resource in NLTK_RESOURCES:
        try:
            nltk.data.find(resource)  # Try to find the resource
        except LookupError:
            # If not found, download it (quietly, without progress bar)
            nltk.download(resource.split("/")[1], quiet=True)


# --- Validation Result NamedTuple ---
# A structured way to return validation results (more readable than a dictionary).
class ValidationResult(NamedTuple):
    valid: bool  # True if validation passed, False otherwise
    value: Optional[str]  # The validated/sanitized value (if valid), otherwise None
    reason: Optional[str] = None  # Reason for failure (if invalid), otherwise None


# --- EnhancedTextPreprocessor Class ---
class EnhancedTextPreprocessor:
    """
    Optimized text preprocessing with memoization (caching) and batch processing.

    Handles:
        - Stop word removal (configurable)
        - URL and email removal
        - Special character removal (configurable)
        - Lowercasing
        - Lemmatization (using spaCy)
        - Batch processing for efficiency
        - Caching of preprocessed text to avoid redundant computations
    """

    def __init__(self, config: Dict, nlp):
        """
        Initialize the preprocessor.

        Args:
            config: Configuration dictionary.
            nlp: spaCy language model instance.
        """
        self.config = config  # Store the configuration
        self.nlp = nlp  # Store the spaCy model instance
        self.stop_words = self._load_stop_words()  # Load stop words
        self.regex_patterns = {  # Pre-compile regular expressions for efficiency
            "url": re.compile(r"http\S+|www\.\S+"),  # URLs
            "email": re.compile(r"\S+@\S+"),  # Email addresses
            "special_chars": re.compile(r"[^\w\s\-😊-🙏🚀-🇿]"),  # Special characters (excluding emojis)
            "whitespace": re.compile(r"\s+"),  # Multiple whitespaces
        }
        self._cache = OrderedDict()  # Use OrderedDict as an LRU cache
        self._CACHE_SIZE = config.get("cache_size", 1000)  # Cache size (default 1000)
        self.config_hash = self._calculate_config_hash()  # Hash of relevant config for cache invalidation

    def _calculate_config_hash(self) -> str:
        """
        Calculate a hash of the relevant configuration parts for cache invalidation.

        If the configuration changes (e.g., stop words are added/removed), the cache
        should be invalidated. This hash is used to detect such changes.

        Returns:
            A SHA256 hash of the relevant configuration options.
        """
        relevant_config = {  # Only include config options that affect preprocessing
            "stop_words": self.config.get("stop_words", []),
            "stop_words_add": self.config.get("stop_words_add", []),
            "stop_words_exclude": self.config.get("stop_words_exclude", []),
            # Add other relevant config options here if needed
        }
        config_str = json.dumps(relevant_config, sort_keys=True)  # Serialize to JSON
        return hashlib.sha256(config_str.encode("utf-8")).hexdigest()  # Calculate SHA256 hash

    def _load_stop_words(self) -> Set[str]:
        """
        Load and validate stop words from the configuration.

        Combines base stop words, additional stop words, and removes excluded
        stop words.  Issues a warning if the final stop word list is too small.

        Returns:
            A set of stop words.
        """
        stop_words = set(self.config.get("stop_words", []))  # Get base stop words
        stop_words.update(self.config.get("stop_words_add", []))  # Add additional stop words
        stop_words.difference_update(self.config.get("stop_words_exclude", []))  # Remove excluded words

        if len(stop_words) < 50:
            logger.warning("Stop words list seems unusually small. Verify config.")

        return stop_words

    def preprocess(self, text: str) -> str:
        """
        Preprocess a single text string.

        Applies lowercasing, URL/email/special character removal, and whitespace
        normalization.  Uses an LRU cache to avoid redundant preprocessing.

        Args:
            text: The input text string.

        Returns:
            The preprocessed text string.
        """
        # Check if config has changed; if so, clear the cache
        current_hash = self._calculate_config_hash()
        if current_hash != self.config_hash:
            self._cache.clear()
            self.config_hash = current_hash

        # Check if the text is already in the cache
        if text in self._cache:
            self._cache.move_to_end(text)  # Move to the end (most recently used)
            return self._cache[text]

        # Preprocessing steps
        cleaned = text.lower()  # Lowercase
        cleaned = self.regex_patterns["url"].sub("", cleaned)  # Remove URLs
        cleaned = self.regex_patterns["email"].sub("", cleaned)  # Remove emails
        cleaned = self.regex_patterns["special_chars"].sub(" ", cleaned)  # Remove special chars (replace with space)
        cleaned = self.regex_patterns["whitespace"].sub(" ", cleaned).strip()  # Normalize whitespace

        # Add to cache (if it's full, remove the least recently used item)
        if len(self._cache) >= self._CACHE_SIZE:
            self._cache.popitem(last=False)  # Remove from the beginning (least recently used)
        self._cache[text] = cleaned
        return cleaned

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocess a list of text strings.  Applies `preprocess` to each text.

        Args:
            texts: A list of text strings.

        Returns:
            A list of preprocessed text strings.
        """
        return [self.preprocess(text) for text in texts]  # List comprehension for efficiency

    def _process_doc_tokens(self, doc):
        """Helper function to process tokens from a single spaCy doc."""
        tokens = []
        for token in doc:
            if (
                token.text in self.stop_words
                or len(token.text) <= 1
                or token.text.isnumeric()
            ):
                continue
            try:
                lemma = token.lemma_.lower().strip()
            except AttributeError:
                lemma = token.text.lower().strip()
            tokens.append(lemma)
        return tokens

    def tokenize_batch(self, texts: List[str]) -> List[List[str]]:
        """Tokenize a list of text strings, with batch processing."""
        try:
            from multiprocessing import cpu_count
            n_process = min(cpu_count(), 4)
        except ImportError:
            n_process = 1

        tokens_list = []
        for doc in self.nlp.pipe(texts, batch_size=50, n_process=n_process):
            tokens_list.append(self._process_doc_tokens(doc))
        return tokens_list


# --- AdvancedKeywordExtractor Class ---
class AdvancedKeywordExtractor:
    """
    Enhanced keyword extraction with phrase detection (n-grams) and semantic analysis.

    Features:
        - Whitelist of important terms (with synonym expansion)
        - Configurable n-gram range for phrase extraction
        - Semantic filtering to keep keywords that appear in a meaningful context
    """

    def __init__(self, config: Dict, nlp):
        """
        Initialize the keyword extractor.

        Args:
            config: Configuration dictionary.
            nlp: spaCy language model instance.
        """
        self.config = config  # Store the configuration
        self.nlp = nlp  # Store the spaCy model instance
        self.preprocessor = EnhancedTextPreprocessor(config, nlp)  # Initialize text preprocessor
        self.whitelist = self._create_expanded_whitelist()  # Create the expanded whitelist
        self.ngram_range = tuple(config.get("ngram_range", [1, 3]))  # N-gram range (default: 1-3)
        self.whitelist_ngram_range = tuple(config.get("whitelist_ngram_range", [1, 3]))  # Whitelist n-gram range

    def _create_expanded_whitelist(self) -> Set[str]:
        """
        Create an expanded whitelist with multi-word phrases and synonyms.

        Expands the base skills whitelist by:
            - Generating n-grams (phrases) from the whitelist terms
            - Generating synonyms using WordNet and spaCy

        Returns:
            A set of whitelisted terms (including phrases and synonyms).
        """
        base_skills = self.config.get("skills_whitelist", [])  # Get the base skills list
        processed = set()

        # Batch preprocess and tokenize the skills for efficiency
        cleaned_skills = self.preprocessor.preprocess_batch(base_skills)
        tokenized = self.preprocessor.tokenize_batch(cleaned_skills)

        # Generate n-grams from the tokenized skills
        for tokens in tokenized:
            for n in range(
                self.whitelist_ngram_range[0], self.whitelist_ngram_range[1] + 1
            ):
                processed.update(self._generate_ngrams(tokens, n))

        # Add synonyms to the whitelist
        synonyms = self._generate_synonyms(base_skills)
        processed.update(synonyms)

        return processed

    def _generate_synonyms(self, skills: List[str]) -> Set[str]:
        """
        Generate semantic synonyms using WordNet and spaCy.

        For each skill in the input list:
            - Lemmatize the skill using spaCy.
            - Find synonyms using WordNet.
            - Add both the lemmatized form and WordNet synonyms to the output set.

        Args:
            skills: A list of skill terms.

        Returns:
            A set of synonyms (including lemmatized forms).
        """
        synonyms = set()
        for skill in skills:
            if not skill.strip():  # Skip empty skills
                logger.warning("Skipping empty skill in whitelist")
                continue
            doc = self.nlp(skill)  # Process the skill with spaCy
            lemmatized = " ".join([token.lemma_ for token in doc]).lower()  # Lemmatize
            if lemmatized != skill.lower():
                synonyms.add(lemmatized)  # Add the lemmatized form

            # Find WordNet synonyms
            for token in doc:
                if token.text.strip():
                    synonyms.update(
                        lemma.name().replace("_", " ").lower()  # Normalize synonyms
                        for syn in wordnet.synsets(token.text)
                        for lemma in syn.lemmas()
                        if lemma.name().replace("_", " ").lower() != token.text and lemma.name().replace("_", " ").lower() != lemmatized
                    )
        return synonyms

    def _generate_ngrams(self, tokens: List[str], n: int) -> Set[str]:
        """
        Generate n-grams (contiguous sequences of n tokens) from a list of tokens.

        Args:
            tokens: A list of tokens.
            n: The length of the n-grams.

        Returns:
            A set of n-grams (as strings).
        """
        filtered_tokens = [token for token in tokens if token.strip()]  # Remove empty tokens
        return {" ".join(filtered_tokens[i : i + n]) for i in range(len(filtered_tokens) - n + 1)}

    def extract_keywords(self, texts: List[str]) -> List[List[str]]:
        """
        Extract keywords from a list of text strings.

        Steps:
            1. Preprocess the texts.
            2. Tokenize the preprocessed texts.
            3. Generate n-grams from the tokens.
            4. Filter n-grams based on the whitelist.
            5. Apply semantic filtering (optional).

        Args:
            texts: A list of text strings (job descriptions).

        Returns:
            A list of lists of keywords (one list of keywords per input text).
        """
        cleaned = self.preprocessor.preprocess_batch(texts)  # Preprocess
        tokenized = self.preprocessor.tokenize_batch(cleaned)  # Tokenize

        all_keywords = []
        for tokens in tokenized:
            keywords = set()  # Use a set to avoid duplicates

            # Whitelist matching (using the configured whitelist n-gram range)
            wl_min, wl_max = self.whitelist_ngram_range
            for n in range(wl_min, wl_max + 1):
                keywords.update(self._generate_ngrams(tokens, n))  # Add to set

            # General n-gram extraction (using the configured general n-gram range)
            min_n, max_n = self.ngram_range
            for n in range(min_n, max_n + 1):
                keywords.update(self._generate_ngrams(tokens, n)) # Add to set
            #convert to list
            all_keywords.append(list(keywords))

        # Apply semantic filtering if enabled
        if self.config.get("semantic_validation", False):
            return self._semantic_filter(all_keywords, texts)

        return all_keywords

    def _semantic_filter(self, keyword_lists: List[List[str]], texts: List[str]) -> List[List[str]]:
        """
        Filter keywords based on semantic context.

        Keeps only keywords that appear in a "meaningful" context within the
        corresponding job description.  Context is determined by:
            1. Section headings (e.g., "Responsibilities," "Requirements")
            2. Sentence-level co-occurrence with other relevant terms (using spaCy)

        Args:
            keyword_lists: A list of lists of keywords (one list per job description).
            texts: A list of original job description texts.

        Returns:
            A list of lists of filtered keywords.
        """
        filtered_keyword_lists = []
        for keywords, text in zip(keyword_lists, texts):
            filtered_keywords = [
                keyword for keyword in keywords if self._is_in_context(keyword, text)
            ]
            filtered_keyword_lists.append(filtered_keywords)
        return filtered_keyword_lists

    def _is_in_context(self, keyword: str, text: str) -> bool:
        """
        Check if a keyword appears in a meaningful context within a text.

        Args:
            keyword: The keyword to check.
            text: The text (job description) to search within.

        Returns:
            True if the keyword is considered to be in a meaningful context,
            False otherwise.
        """
        # 1. Check if the keyword is in a relevant section
        sections = self._extract_sections(text)
        for section_name, section_text in sections.items():
            if keyword.lower() in section_text.lower():
                return True  # Found in a section

        # 2. If not found in sections, check all sentences
        doc = self.nlp(text)
        for sent in doc.sents:
            if keyword.lower() in sent.text.lower():
                return True  # Found in a sentence

        return False  # Not found in any relevant context

    def _extract_sections(self, text: str) -> Dict[str, str]:
        """
        Extract sections from a job description using regex patterns.
        """
        # Pre-compile regex patterns for section headings
        if not hasattr(self, '_section_heading_re'):
            self._section_heading_re = re.compile(
                r"^(?:" + "|".join(re.escape(heading) for heading in self.config.get("section_headings", [])) + r")(?:\s*:)?",
                re.MULTILINE | re.IGNORECASE
            )
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


# --- ATSOptimizer Class ---
class ATSOptimizer:
    """
    Main analysis class for optimizing job descriptions for Applicant Tracking Systems (ATS).

    Performs:
        - Configuration loading and validation
        - Keyword extraction (using AdvancedKeywordExtractor)
        - TF-IDF calculation
        - Keyword scoring (with whitelist boosting)
        - Keyword categorization
        - Result generation (DataFrames)
        - Memory management (chunking, cache clearing)
        - Error handling (retries, fallbacks)
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the ATS Optimizer.

        Args:
            config_path: Path to the configuration file (YAML format).
        """
        if sys.version_info < (3, 8):
            raise Exception("Requires Python 3.8+")  # Python version check

        self.config = self._load_config(config_path)  # Load configuration
        self.nlp = self._load_and_configure_spacy_model()  # Load and configure spaCy model
        self._add_entity_ruler()  # Add EntityRuler to spaCy pipeline
        self.keyword_extractor = AdvancedKeywordExtractor(self.config, self.nlp)  # Initialize keyword extractor
        self._init_categories()  # Initialize keyword categories
        self._validate_config()  # Validate configuration

        # Check spaCy version (warning only)
        if spacy.__version__ < "3.0.0":
            logger.warning("spaCy version <3.0 may have compatibility issues")

    def _add_entity_ruler(self):
        """
        Add an EntityRuler to the spaCy pipeline for section detection.

        Defines patterns for common section headings (e.g., "Responsibilities,"
        "Requirements") and adds them to the EntityRuler.  The EntityRuler
        is added *before* the "ner" component in the pipeline.
        """
        if "entity_ruler" not in self.nlp.pipe_names:
            ruler = self.nlp.add_pipe("entity_ruler", before="ner")  # Add before "ner"
            patterns = [  # Define patterns for section headings
                {"label": "SECTION", "pattern": [{"LOWER": "responsibilities"}]},
                {"label": "SECTION", "pattern": [{"LOWER": "requirements"}]},
                {"label": "SECTION", "pattern": [{"LOWER": "skills"}]},
                {"label": "SECTION", "pattern": [{"LOWER": "qualifications"}]},
                {"label": "SECTION", "pattern": [{"LOWER": "experience"}]},
                {"label": "SECTION", "pattern": [{"LOWER": "education"}]},
                {"label": "SECTION", "pattern": [{"LOWER": "benefits"}]},
                {"label": "SECTION", "pattern": [{"LOWER": "about us"}]},
                # Add more patterns as needed
            ]
            ruler.add_patterns(patterns)  # Add the patterns to the ruler

    def _load_config(self, config_path: str) -> Dict:
        """
        Load and validate configuration from a YAML file, with a backup system.

        Loads configuration from the specified path.  If that fails, tries a
        backup file ("config_backup.yaml").  Sets default values for missing
        configuration options.

        Args:
            config_path: Path to the primary configuration file.

        Returns:
            The configuration dictionary.

        Raises:
            ConfigError: If the configuration file is invalid or not found.
        """
        CONFIG_BACKUPS = [config_path, "config_backup.yaml"]  # Primary and backup config files

        for cfg_file in CONFIG_BACKUPS:
            try:
                with open(cfg_file) as f:
                    config = yaml.safe_load(f)  # Load YAML

                    # --- Set Default Values ---
                    # Ensures that all required config options have defaults,
                    # making the script more robust to missing or incomplete config files.
                    config.setdefault(
                        "weighting",  # Default weighting for TF-IDF, frequency, and whitelist
                        {
                            "tfidf_weight": 0.7,
                            "frequency_weight": 0.3,
                            "whitelist_boost": 1.5,
                        },
                    )
                    config.setdefault("spacy_model", "en_core_web_sm")  # Default spaCy model
                    config.setdefault("cache_size", 1000)  # Default cache size
                    config.setdefault("whitelist_ngram_range", [1, 3])  # Default whitelist n-gram range
                    config.setdefault("max_desc_length", 100000)  # Default max description length (~100KB)
                    config.setdefault("timeout", 600)  # Default timeout (10 minutes)
                    config.setdefault("model_download_retries", 2)  # Default model download retries
                    config.setdefault("auto_chunk_threshold", 100)  # Default auto-chunking threshold
                    config.setdefault("memory_threshold", 70)  # Default memory usage threshold
                    config.setdefault("max_memory_percent", 85)  # Default max memory usage percentage
                    config.setdefault("max_workers", 4)  # Default max worker processes
                    config.setdefault("min_chunk_size", 1)  # Default minimum chunk size
                    config.setdefault("max_chunk_size", 1000)  # Default maximum chunk size
                    config.setdefault("max_retries", 2)  # Default maximum retries
                    config.setdefault("strict_mode", True)  # Default strict mode (raise exceptions)
                    config.setdefault("semantic_validation", False)  # Default semantic validation (disabled)
                    config.setdefault("similarity_threshold", 0.6) # Default similarity threshold
                    config.setdefault("validation", {}).setdefault("allow_numeric_titles", True)  # Nested default
                    config.setdefault("text_encoding", "utf-8")

                    return config  # Return the loaded and validated config

            except FileNotFoundError:
                continue  # Try the next backup file
            except yaml.YAMLError as e:
                logger.error(f"Error parsing YAML in {cfg_file}: {e}")
                raise ConfigError(f"Error parsing YAML in {cfg_file}: {e}")
            except Exception as e:
                logger.error(f"Unexpected config error: {str(e)}")
                raise ConfigError(f"Unexpected config error: {str(e)}")
        raise ConfigError(f"No valid config found in: {CONFIG_BACKUPS}")  # No valid config found

    def _init_categories(self):
        """
        Initialize category vectors and metadata.

        Calculates centroids for each keyword category based on the provided
        terms in the configuration.  Handles cases where terms might not have
        valid word vectors.
        """
        self.categories = self.config.get("keyword_categories", {})  # Get keyword categories
        self.category_vectors = {}  # Initialize category vectors

        for category, terms in self.categories.items():
            # Get vectors for terms, filtering out terms without valid vectors
            vectors = [self._get_term_vector(term) for term in terms if self._get_term_vector(term).any()]
            if vectors:  # Check if there are any valid vectors
                self.category_vectors[category] = {
                    "centroid": np.mean(vectors, axis=0),  # Calculate centroid
                    "terms": terms,  # Store the terms
                }
            else:
                logger.warning(f"Category {category} has no valid terms with vectors. Cannot calculate centroid.")
                self.category_vectors[category] = {  # Still initialize, but with None centroid
                    "centroid": None,
                    "terms": terms,
                }

    def _validate_config(self):
        """
        Validate critical configuration parameters using a schema.

        Checks for the presence and correct data types of required and optional
        configuration options.  Issues warnings for potentially problematic
        settings (e.g., a very small skills whitelist).

        Raises:
            ConfigError: If the configuration is invalid.
        """
        # Define the configuration schema (key: (type, required))
        CONFIG_SCHEMA = {
            "skills_whitelist": (list, True),  # Skills whitelist (required)
            "stop_words": (list, True),  # Stop words (required)
            "weighting": (dict, False),  # Weighting parameters (optional)
            "ngram_range": (list, False),  # N-gram range (optional)
            "spacy_model": (str, False),  # spaCy model (optional)
            "cache_size": (int, False),  # Cache size (optional)
            "whitelist_ngram_range": (list, False),  # Whitelist n-gram range (optional)
            "keyword_categories": (dict, False),  # Keyword categories (optional)
            "max_desc_length": (int, False),  # Max description length (optional)
            "min_desc_length": (int, False),  # Min description length (optional)
            "min_jobs": (int, False),  # Min job descriptions (optional)
            "similarity_threshold": (float, False),  # Similarity threshold (optional)
            "timeout": (int, False),  # Timeout (optional)
            "model_download_retries": (int, False),  # Model download retries
            "auto_chunk_threshold": (int, False),  # Auto chunk threshold
            "memory_threshold": (int, False),  # Memory threshold
            "max_memory_percent": (int, False),  # Max memory percent
            "max_workers": (int, False),  # Max workers
            "min_chunk_size": (int, False),  # Minimum chunk size
            "max_chunk_size": (int, False),  # Maximum chunk size
            "max_retries": (int, False),  # Maximum retries
            "strict_mode": (bool, False),  # Strict mode
            "semantic_validation": (bool, False), # Semantic validation
            "validation": (dict, False),  # Validation settings
            "text_encoding": (str, False), # Text encoding
        }

        for key, (type_, required) in CONFIG_SCHEMA.items():
            if required and key not in self.config:
                raise ConfigError(f"Missing required config key: {key}")
            if key in self.config and not isinstance(self.config[key], type_):
                raise ConfigError(
                    f"Invalid type for {key}: expected {type_}, got {type(self.config[key])}"
                )

        if len(self.config["skills_whitelist"]) < 10:
            logger.warning("Skills whitelist seems small. Consider expanding it.")

    def _try_load_model(self, model_name):
        """Attempt to load a spaCy model, disabling unnecessary components."""
        try:
            nlp = spacy.load(model_name, disable=["parser", "ner"])
            if "lemmatizer" not in nlp.pipe_names:
                nlp.add_pipe("lemmatizer", config={"mode": "rule"})
            return nlp
        except OSError:
            return None

    def _download_model(self, model_name):
        """Attempt to download a spaCy model."""
        try:
            spacy.cli.download(model_name)
            return True
        except Exception as e:
            logger.warning(f"  Download failed: {e}")
            return False

    def _create_fallback_model(self):
        """Create a fallback spaCy model (blank 'en' with sentencizer)."""
        try:
            nlp = spacy.blank("en")
            nlp.add_pipe("sentencizer")
        except Exception:
            logger.critical("spaCy sentencizer cannot be added. Check your spaCy installation")
            sys.exit(1)
        if "lemmatizer" not in nlp.pipe_names:
            nlp.add_pipe("lemmatizer", config={"mode": "rule"})
        return nlp

    def _load_and_configure_spacy_model(self):
        """Load and configure the spaCy language model with fallbacks."""
        model_name = self.config.get("spacy_model", "en_core_web_sm")
        retry_attempts = self.config.get("model_download_retries", 2)
        models_to_try = [model_name, "en_core_web_sm"]  # List of models to try

        for model in models_to_try:
            for attempt in range(retry_attempts + 1):
                nlp = self._try_load_model(model)
                if nlp:
                    return nlp
                logger.warning(f"Model '{model}' not found. Attempt {attempt + 1}/{retry_attempts + 1}")
                if attempt < retry_attempts and model == model_name:  # Only download the specified model
                    if self._download_model(model):
                        nlp = self._try_load_model(model)
                        if nlp:
                            return nlp

        logger.warning("Falling back to basic tokenizer.")
        return self.create_fallback_model()

    def _calculate_scores(self, dtm, feature_names, keyword_sets, job_descriptions):
        """
        Calculate keyword scores based on TF-IDF, frequency, and whitelist status.

        Args:
            dtm: Document-term matrix (TF-IDF).
            feature_names: List of feature names (terms) from the vectorizer.
            keyword_sets: List of lists of keywords (one list per job description).
            job_descriptions: List of job titles.

        Returns:
            A list of dictionaries, where each dictionary represents a keyword
            and its associated scores for a specific job description.
        """
        results = []
        for idx, title in enumerate(job_descriptions):
            row = dtm[idx]  # Get the TF-IDF row for the current job description
            keywords = keyword_sets[idx]  # Get the keywords for the current job description

            for col in row.nonzero()[1]:  # Iterate over non-zero elements (terms present in the document)
                term = feature_names[col]  # Get the term
                tfidf = row[0, col]  # Get the TF-IDF score
                freq = keywords.count(term)  # Get the term frequency

                # Calculate the combined score (weighted TF-IDF + frequency)
                score = (
                    self.config["weighting"]["tfidf_weight"] * tfidf
                    + self.config["weighting"]["frequency_weight"] * np.log1p(freq)  # Log-transform frequency
                )

                # Apply whitelist boost if the term is in the whitelist
                if term in self.keyword_extractor.whitelist:
                    score *= self.config["weighting"]["whitelist_boost"]

                # Store the results in a dictionary
                results.append({
                    "Keyword": term,
                    "Job Title": title,
                    "Score": score,
                    "TF-IDF": tfidf,
                    "Frequency": freq,
                    "Category": self._categorize_term(term),  # Categorize the term
                    "In Whitelist": term in self.keyword_extractor.whitelist,  # Boolean flag
                })
        return results

    def _create_tfidf_matrix(self, texts, keyword_sets):
        """
        Create and fit the TF-IDF vectorizer and transform the keyword sets.

        Args:
            texts: List of (preprocessed) job description texts.
            keyword_sets: List of lists of keywords (one list per job description).

        Returns:
            A tuple containing:
                - The document-term matrix (sparse matrix).
                - The list of feature names        """
        # Initialize the TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            ngram_range=self.keyword_extractor.ngram_range,  # Use the configured n-gram range
            lowercase=False,  # Don't lowercase (already done in preprocessing)
            tokenizer=lambda x: x,  # Use a dummy tokenizer (already tokenized)
            preprocessor=lambda x: x,  # Use a dummy preprocessor (already preprocessed)
            max_features=5000,  # Limit the number of features (terms)
            dtype=np.float32,  # Use float32 for memory efficiency
        )
        # Fit and transform the keyword sets (not the original texts)
        dtm = vectorizer.fit_transform([" ".join(kw) for kw in keyword_sets])

        if len(vectorizer.get_feature_names_out()) == 5000:
            logger.warning("TF-IDF vocabulary truncated to 5000 features.  Consider increasing max_features.")
        return dtm, vectorizer.get_feature_names_out()  # Return the DTM and feature names

    def analyze_jobs(self, job_descriptions: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Main analysis method.  Handles chunking, error recovery, and retries."""
        self._validate_input(job_descriptions)

        max_retries = self.config.get("max_retries", 2)
        strict_mode = self.config.get("strict_mode", True)

        for attempt in range(max_retries + 1):
            try:
                if self._needs_chunking(job_descriptions):
                    return self._analyze_jobs_chunked(job_descriptions)
                else:
                    return self._analyze_jobs_internal(job_descriptions)
            except (MemoryError, TimeoutError) as e:
                logger.warning(f"{type(e).__name__} (attempt {attempt + 1}/{max_retries + 1}): {e}")
                self._clear_caches()
                gc.collect()
                if strict_mode and attempt == max_retries:
                    raise
            except Exception as e:
                logger.error(f"An unexpected error occurred: {e}")
                if strict_mode:
                    raise
                else:
                    return pd.DataFrame(), pd.DataFrame()

        logger.critical("All analysis attempts failed due to errors.")
        return pd.DataFrame(), pd.DataFrame()

    def _analyze_jobs_chunked(self, job_descriptions: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Analyze job descriptions in chunks to manage memory usage.

        Splits the job descriptions into smaller chunks, processes each chunk
        in parallel (using multiprocessing), and then aggregates the results.

        Args:
            job_descriptions: A dictionary of job titles and descriptions.

        Returns:
            A tuple of two pandas DataFrames (summary and details).
        """
        chunk_size = self._calculate_chunk_size(job_descriptions)  # Calculate initial chunk size
        # Split job descriptions into chunks
        chunks = [
            dict(list(job_descriptions.items())[i:i + chunk_size])
            for i in range(0, len(job_descriptions), chunk_size)
        ]

        num_workers = min(self.config.get("max_workers", 4), len(chunks))  # Limit workers
        results = []

        with Pool(processes=num_workers) as pool:  # Create a process pool
            # Process chunks in parallel (unordered for efficiency)
            for chunk_result in pool.imap_unordered(self._process_chunk, chunks):
                if chunk_result:  # Handle potential None results (empty chunks)
                    results.append(chunk_result)
                # Dynamic chunk sizing: If memory usage is high, recalculate chunk size
                if psutil.virtual_memory().percent > self.config.get("memory_threshold", 70):
                    chunk_size = self._calculate_chunk_size(job_descriptions)  # Recalculate
                    logger.warning(f"Memory usage is high. Adjusting chunk size to: {chunk_size}")
                    break  # Restart processing with the new chunk size

        # Aggregate results from all chunks
        if not results:  # Handle the case where all chunks were empty
            return pd.DataFrame(), pd.DataFrame()

        summaries, details = zip(*results)  # Unzip the results
        full_summary = pd.concat(summaries)  # Concatenate summary DataFrames
        full_details = pd.concat(details)  # Concatenate details DataFrames

        # Regroup the summary DataFrame (because of chunking)
        final_summary = full_summary.groupby("Keyword").agg({
                    "Total_Score": "sum",
                    "Avg_Score": "mean",
                    "Job_Count": "sum"
                }).sort_values("Total_Score", ascending=False)
        return final_summary, full_details

    def _process_chunk(self, chunk: Dict) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Process a single chunk of job descriptions.

        This function encapsulates the core analysis logic (keyword extraction,
        TF-IDF, scoring) that was previously in `analyze_jobs`.  It handles
        empty chunks and exceptions gracefully.

        Args:
            chunk: A dictionary of job titles and descriptions (a subset of the input).

        Returns:
            A tuple of two pandas DataFrames (summary and details), or None if
            the chunk is empty or an error occurs (and strict mode is disabled).
        """
        if not chunk:
            return None  # Return None for empty chunks

        try:
            # Core analysis logic (same as original analyze_jobs)
            texts = list(chunk.values())
            keyword_sets = self.keyword_extractor.extract_keywords(texts)
            dtm, features = self._create_tfidf_matrix(texts, keyword_sets)
            results = self._calculate_scores(dtm, features, keyword_sets, list(chunk.keys()))

            df = pd.DataFrame(results)
            if df.empty:
                return pd.DataFrame(), pd.DataFrame()  # Handle empty results

            # Create summary and details DataFrames
            summary_chunk = df.groupby("Keyword").agg({
                "Score": ["sum", "mean"],
                "Job Title": "nunique"
            }).rename(columns={
                "Score": {"sum": "Total_Score", "mean": "Avg_Score"},
                "Job Title": {"nunique": "Job_Count"}
            })
            details_chunk = df

            self._memory_check()  # Check memory usage after processing
            return summary_chunk, details_chunk

        except Exception as e:
            logger.error(f"Error processing chunk: {e}")  # Log the error
            if self.config.get("strict_mode", True):
                raise  # Re-raise if in strict mode
            else:
                return None  # Return None if not in strict mode

    def _analyze_jobs_internal(self, job_descriptions: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Original analysis logic for non-chunked processing (single process).

        This is the same analysis logic as the original `analyze_jobs` method,
        used when chunking is not necessary.

        Args:
            job_descriptions: A dictionary of job titles and descriptions.

        Returns:
            A tuple of two pandas DataFrames (summary and details).
        """
        texts = list(job_descriptions.values())  # Get job descriptions
        keyword_sets = self.keyword_extractor.extract_keywords(texts)  # Extract keywords
        dtm, feature_names = self._create_tfidf_matrix(texts, keyword_sets)  # Create TF-IDF matrix
        results = self._calculate_scores(dtm, feature_names, keyword_sets, list(job_descriptions.keys()))  # Calculate scores

        df = pd.DataFrame(results)  # Create DataFrame from results
        # Create summary DataFrame
        summary = df.groupby("Keyword").agg({
            "Score": ["sum", "mean"],
            "Job Title": "nunique"
        }).rename(columns={
            "Score": {"sum": "Total_Score", "mean": "Avg_Score"},
            "Job Title": {"nunique": "Job_Count"}
        }).sort_values("Total_Score", ascending=False)

        # Create details (pivot) DataFrame
        pivot = df.pivot_table(
            values="Score",
            index="Keyword",
            columns="Job Title",
            aggfunc="sum",
            fill_value=0,
        )
        return summary, pivot

    def _needs_chunking(self, jobs: Dict) -> bool:
        """
        Determine if chunking is required based on job count and memory usage.

        Args:
            jobs: A dictionary of job titles and descriptions.

        Returns:
            True if chunking is required, False otherwise.
        """
        num_jobs = len(jobs)  # Number of job descriptions
        auto_chunk_threshold = self.config.get("auto_chunk_threshold", 100)  # Get threshold from config
        memory_percent = psutil.virtual_memory().percent  # Current memory usage
        memory_threshold = self.config.get("memory_threshold", 70)  # Get memory threshold

        return num_jobs > auto_chunk_threshold or memory_percent > memory_threshold

    def _calculate_chunk_size(self, jobs: Dict) -> int:
        """
        Calculate an appropriate chunk size based on available memory.

        Estimates the average size of a job description and calculates a chunk
        size that aims to keep memory usage below a configurable threshold.
        Considers minimum and maximum chunk sizes.

        Args:
            jobs: A dictionary of job titles and descriptions.

        Returns:
            The calculated chunk size.
        """
        if not jobs:
            return 1  # Handle empty input

        total_size = sum(len(desc) for desc in jobs.values())  # Total size of job descriptions
        avg_job_size = total_size / len(jobs)  # Average size per job description
        free_mem = psutil.virtual_memory().available  # Available memory

        # Aim for a chunk size that uses less than max_memory_percent of available RAM
        max_memory_percent = self.config.get("max_memory_percent", 85) / 100
        target_memory_per_chunk = free_mem * max_memory_percent

        # Introduce min and max chunk size (from config)
        min_chunk_size = self.config.get("min_chunk_size", 1)
        max_chunk_size = self.config.get("max_chunk_size", 1000)

        # Calculate chunk size, ensuring it's within the min/max bounds
        chunk_size = max(min_chunk_size, min(max_chunk_size, int(target_memory_per_chunk / (avg_job_size * 2))))  # 2x buffer

        return chunk_size

    def _memory_check(self):
        """
        Check memory usage and clear caches if necessary.

        If memory usage exceeds a configurable threshold, clears all caches
        (spaCy pipeline cache, preprocessor cache, LRU cache) and forces
        garbage collection to free up memory.
        """
        memory_percent = psutil.virtual_memory().percent  # Current memory usage
        memory_threshold = self.config.get("memory_threshold", 70)  # Get threshold
        if memory_percent > memory_threshold:
            logger.warning(f"Memory usage high ({memory_percent:.1f}%). Clearing caches.")
            self._clear_caches()  # Clear caches

    def _clear_caches(self):
        """
        Clear all caches to reduce memory usage.

        Clears:
            - The EnhancedTextPreprocessor's cache.
            - The spaCy pipeline cache.
            - The LRU cache for term vectors.
            - Calls the garbage collector.
        """
        if hasattr(self.keyword_extractor.preprocessor, '_cache'):
            self.keyword_extractor.preprocessor._cache.clear()  # Clear preprocessor cache

        # Clear spaCy pipeline cache (safely)
        if hasattr(self.nlp, 'pipeline'):
            self.nlp.pipeline = []
        if hasattr(self.nlp, 'vocab'):
            self.nlp.vocab.reset_cache()

        # Clear LRU cache (for term vectors)
        if hasattr(self, '_get_term_vector'):
            self._get_term_vector.cache_clear()

        gc.collect()  # Force garbage collection

    def _categorize_term(self, term: str) -> str:
        """
        Categorize a term using a hybrid approach: direct match + semantic similarity.

        First tries to find a direct match (substring) in the predefined
        keyword categories.  If no direct match is found, uses semantic
        similarity (cosine similarity between word vectors) to find the
        closest category.

        Args:
            term: The term to categorize.

        Returns:
            The category name.
        """
        # First try direct matches (substring matching)
        for category, data in self.category_vectors.items():
            if data["terms"] is not None: # Check that terms exists
                if any(keyword.lower() in term.lower() for keyword in data["terms"]):
                    return category

        # If no direct match, use semantic similarity
        return self._semantic_categorization(term)

    @lru_cache(maxsize=5000)  # Cache term vectors (LRU cache)
    def _get_term_vector(self, term: str) -> np.ndarray:
        """
        Get the word vector for a term using spaCy, with error handling.

        Uses an LRU cache to avoid redundant computations.  If vectorization
        fails (e.g., the term is not in spaCy's vocabulary), returns an empty
        NumPy array and logs a warning.

        Args:
            term: The term to vectorize.

        Returns:
            The word vector (NumPy array), or an empty array if vectorization fails.
        """
        try:
            return self.nlp(term).vector  # Get the word vector
        except Exception as e:
            logger.warning(f"Vectorization failed for '{term}': {str(e)}")
            return np.array([])  # Return an empty array

    def _semantic_categorization(self, term: str) -> str:
        """
        Categorize a term using semantic similarity (cosine similarity).

        Calculates the cosine similarity between the term's word vector and
        the centroid of each category.  Assigns the term to the category with
        the highest similarity score (above a configurable threshold).

        Args:
            term: The term to categorize.

        Returns:
            The category name.
        """
        term_vec = self._get_term_vector(term)  # Get the term's vector
        if not term_vec.any():  # Check if the vector is valid
            return "Other"  # Fallback category

        best_score = self.config.get("similarity_threshold", 0.6)  # Get similarity threshold
        best_category = "Other"  # Default category
        valid_categories = 0

        for category, data in self.category_vectors.items():
            if data["centroid"] is not None:  # Check if centroid is valid
                similarity = cosine_similarity(term_vec, data["centroid"])  # Calculate similarity
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

        return best_category  # Return the best category

    def _validate_input(self, raw_jobs: Dict) -> Dict:
        """
        Validate and sanitize the input job descriptions.

        Performs several checks:
            - Ensures the input is a dictionary.
            - Checks for a minimum number of valid job descriptions.
            - Validates each job title (type, length, emptiness).
            - Validates each job description (type, length, encoding, control characters).
            - Sanitizes job titles and descriptions (stripping whitespace, removing URLs/emails/control chars).

        Args:
            raw_jobs: A dictionary of job titles and descriptions.

        Returns:
            A dictionary of *valid* and *sanitized* job titles and descriptions.

        Raises:
            InputValidationError: If the input is invalid.
        """
        valid_jobs = {}  # Store valid job descriptions
        errors = []  # Store validation errors

        for raw_title, raw_desc in raw_jobs.items():
            # Validate title
            title_result = self._validate_title(raw_title)
            # Validate description
            desc_result = self._validate_description(raw_desc)

            # If both title and description are valid, add to valid_jobs
            if title_result.valid and desc_result.valid:
                valid_jobs[title_result.value] = desc_result.value
            else:
                # Accumulate errors
                if not title_result.valid:
                    errors.append(f"Job '{raw_title}': Invalid title - {title_result.reason}")
                if not desc_result.valid:
                    errors.append(f"Job '{raw_title}': Invalid description - {desc_result.reason}")

        # Check if enough valid job descriptions were found
        if len(valid_jobs) < self.config.get("min_jobs", 2):
            error_message = "Insufficient valid job descriptions:\n" + "\n".join(errors)
            logger.error(error_message)
            raise InputValidationError(error_message)

        return valid_jobs  # Return the validated and sanitized job descriptions
    def _validate_title(self, title) -> ValidationResult:
        """Validate and sanitize a single job title."""
        allow_numeric = self.config.get("validation", {}).get("allow_numeric_titles", True)
        if not isinstance(title, str):
            if allow_numeric:
                title = str(title)
                logger.warning(f"Job title converted to string: {title}")
            else:
                return ValidationResult(False, None, "Job title must be a string")

        stripped_title = title.strip()
        if not stripped_title:
            return ValidationResult(False, None, "Job title cannot be empty")

        # Removed: if stripped_title != title: pass

        min_len = self.config.get("validation", {}).get("title_min_length", 2)
        max_len = self.config.get("validation", {}).get("title_max_length", 100)
        if not min_len <= len(stripped_title) <= max_len:
            return ValidationResult(False, None, f"Invalid length (must be {min_len}-{max_len} characters)")

        return ValidationResult(True, stripped_title)
    def _validate_description(self, desc) -> ValidationResult:
        """
        Validate and sanitize a single job description.

        Checks:
            - Type (must be string).
            - Encoding (attempts to decode with specified encoding, replaces invalid characters).
            - Removes URLs, email addresses, and control characters.
            - Length (minimum and maximum).

        Args:
            desc: The job description.

        Returns:
            A ValidationResult object.
        """
        if not isinstance(desc, str):
            return ValidationResult(False, None, "Description must be a string")

        try:
            # Attempt to decode with specified encoding, replace invalid characters
            encoding = self.config.get("text_encoding", "utf-8")
            cleaned_desc = desc.encode(encoding, errors='replace').decode(encoding)
            if cleaned_desc != desc:
                logger.warning(f"Invalid characters replaced in description")
        except Exception as e:
            return ValidationResult(False, None, f"Encoding error: {e}")

        # Remove URLs and email addresses
        cleaned_desc = re.sub(r"http\S+|www\.\S+", "", cleaned_desc, flags=re.UNICODE)
        cleaned_desc = re.sub(r"\S+@\S+", "", cleaned_desc, flags=re.UNICODE)

        # Remove control characters
        cleaned_desc = re.sub(r"[\x00-\x1F\x7F-\x9F]", "", cleaned_desc, flags=re.UNICODE)

        # Keep emojis and accented characters, remove other special characters
        cleaned_desc = re.sub(r"[^\w\s\-😊-🙏🚀-🇿]", " ", cleaned_desc, flags=re.UNICODE)
        cleaned_desc = re.sub(r"\s+", " ", cleaned_desc).strip()  # Normalize whitespace

        min_len = self.config.get("min_desc_length", 50)
        max_len = self.config.get("max_desc_length", 100000)

        if len(cleaned_desc) < min_len:
            logger.warning(f"Description is shorter than minimum length ({min_len})")
            return ValidationResult(True, cleaned_desc, "Description is short")  # Return as valid, but with reason

        if len(cleaned_desc) > max_len:
            return ValidationResult(False, None, f"Description is longer than maximum length ({max_len})")

        return ValidationResult(True, cleaned_desc)  # Valid description


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate the cosine similarity between two vectors.

    Handles cases where either vector has a zero norm (returns 0.0 in that case).

    Args:
        vec1: The first vector (NumPy array).
        vec2: The second vector (NumPy array).

    Returns:
        The cosine similarity (a float between -1 and 1).
    """
    norm1 = np.linalg.norm(vec1)  # Calculate the norm (magnitude) of the first vector
    norm2 = np.linalg.norm(vec2)  # Calculate the norm of the second vector
    if norm1 == 0 or norm2 == 0:  # Handle zero-norm vectors
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)  # Calculate cosine similarity


# --- Command Line Interface ---
def parse_arguments():
    """
    Parse command-line arguments.

    Defines the command-line interface for the script, allowing the user to
    specify input, configuration, and output files.

    Returns:
        An argparse.Namespace object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description="ATS Keyword Optimizer")
    parser.add_argument("-i", "--input", default="job_descriptions.json", help="Input JSON file")
    parser.add_argument("-c", "--config", default="config.yaml", help="Config file")
    parser.add_argument("-o", "--output", default="results.xlsx", help="Output file")
    return parser.parse_args()

def initialize_analyzer(config_path: str):
    """
    Initialize the ATSOptimizer, including NLTK resources.

    Args:
        config_path: Path to the configuration file.

    Returns:
        An instance of the ATSOptimizer.
    """
    ensure_nltk_resources()  # Ensure NLTK resources are downloaded
    return ATSOptimizer(config_path)  # Initialize and return the ATSOptimizer

def save_results(summary: pd.DataFrame, details: pd.DataFrame, output_file: str):
    """
    Save the analysis results to an Excel file.

    Creates an Excel file with two sheets: "Summary" and "Detailed Scores".

    Args:
        summary: Summary DataFrame.
        details: Details DataFrame.
        output_file: Path to the output Excel file.
    """
    with pd.ExcelWriter(output_file) as writer:
        summary.to_excel(writer, sheet_name="Summary")  # Save summary to "Summary" sheet
        details.to_excel(writer, sheet_name="Detailed Scores")  # Save details to "Detailed Scores" sheet
    print(f"Analysis complete. Results saved to {output_file}")

def load_job_data(input_file: str) -> Dict:
    """
    Load and validate job data from a JSON file.

    Args:
        input_file: Path to the input JSON file.

    Returns:
        A dictionary of job titles and descriptions.

    Raises:
        SystemExit: If the input file is not found or contains invalid JSON.
    """
    try:
        with open(input_file) as f:
            jobs = json.load(f)  # Load JSON data
        return jobs
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {input_file}: {e}")
        sys.exit(1)  # Exit with an error code
    except FileNotFoundError as e:
        logger.error(f"Input file not found {input_file}: {e}")
        sys.exit(1)  # Exit with an error code

def run_analysis(args):
    """Runs the core analysis logic."""
    analyzer = initialize_analyzer(args.config)
    jobs = load_job_data(args.input)
    summary, details = analyzer.analyze_jobs(jobs)
    save_results(summary, details, args.output)

def main():
    """Main function to run the ATS keyword analysis."""
    if sys.version_info < (3, 8):
        logger.error("Requires Python 3.8+")
        sys.exit(1)

    args = parse_arguments()

    try:
        run_analysis(args)  # Call the new function

    except ConfigError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except InputValidationError as e:
        logger.error(f"Input validation error: {e}")
        sys.exit(1)
    except MemoryError as e:
        logger.error(f"Memory error: {e}")
        sys.exit(1)
    except TimeoutError as e:
        logger.error(f"Timeout error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(
            f"""
            CRITICAL ERROR: {str(e)}
            System Info:
            - Python: {sys.version}
            - Platform: {platform.platform()}
            - Memory: {psutil.virtual_memory().percent}% used
            - CPU: {psutil.cpu_percent()}% utilization
            """,
        )
        sys.exit(1)

if __name__ == "__main__":
    main()  # Run the main function if the script is executed directly"
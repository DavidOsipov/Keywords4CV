import argparse  # For command-line argument parsing
import hashlib  # For generating hashes of configuration for caching
import json  # For handling JSON data (input and config hashing)
import logging  # For logging messages (info, warnings, errors)
import re  # For regular expressions (text cleaning)
import sys  # For system-specific parameters and functions (exiting)
from collections import OrderedDict  # For creating an LRU cache
from typing import Dict, List, Set, Tuple, NamedTuple, Optional  # For type hinting
from multiprocessing import (
    TimeoutError as MPTimeoutError,
)  # For parallel processing
import concurrent.futures  # For multithreading category vector calculation
import platform  # For getting platform information (in error messages)
import nltk  # Natural Language Toolkit (for WordNet and other resources)
import pandas as pd  # For data manipulation and analysis (DataFrames)
import spacy  # For natural language processing (tokenization, lemmatization, etc.)
import yaml  # For reading the configuration file (YAML format)
from nltk.corpus import wordnet  # For accessing WordNet lexical database
from sklearn.feature_extraction.text import TfidfVectorizer  # For TF-IDF calculations
from functools import lru_cache  # For caching function results (term vectors)
import psutil  # For monitoring system resources (memory, CPU)
import gc  # For garbage collection
import numpy as np
import os
from multiprocessing import current_process


# --- Custom Exceptions ---
class ConfigError(Exception):
    """Custom exception for configuration errors.
    Raised when there is an issue with the configuration file (e.g., invalid format, missing keys).
    """


class InputValidationError(Exception):
    """Custom exception for input validation errors.
    Raised when the input job descriptions are invalid (e.g., incorrect format, missing data).
    """


# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,  # Log level: INFO, WARNING, ERROR, CRITICAL
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log message format
    handlers=[
        logging.FileHandler("ats_optimizer.log"),
        logging.StreamHandler(),
    ],  # Logs to file and console
)
logger = logging.getLogger(__name__)  # Get the root logger


# --- NLTK Resource Management ---
NLTK_RESOURCES = [
    "corpora/wordnet",  # WordNet lexical database
    "corpora/averaged_perceptron_tagger",  # POS tagger
    "tokenizers/punkt",  # Sentence tokenizer
]


def ensure_nltk_resources():
    """Ensure required NLTK resources are downloaded. If not, download them.

    Downloads the necessary NLTK resources (WordNet, POS tagger, sentence tokenizer)
    if they are not already present on the system.  This ensures that the script can
    function correctly even if the user does not have these resources installed.
    """
    for resource in NLTK_RESOURCES:
        try:
            nltk.data.find(resource)  # Check if the resource is already downloaded
        except LookupError:
            logger.info(f"Downloading NLTK resource: {resource}")
            nltk.download(resource.split("/")[1], quiet=True)  # Download the resource


# --- Validation Result NamedTuple ---
class ValidationResult(NamedTuple):
    """Represents the result of a validation check.

    Attributes:
        valid: True if the validation passed, False otherwise.
        value: The validated/sanitized value (if valid), otherwise None.
        reason: The reason for failure (if invalid), otherwise None.
    """

    valid: bool  # True if validation passed, False otherwise
    value: Optional[str]  # The validated/sanitized value (if valid), otherwise None
    reason: Optional[str] = None  # Reason for failure (if invalid), otherwise None


# --- EnhancedTextPreprocessor Class ---
class EnhancedTextPreprocessor:
    """
    Optimized text preprocessing with memoization (caching) and batch processing.
    Handles stop word removal, URL/email/special character removal, lowercasing,
    lemmatization (using spaCy), batch processing and caching.
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
        self.stop_words = (
            self._load_stop_words()
        )  # Load stop words from the configuration
        self.regex_patterns = {  # Pre-compile regular expressions for efficiency
            "url": re.compile(r"http\S+|www\.\S+"),  # Regular expression for URLs
            "email": re.compile(r"\S+@\S+"),  # Regular expression for email addresses
            "special_chars": re.compile(
                r"[^\w\s\-]"
            ),  # Regular expression for special characters
            "whitespace": re.compile(
                r"\s+"
            ),  # Regular expression for multiple whitespaces
        }
        self._cache = OrderedDict()  # Use OrderedDict as an LRU cache
        self._CACHE_SIZE = max(
            100, config.get("cache_size", 1000)
        )  # Maximum cache size (at least 100)
        self.config_hash = None  # Initialize config hash to None
        self._update_config_hash()  # Calculate initial config hash

    def _update_config_hash(self):
        """Update the config hash if the configuration has changed."""
        new_hash = self._calculate_config_hash()
        if new_hash != self.config_hash:
            self.config_hash = new_hash
            self._cache.clear()  # Clear the cache if the config has changed

    def _calculate_config_hash(self) -> str:
        """Calculate a hash of the relevant configuration parts for cache invalidation.

        Returns:
            A SHA256 hash of the relevant configuration options.
        """
        relevant_config = {  # Only include config options that affect preprocessing
            "stop_words": self.config.get("stop_words", []),
            "stop_words_add": self.config.get("stop_words_add", []),
            "stop_words_exclude": self.config.get("stop_words_exclude", []),
        }
        config_str = json.dumps(relevant_config, sort_keys=True)  # Serialize to JSON
        return hashlib.sha256(
            config_str.encode("utf-8")
        ).hexdigest()  # Calculate SHA256 hash

    def _load_stop_words(self) -> Set[str]:
        """Load and validate stop words from the configuration.

        Returns:
            A set of stop words.
        """
        stop_words = set(self.config.get("stop_words", []))  # Get base stop words
        stop_words.update(
            self.config.get("stop_words_add", [])
        )  # Add additional stop words
        stop_words.difference_update(
            self.config.get("stop_words_exclude", [])
        )  # Remove excluded words

        if len(stop_words) < 50:
            logger.warning(
                "Stop words list seems unusually small (less than 50 words). Consider adding more stop words to improve text preprocessing."
            )  # Warn if stop word list is too small
        return stop_words

    def preprocess(self, text: str) -> str:
        self._update_config_hash()  # Update config hash if needed
        if text in self._cache:
            self._cache.move_to_end(text)
            return self._cache[text]

        # Perform cleaning steps
        cleaned = text.lower()
        cleaned = self.regex_patterns["url"].sub("", cleaned)
        cleaned = self.regex_patterns["email"].sub("", cleaned)
        cleaned = self.regex_patterns["special_chars"].sub(" ", cleaned)
        cleaned = self.regex_patterns["whitespace"].sub(" ", cleaned).strip()

        # Check cache size BEFORE inserting—ensuring LRU size is enforced
        while len(self._cache) >= self._CACHE_SIZE:
            self._cache.popitem(last=False)
        self._cache[text] = cleaned
        return cleaned

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """Preprocess a list of text strings.

        Args:
            texts: A list of text strings.

        Returns:
            A list of preprocessed text strings.
        """
        return [
            self.preprocess(text) for text in texts
        ]  # Preprocess each text in the list

    def _process_doc_tokens(self, doc):
        tokens = []
        skill_spans = []
        # First, add SKILL entity texts as tokens and record their spans.
        for ent in doc.ents:
            if ent.label_ == "SKILL":
                tokens.append(ent.text)
                skill_spans.append((ent.start, ent.end))
        # Process remaining tokens that are not part of a SKILL entity.
        for i, token in enumerate(doc):
            if any(start <= i < end for start, end in skill_spans):
                continue
            if token.text.lower() in self.stop_words or len(token.text) <= 1 or token.text.isnumeric():
                continue
            try:
                lemma = token.lemma_.lower().strip()
            except AttributeError:
                lemma = token.text.lower().strip()
            tokens.append(lemma)
        return tokens

    def tokenize_batch(self, texts: List[str]) -> List[List[str]]:
        try:
            # Guarantee at least one process
            n_process = max(1, min(os.cpu_count() or 1, 4))
            if current_process().daemon:
                logger.warning("Daemon process detected – forcing single-process tokenization.")
                n_process = 1
        except Exception:
            n_process = 1

        tokens_list = []
        available_memory = psutil.virtual_memory().available
        memory_per_process = self.config.get("memory_per_process", 50 * 1024 * 1024)
        safe_memory_per_process = available_memory // (n_process + 1) if n_process > 0 else available_memory
        batch_size = min(max(1, safe_memory_per_process // memory_per_process), 100)

        if spacy.__version__ >= "3.0.0":
            docs = self.nlp.pipe(texts, batch_size=batch_size, n_process=n_process)
        else:
            docs = self.nlp.pipe(texts, batch_size=batch_size)

        for doc in docs:
            tokens_list.append(self._process_doc_tokens(doc))
        return tokens_list


# --- AdvancedKeywordExtractor Class ---
class AdvancedKeywordExtractor:
    """
    Enhanced keyword extraction with n-gram generation and semantic analysis.
    """

    def __init__(self, config: Dict, nlp):
        """Initialize the AdvancedKeywordExtractor.

        Args:
            config: Configuration dictionary.
            nlp: spaCy language model instance.
        """
        self.config = config  # Store the configuration
        self.nlp = nlp  # Store the spaCy model instance
        self.ngram_range = tuple(config.get("ngram_range", [1, 3]))  # Set n-gram range
        self.whitelist_ngram_range = tuple(
            config.get("whitelist_ngram_range", [1, 3])
        )  # Set whitelist n-gram range
        self.preprocessor = EnhancedTextPreprocessor(
            config, nlp
        )  # Initialize the text preprocessor
        self.whitelist = (
            self._create_expanded_whitelist()
        )  # Create the expanded whitelist
        self._section_heading_re = re.compile(  # Compile regex for section headings
            r"^(?:"
            + "|".join(
                re.escape(heading)
                for heading in self.config.get("section_headings", [])
            )
            + r")(?:\s*:)?",
            re.MULTILINE | re.IGNORECASE,
        )

    def _create_expanded_whitelist(self) -> Set[str]:
        """Create an expanded whitelist with n-grams and synonyms.

        Returns:
            A set of whitelisted terms.
        """
        base_skills = self.config.get(
            "skills_whitelist", []
        )  # Get base skills from the config
        processed = set()  # Initialize the set of processed skills
        cleaned_skills = self.preprocessor.preprocess_batch(
            base_skills
        )  # Preprocess batch skills
        tokenized = self.preprocessor.tokenize_batch(
            cleaned_skills
        )  # Tokenize batch skills

        for tokens in tokenized:  # Iterate through tokenized skills
            for n in range(
                self.whitelist_ngram_range[0], self.whitelist_ngram_range[1] + 1
            ):  # Iterate through n-gram range
                processed.update(
                    self._generate_ngrams(tokens, n)
                )  # Update the processed set with generated n-grams
        synonyms = self._generate_synonyms(
            base_skills
        )  # Generate synonyms for base skills
        processed.update(synonyms)  # Update the processed set with synonyms
        return processed  # Return the processed set

    def _generate_synonyms(self, skills: List[str]) -> Set[str]:
        """Generate synonyms for a list of skills using WordNet.

        Args:
            skills: A list of skill terms.

        Returns:
            A set of synonyms.
        """
        synonyms = set()  # Initialize the set of synonyms
        for skill in skills:  # Iterate through the skills
            if not skill.strip():  # Check if the skill is empty
                logger.warning(
                    "Skipping empty skill in whitelist"
                )  # Warn if the skill is empty
                continue  # Continue to the next skill
            doc = self.nlp(skill)  # Process the skill with spaCy
            lemmatized = " ".join(
                [token.lemma_ for token in doc]
            ).lower()  # Lemmatize the skill
            if (
                lemmatized != skill.lower()
            ):  # Check if the lemmatized skill is different from the original skill
                synonyms.add(
                    lemmatized
                )  # Add the lemmatized skill to the set of synonyms
            for token in doc:  # Iterate through the tokens in the skill
                if token.text.strip():  # Check if the token is empty
                    synonyms.update(  # Update the set of synonyms with synonyms from WordNet
                        lemma.name().replace("_", " ").lower()
                        for syn in wordnet.synsets(token.text)
                        for lemma in syn.lemmas()
                        if lemma.name().replace("_", " ").lower()
                        not in (token.text.lower(), lemmatized)
                    )
        return synonyms  # Return the set of synonyms

    # --- Improved _generate_ngrams function ---
    def _generate_ngrams(self, tokens: List[str], n: int) -> Set[str]:
        """Generate n-grams from tokens and exclude any n-grams that contain single-letter words.
        
        This extra filter ensures that even if a single-letter slips through tokenization,
        the n-grams used for TF-IDF contain only valid multi-character words.
        """
        # Remove any tokens that are empty after stripping whitespace.
        filtered_tokens = [
            token for token in tokens
            if token.strip() and len(token.strip()) > 1 and token not in self.preprocessor.stop_words
        ]
        if len(filtered_tokens) < n:
            return set()
        # Generate all n-grams from the filtered tokens.
        ngrams = {
            " ".join(filtered_tokens[i : i + n])
            for i in range(len(filtered_tokens) - n + 1)
            if all(len(word.strip()) > 1 for word in filtered_tokens[i : i + n])
        }
        return ngrams

    def extract_keywords(self, texts: List[str]) -> List[List[str]]:
        all_keywords = []
        for text in texts:
            doc = self.nlp(text)
            # Extract preserved SKILL entities.
            entity_keywords = [ent.text for ent in doc.ents if ent.label_ == "SKILL"]
            
            # Exclude SKILL entities from regular tokenization.
            skill_spans = [(ent.start, ent.end) for ent in doc.ents if ent.label_ == "SKILL"]
            non_entity_tokens = []
            for i, token in enumerate(doc):
                if any(start <= i < end for start, end in skill_spans):
                    continue
                non_entity_tokens.append(token.text)
            
            # Preprocess and split to get cleaned tokens.
            preprocessed_text = self.preprocessor.preprocess(" ".join(non_entity_tokens))
            token_list = preprocessed_text.split()
            
            non_entity_keywords = set()
            # Generate n-grams only for non-entity tokens.
            for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
                non_entity_keywords.update(self._generate_ngrams(token_list, n))
            
            # Combine entity and non-entity keywords.
            keywords = set(entity_keywords) | non_entity_keywords
            filtered_keywords = [
                kw for kw in keywords
                if len(kw.strip()) > 1
                and not any(len(w.strip()) <= 1 for w in kw.split())
                and not all(w in self.preprocessor.stop_words for w in kw.split())
            ]
            all_keywords.append(filtered_keywords)
        
        if self.config.get("semantic_validation", False):
            return self._semantic_filter(all_keywords, texts)
        return all_keywords

    def _semantic_filter(
        self, keyword_lists: List[List[str]], texts: List[str]
    ) -> List[List[str]]:
        """Filter keywords based on semantic context.

        Args:
            keyword_lists: A list of lists of keywords.
            texts: A list of text strings.

        Returns:
            A list of lists of filtered keywords.
        """
        filtered_keyword_lists = []  # Initialize the list of filtered keywords
        for keywords, text in zip(
            keyword_lists, texts
        ):  # Iterate through the keywords and text strings
            filtered_keywords = [
                keyword for keyword in keywords if self._is_in_context(keyword, text)
            ]  # Filter the keywords based on semantic context
            filtered_keyword_lists.append(
                filtered_keywords
            )  # Add the filtered keywords to the list
        return filtered_keyword_lists  # Return the list of filtered keywords

    def _is_in_context(self, keyword: str, text: str) -> bool:
        # Use regex word boundaries to avoid partial matches
        pattern = re.compile(rf'\b{re.escape(keyword)}\b', re.IGNORECASE)
        if pattern.search(text):
            return True
        # Fallback: check sentence by sentence
        doc = self.nlp(text)
        for sent in doc.sents:
            if pattern.search(sent.text):
                return True
        return False

    def _extract_sections(self, text: str) -> Dict[str, str]:
        """Extract sections from a text string based on section headings.

        Args:
            text: The text string to extract sections from.

        Returns:
            A dictionary of sections, where the keys are section headings and the values are section text.
        """
        doc = self.nlp(text)  # Process the text with spaCy
        sections = {}  # Initialize the dictionary of sections
        current_section = "General"  # Initialize the current section to "General"
        sections[current_section] = ""  # Initialize the text for the current section
        for sent in doc.sents:  # Iterate through the sentences in the text
            match = self._section_heading_re.match(
                sent.text
            )  # Check if the sentence matches a section heading
            if match:  # If the sentence matches a section heading
                current_section = (
                    match.group(0).strip().rstrip(":")
                )  # Get the section heading from the match
                sections[current_section] = (
                    ""  # Initialize the text for the new section
                )
            sections[current_section] += (
                " " + sent.text.strip()
            )  # Add the sentence to the text for the current section
        return sections  # Return the dictionary of sections

    @lru_cache(maxsize=10000)
    def _categorize_term(self, term: str) -> str:
        """Categorize a term using a hybrid approach (direct match + semantic similarity).

        Args:
            term: The term to categorize.

        Returns:
            The category name.
        """
        for category, data in self.category_vectors.items():
            if data["terms"] is not None:
                if any(keyword.lower() in term.lower() for keyword in data["terms"]):
                    return category
        return self._semantic_categorization(term)

    @lru_cache(maxsize=5000)
    def _get_term_vector(self, term: str) -> np.ndarray:
        """Get the word vector for a term using spaCy.

        Args:
            term: The term to vectorize.

        Returns:
            The word vector (NumPy array), or an empty array if vectorization fails.
        """
        try:
            return self.nlp(term).vector
        except AttributeError as e:
            logger.warning(
                f"AttributeError during vectorization for '{term}': {str(e)}"
            )
            return np.array([])
        except ValueError as e:
            logger.warning(f"ValueError during vectorization for '{term}': {str(e)}")
            return np.array([])
        except TypeError as e:
            logger.warning(f"TypeError during vectorization for '{term}': {str(e)}")
            return np.array([])
        except Exception as e:
            logger.warning(
                f"Unexpected error during vectorization for '{term}': {str(e)}"
            )
            return np.array([])

    def _semantic_categorization(self, term: str) -> str:
        """Categorize a term using semantic similarity (cosine similarity).

        Args:
            term: The term to categorize.

        Returns:
            The category name.
        """
        term_vec = self._get_term_vector(term)
        if not term_vec.any():
            return "Other"
        best_score = self.config.get("similarity_threshold", 0.6)
        best_category = "Other"
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


# --- ATSOptimizer Class ---
class ATSOptimizer:
    """
    Main analysis class for optimizing job descriptions for ATS.
    """

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the ATSOptimizer.

        Args:
            config_path: Path to the configuration file (YAML format).
        """
        if sys.version_info < (3, 8):  # Check the Python version
            raise Exception(
                "Requires Python 3.8+"
            )  # Raise an exception if the Python version is not supported

        self.config = self._load_config(
            config_path
        )  # Load the configuration from the configuration file
        self.nlp = (
            self._load_and_configure_spacy_model()
        )  # Load and configure the spaCy model
        self._add_entity_ruler()  # Add the entity ruler to the spaCy pipeline
        self.keyword_extractor = AdvancedKeywordExtractor(
            self.config, self.nlp
        )  # Initialize the keyword extractor
        self._init_categories()  # Initialize the keyword categories
        self._validate_config()  # Validate the configuration
        if (
            "entity_ruler" not in self.nlp.pipe_names
        ):  # Check if the entity ruler is already in the pipeline
            if (
                "ner" in self.nlp.pipe_names
            ):  # Check if the named entity recognizer is in the pipeline
                self.nlp.add_pipe(
                    "entity_ruler", before="ner"
                )  # Add the entity ruler before the named entity recognizer
            else:
                self.nlp.add_pipe(
                    "entity_ruler"
                )  # Add the entity ruler to the end of the pipeline
        if spacy.__version__ < "3.0.0":  # Check the spaCy version
            logger.warning(
                "spaCy version <3.0 may have compatibility issues"
            )  # Warn if the spaCy version is too low

        whitelisted_phrases = self.config.get("skills_whitelist", [])
        if whitelisted_phrases and "entity_ruler" in self.nlp.pipe_names:
            patterns = [{"label": "SKILL", "pattern": phrase} for phrase in whitelisted_phrases]
            self.nlp.get_pipe("entity_ruler").add_patterns(patterns)

    def _add_entity_ruler(self):
        """Add an EntityRuler to the spaCy pipeline for section detection.

        Defines patterns for common section headings (e.g., "Responsibilities,"
        "Requirements") and adds them to the EntityRuler.  The EntityRuler
        is added *before* the "ner" component in the pipeline.
        """
        if (
            "entity_ruler" not in self.nlp.pipe_names
        ):  # Check if an entity ruler is already present
            ruler = self.nlp.add_pipe(
                "entity_ruler", before="ner"
            )  # Add the entity ruler before the "ner" component
            patterns = [  # Define patterns for section headings
                {
                    "label": "SECTION",
                    "pattern": [{"LOWER": "responsibilities"}],
                },  # Pattern for "responsibilities"
                {
                    "label": "SECTION",
                    "pattern": [{"LOWER": "requirements"}],
                },  # Pattern for "requirements"
                {
                    "label": "SECTION",
                    "pattern": [{"LOWER": "skills"}],
                },  # Pattern for "skills"
                {
                    "label": "SECTION",
                    "pattern": [{"LOWER": "qualifications"}],
                },  # Pattern for "qualifications"
                {
                    "label": "SECTION",
                    "pattern": [{"LOWER": "experience"}],
                },  # Pattern for "experience"
                {
                    "label": "SECTION",
                    "pattern": [{"LOWER": "education"}],
                },  # Pattern for "education"
                {
                    "label": "SECTION",
                    "pattern": [{"LOWER": "benefits"}],
                },  # Pattern for "benefits"
                {
                    "label": "SECTION",
                    "pattern": [{"LOWER": "about us"}],
                },  # Pattern for "about us"
            ]
            ruler.add_patterns(patterns)  # Add the patterns to the entity ruler

    def _load_config(self, config_path: str) -> Dict:
        """Load and validate configuration from a YAML file.

        Args:
            config_path: Path to the configuration file.

        Returns:
            The configuration dictionary.
        """
        CONFIG_BACKUPS = [
            config_path,
            "config_backup.yaml",
        ]  # List of configuration file backups
        for (
            cfg_file
        ) in CONFIG_BACKUPS:  # Iterate through the configuration file backups
            try:  # Try to load the configuration file
                with open(cfg_file) as f:  # Open the configuration file
                    config = yaml.safe_load(f)  # Load the configuration from the file

                    config.setdefault(  # Set default values for the configuration
                        "weighting",  # Default weighting for TF-IDF, frequency, and whitelist
                        {
                            "tfidf_weight": 0.7,  # Default weight for TF-IDF
                            "frequency_weight": 0.3,  # Default weight for frequency
                            "whitelist_boost": 1.5,  # Default boost for whitelist
                        },
                    )
                    config.setdefault(
                        "spacy_model", "en_core_web_sm"
                    )  # Default spaCy model
                    config.setdefault("cache_size", 1000)  # Default cache size
                    config.setdefault(
                        "whitelist_ngram_range", [1, 3]
                    )  # Default whitelist n-gram range
                    config.setdefault(
                        "max_desc_length", 100000
                    )  # Default maximum description length
                    config.setdefault("timeout", 600)  # Default timeout
                    config.setdefault(
                        "model_download_retries", 2
                    )  # Default model download retries
                    config.setdefault(
                        "auto_chunk_threshold", 100
                    )  # Default auto-chunking threshold
                    config.setdefault(
                        "memory_threshold", 70
                    )  # Default memory threshold
                    config.setdefault(
                        "max_memory_percent", 85
                    )  # Default maximum memory percent
                    config.setdefault("max_workers", 4)  # Default maximum workers
                    config.setdefault("min_chunk_size", 1)  # Default minimum chunk size
                    config.setdefault(
                        "max_chunk_size", 1000
                    )  # Default maximum chunk size
                    config.setdefault("max_retries", 2)  # Default maximum retries
                    config.setdefault("strict_mode", True)  # Default strict mode
                    config.setdefault(
                        "semantic_validation", False
                    )  # Default semantic validation
                    config.setdefault(
                        "similarity_threshold", 0.6
                    )  # Default similarity threshold
                    config.setdefault("text_encoding", "utf-8")  # Default text encoding
                    return config  # Return the configuration

            except FileNotFoundError:  # Handle file not found errors
                continue  # Continue to the next configuration file backup
            except yaml.YAMLError as e:  # Handle YAML errors
                logger.error(f"Error parsing YAML in {cfg_file}: {e}")  # Log the error
                raise ConfigError(
                    f"Error parsing YAML in {cfg_file}: {e}")
            except Exception as e:  # Handle other exceptions
                logger.error(f"Unexpected config error: {str(e)}")  # Log the error
                raise ConfigError(
                    f"Unexpected config error: {str(e)}"
                )  # Raise a configuration error

        raise ConfigError(
            f"No valid config found in: {CONFIG_BACKUPS}"
        )  # Raise a configuration error if no valid config is found

    def _init_categories(self):
        """Initialize category vectors and metadata."""
        self.categories = self.config.get(
            "keyword_categories", {}
        )  # Get keyword categories
        self.keyword_extractor.category_vectors = {}  # Initialize category vectors in keyword extractor

        def calculate_category_vector(category, terms):
            """Calculate the centroid vector for a category.

            Args:
                category: The category name.
                terms: A list of terms associated with the category.

            Returns:
                A tuple containing the category name and a dictionary with the centroid vector and terms.
            """
            vectors = [
                self.keyword_extractor._get_term_vector(term)
                for term in terms
                if self.keyword_extractor._get_term_vector(term).any()
            ]  # Get term vectors
            if vectors:  # Check if there are any valid vectors
                return category, {
                    "centroid": np.mean(vectors, axis=0),
                    "terms": terms,
                }  # Return the category and centroid
            else:
                logger.warning(
                    f"Category {category} has no valid terms with vectors. Cannot calculate centroid."
                )  # Warn if no valid terms
                return category, {
                    "centroid": None,
                    "terms": terms,
                }  # Return the category and None for centroid

        with concurrent.futures.ThreadPoolExecutor() as executor:
            """Use a thread pool executor to calculate category vectors concurrently."""
            future_to_category = {
                executor.submit(calculate_category_vector, category, terms): category
                for category, terms in self.categories.items()
            }
            for future in concurrent.futures.as_completed(future_to_category):
                category, data = future.result()
                self.keyword_extractor.category_vectors[category] = data

    def _validate_config(self):
        """Validate the configuration parameters."""
        CONFIG_SCHEMA = {  # Define the configuration schema
            "skills_whitelist": (list, True),  # Skills whitelist (required)
            "stop_words": (list, True),  # Stop words (required)
            "weighting": (dict, False),  # Weighting parameters (optional)
            "ngram_range": (list, False),  # N-gram range (optional)
            "spacy_model": (str, False),  # spaCy model (optional)
            "cache_size": (int, False),  # Cache size (optional)
            "whitelist_ngram_range": (list, False),  # Whitelist n-gram range (optional)
            "keyword_categories": (dict, False),  # Keyword categories (optional)
            "max_desc_length": (int, False),  # Maximum description length (optional)
            "min_desc_length": (int, False),  # Minimum description length (optional)
            "min_jobs": (int, False),  # Minimum job descriptions (optional)
            "similarity_threshold": (float, False),  # Similarity threshold (optional)
            "timeout": (int, False),  # Timeout (optional)
            "model_download_retries": (int, False),  # Model download retries
            "auto_chunk_threshold": (int, False),  # Auto chunk threshold
            "memory_threshold": (int, False),  # Memory threshold
            "max_memory_percent": (int, False),  # Maximum memory percent
            "max_workers": (int, False),  # Maximum workers
            "min_chunk_size": (int, False),  # Minimum chunk size
            "max_chunk_size": (int, False),  # Maximum chunk size
            "max_retries": (int, False),  # Maximum retries
            "strict_mode": (bool, False),  # Strict mode
            "semantic_validation": (bool, False),  # Semantic validation
            "validation": (dict, False),  # Validation settings
            "text_encoding": (str, False),  # Text encoding
        }
        for key, (
            expected_type,
            required,
        ) in CONFIG_SCHEMA.items():  # Iterate through the configuration schema
            if (
                required and key not in self.config
            ):  # Check if the key is required and not in the configuration
                raise ConfigError(
                    f"Missing required config key: {key}"
                )  # Raise a configuration error
            if key in self.config and not isinstance(
                self.config[key], expected_type
            ):  # Check if the key is in the configuration and the type is incorrect
                raise ConfigError(
                    f"Invalid type for {key}: expected {expected_type}, got {type(self.config[key])}"
                )  # Raise a configuration error
        if (
            len(self.config["skills_whitelist"]) < 10
        ):  # Check if the skills whitelist is too small
            logger.warning(
                "Skills whitelist seems small. Consider expanding it."
            )  # Warn if the skills whitelist is too small

    def _try_load_model(self, model_name):
        """Attempt to load a spaCy model, disabling unnecessary components."""
        try:
            nlp = spacy.load(model_name, disable=["parser", "ner"])
            if "lemmatizer" not in nlp.pipe_names:
                nlp.add_pipe("lemmatizer", config={"mode": "rule"})
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
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
            logger.critical(
                "spaCy sentencizer cannot be added. Check your spaCy installation"
            )
            sys.exit(1)
        if "lemmatizer" not in nlp.pipe_names:
            nlp.add_pipe("lemmatizer", config={"mode": "rule"})
        return nlp

    def _load_and_configure_spacy_model(self):
        """Load and configure the spaCy language model with fallbacks."""
        model_name = self.config.get("spacy_model", "en_core_web_sm")
        models_to_try = [model_name]
        retry_attempts = self.config.get("model_download_retries", 2)
        import time

        for model in models_to_try:
            for attempt in range(retry_attempts + 1):
                nlp = self._try_load_model(model)
                if nlp:
                    logger.debug(f"Loaded spaCy pipeline: {nlp.pipe_names}")
                    return nlp
                logger.warning(
                    f"Model '{model}' not found. Attempt {attempt + 1}/{retry_attempts + 1}"
                )
                if attempt < retry_attempts and model == model_name:
                    if self._download_model(model):
                        nlp = self._try_load_model(model)
                        if nlp:
                            logger.debug(f"Loaded spaCy pipeline: {nlp.pipe_names}")
                            return nlp
                time.sleep(2)  # Add a delay of 2 seconds between retries

        logger.warning("Falling back to basic tokenizer.")
        return self._create_fallback_model()

    def _calculate_scores(self, dtm, feature_names, keyword_sets, job_descriptions):
        results = []
        dtm_coo = dtm.tocoo()
        job_descriptions_list = list(job_descriptions.items())
        for row, col, tfidf in zip(dtm_coo.row, dtm_coo.col, dtm_coo.data):
            job_index = row
            term_index = col
            title = job_descriptions_list[job_index][0]
            term = feature_names[term_index]
            # Use binary presence directly (0 or 1)
            presence = 1 if term in keyword_sets[job_index] else 0
            score = (self.config["weighting"]["tfidf_weight"] * tfidf +
                     self.config["weighting"]["frequency_weight"] * presence)
            if term in self.keyword_extractor.whitelist:
                score *= self.config["weighting"]["whitelist_boost"]
            results.append({
                "Keyword": term,
                "Job Title": title,
                "Score": score,
                "TF-IDF": tfidf,
                "Frequency": presence,
                "Category": self.keyword_extractor._categorize_term(term),
                "In Whitelist": term in self.keyword_extractor.whitelist,
            })
        return results

    # --- Improved _create_tfidf_matrix function ---
    def _create_tfidf_matrix(self, texts, keyword_sets):
        """Create and return a TF-IDF matrix for the job descriptions using pre-validated keyword sets.

        The vectorizer is adjusted to treat each document as a pre-tokenized list,
        preserving multi-word keywords without re-tokenization. As an extra safety
        measure, we also filter each document's keyword set to remove invalid tokens.
        """
        max_features = self.config.get("tfidf_max_features", 10000)
        vectorizer = TfidfVectorizer(
            ngram_range=self.keyword_extractor.ngram_range,
            lowercase=False,
            # Use identity functions so that input is treated as pre-tokenized lists.
            tokenizer=lambda x: x,
            preprocessor=lambda x: x,
            max_features=max_features,
            dtype=np.float32,
        )
        # Validate each document's keyword set to discard any term that includes a word of length <=1.
        validated_sets = [
            [kw for kw in kw_set if all(len(word) > 1 for word in kw.split())]
            for kw_set in keyword_sets
        ]
        logger.debug(f"Validated keyword sets sample: {validated_sets[:2]}")
        try:
            dtm = vectorizer.fit_transform(validated_sets)
        except ValueError as e:
            logger.error(f"TF-IDF vectorization failed: {e}. Check keyword_sets content.")
            return None, []
        feature_names = vectorizer.get_feature_names_out()
        if len(feature_names) == max_features:
            logger.warning(f"TF-IDF vocabulary reached the limit of {max_features} features")
        if not feature_names.size:
            logger.warning("No features extracted by TF-IDF. Check input keyword sets.")
        return dtm, feature_names

    def analyze_jobs(self, job_descriptions: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Analyze job descriptions and extract keywords.

        Args:
            job_descriptions: A dictionary of job titles and descriptions.

        Returns:
            A tuple containing:
                - A pandas DataFrame summarizing the keyword scores.
                - A pandas DataFrame containing the detailed keyword scores for each job description.
        """
        self._validate_input(job_descriptions)
        max_retries = self.config.get("max_retries", 2)
        strict_mode = self.config.get("strict_mode", True)
        for attempt in range(max_retries + 1):
            try:
                if self._needs_chunking(job_descriptions):
                    return self._analyze_jobs_chunked(job_descriptions)
                else:
                    return self._analyze_jobs_internal(job_descriptions)
            except (MemoryError, MPTimeoutError) as e:
                logger.warning(
                    f"{type(e).__name__} (attempt {attempt + 1}/{max_retries + 1}): {e}"
                )
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

    def _analyze_jobs_chunked(
        self, job_descriptions: Dict
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        final_summaries = []
        final_details = []
        remaining_jobs = job_descriptions.copy()
        while remaining_jobs:
            chunk_size = self._calculate_chunk_size(remaining_jobs)
            chunk = dict(list(remaining_jobs.items())[:chunk_size])
            # Remove processed jobs from remaining dictionary
            for key in list(chunk.keys()):
                remaining_jobs.pop(key)

            try:
                chunk_results = self._process_chunk(chunk)
                if chunk_results:
                    summary_chunk, details_chunk = chunk_results
                    final_summaries.append(summary_chunk)
                    final_details.append(details_chunk)
                self._memory_check()
            except Exception as e:
                logger.error(f"Error processing a chunk: {e}")
                if self.config.get("strict_mode", True):
                    raise
        if not final_summaries:
            return pd.DataFrame(), pd.DataFrame()
        full_summary = pd.concat(final_summaries)
        full_details = pd.concat(final_details)
        final_summary = (full_summary.groupby("Keyword")
                                    .agg({"Total_Score": "sum", "Avg_Score": "mean", "Job_Count": "sum"})
                                    .sort_values("Total_Score", ascending=False))
        return final_summary, full_details

    def _process_chunk(
        self, chunk: Dict
    ) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
        if not chunk:
            return None
        try:
            texts = list(chunk.values())
            keyword_sets = self.keyword_extractor.extract_keywords(texts)
            dtm, features = self._create_tfidf_matrix(texts, keyword_sets)
            # Fix: pass the chunk dict instead of a list of keys.
            results = self._calculate_scores(dtm, features, keyword_sets, chunk)
            df = pd.DataFrame(results)
            if df.empty:
                return pd.DataFrame(), pd.DataFrame()
            summary_chunk = df.groupby("Keyword").agg({"Score": ["sum", "mean"], "Job Title": "nunique"})
            summary_chunk.columns = ["Total_Score", "Avg_Score", "Job_Count"]
            details_chunk = df
            self._memory_check()
            return summary_chunk, details_chunk
        except Exception as e:
            logger.error(f"Error processing chunk: {e}")
            if self.config.get("strict_mode", True):
                raise
            else:
                return None

    def _analyze_jobs_internal(
        self, job_descriptions: Dict
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Analyze job descriptions without chunking.

        Args:
            job_descriptions: A dictionary of job titles and descriptions.

        Returns:
            A tuple containing:
                - A pandas DataFrame summarizing the keyword scores.
                - A pandas DataFrame containing the detailed keyword scores for each job description.
        """
        texts = list(job_descriptions.values())
        keyword_sets = self.keyword_extractor.extract_keywords(texts)
        dtm, feature_names = self._create_tfidf_matrix(texts, keyword_sets)
        results = self._calculate_scores(
            dtm, feature_names, keyword_sets, job_descriptions
        )
        df = pd.DataFrame(results)
        summary = (
            df.groupby("Keyword")
            .agg({"Score": ["sum", "mean"], "Job Title": "nunique"})
            .rename(
                columns={
                    "Score": {"sum": "Total_Score", "mean": "Avg_Score"},
                    "Job Title": {"nunique": "Job_Count"},
                }
            )
            .sort_values("Total_Score", ascending=False)
        )
        pivot = df.pivot_table(
            values="Score",
            index="Keyword",
            columns="Job Title",
            aggfunc="sum",
            fill_value=0,
        )
        return summary, pivot

    def _needs_chunking(self, jobs: Dict) -> bool:
        """Determine if chunking is required based on job count and memory usage.

        Args:
            jobs: A dictionary of job titles and descriptions.

        Returns:
            True if chunking is required, False otherwise.
        """
        num_jobs = len(jobs)
        auto_chunk_threshold = self.config.get("auto_chunk_threshold", 100)
        memory_percent = psutil.virtual_memory().percent
        memory_threshold = self.config.get("memory_threshold", 70)
        return num_jobs > auto_chunk_threshold or memory_percent > memory_threshold

    def _calculate_chunk_size(self, jobs: Dict) -> int:
        """Calculate an appropriate chunk size based on available memory.

        Args:
            jobs: A dictionary of job titles and descriptions.

        Returns:
            The calculated chunk size.
        """
        if not jobs:
            return 1
        total_size = sum(len(desc) for desc in jobs.values())
        avg_job_size = total_size / len(jobs)
        free_mem = psutil.virtual_memory().available
        max_memory_percent = self.config.get("max_memory_percent", 85) / 100
        target_memory_per_chunk = free_mem * max_memory_percent
        min_chunk_size = self.config.get("min_chunk_size", 1)
        max_chunk_size = self.config.get("max_chunk_size", 1000)
        buffer_factor = 2 if free_mem > 2 * target_memory_per_chunk else 1.5
        return max(
            min_chunk_size,
            min(
                max_chunk_size,
                int(target_memory_per_chunk / (avg_job_size * buffer_factor)),
            ),
        )

    def _memory_check(self):
        """Check memory usage and clear caches if necessary."""
        memory_percent = psutil.virtual_memory().percent
        memory_threshold = self.config.get("memory_threshold", 70)
        if memory_percent > memory_threshold:
            logger.warning(
                f"Memory usage high ({memory_percent:.1f}%). Clearing caches."
            )
            self._clear_caches()

    def _clear_caches(self):
        """Clear caches to reduce memory usage."""
        if hasattr(self.keyword_extractor.preprocessor, "_cache"):
            self.keyword_extractor.preprocessor._cache.clear()
        # Remove or fix the problematic line
        # self.nlp.vocab._reset_cache(True)  # Comment out or remove this
        if hasattr(self.keyword_extractor, "_get_term_vector"):
            self.keyword_extractor._get_term_vector.cache_clear()
        gc.collect()

    def _validate_input(self, raw_jobs: Dict) -> Dict:
        """Validate and sanitize the input job descriptions.
    
        Args:
            raw_jobs: A dictionary (or list) of job titles and descriptions.
    
        Returns:
            A dictionary of validated job titles and descriptions.
        """
        # If raw_jobs is a list, convert it to a dictionary with generated titles.
        if isinstance(raw_jobs, list):
            raw_jobs = {f"Job_{i+1}": job for i, job in enumerate(raw_jobs)}
    
        valid_jobs = {}
        errors = []
        for raw_title, raw_desc in raw_jobs.items():
            title_result = self._validate_title(raw_title)
            desc_result = self._validate_description(raw_desc)
            if title_result.valid and desc_result.valid:
                valid_jobs[title_result.value] = desc_result.value
            else:
                if not title_result.valid:
                    errors.append(
                        f"Job '{raw_title}': Invalid title - {title_result.reason}"
                    )
                if not desc_result.valid:
                    errors.append(
                        f"Job '{raw_title}': Invalid description - {desc_result.reason}"
                    )
        if len(valid_jobs) < self.config.get("min_jobs", 2):
            error_message = "Insufficient valid job descriptions:\n" + "\n".join(errors)
            logger.warning(error_message)
            return {}
        return valid_jobs

    def _validate_title(self, title) -> ValidationResult:
        """Validate a job title.

        Args:
            title: The job title to validate.

        Returns:
            A ValidationResult object.
        """
        allow_numeric = self.config.get("validation", {}).get(
            "allow_numeric_titles", True
        )
        if not isinstance(title, str):
            if allow_numeric:
                title = str(title)
                logger.warning(f"Job title converted to string: {title}")
            else:
                return ValidationResult(False, None, "Job title must be a string")
        stripped_title = title.strip()
        if not stripped_title:
            return ValidationResult(False, None, "Job title cannot be empty")
        min_len = self.config.get("validation", {}).get("title_min_length", 2)
        max_len = self.config.get("validation", {}).get("title_max_length", 100)
        if not min_len <= len(stripped_title) <= max_len:
            return ValidationResult(
                False, None, f"Invalid length (must be {min_len}-{max_len} characters)"
            )
        return ValidationResult(True, stripped_title)

    def _validate_description(self, desc) -> ValidationResult:
        """Validate a job description.

        Args:
            desc: The job description to validate.

        Returns:
            A ValidationResult object.
        """
        if not isinstance(desc, str):
            return ValidationResult(False, None, "Description must be a string")
        try:
            encoding = self.config.get("text_encoding", "utf-8")
            cleaned_desc = desc.encode(encoding, errors="replace").decode(encoding)
            if cleaned_desc != desc:
                logger.warning("Invalid characters replaced in description")
        except Exception as e:
            return ValidationResult(False, None, f"Encoding error: {e}")
        cleaned_desc = re.sub(r"http\S+|www\.\S+", "", cleaned_desc, flags=re.UNICODE)
        cleaned_desc = re.sub(r"\S+@\S+", "", cleaned_desc, flags=re.UNICODE)
        cleaned_desc = re.sub(r"[\x00-\x1F\x7F-\x9F]", "", cleaned_desc, flags=re.UNICODE)
        cleaned_desc = re.sub(
            r"[^\w\s\U0001F600-\U0001F64F\U0001F680-\U0001F6FF-]", " ", cleaned_desc
        )  # Keep emojis and literal hyphen (hyphen placed at end)
        cleaned_desc = re.sub(r"\s+", " ", cleaned_desc).strip()  # Normalize whitespace

        min_len = self.config.get("min_desc_length", 50)
        max_len = self.config.get("max_desc_length", 100000)

        if len(cleaned_desc) < min_len:
            logger.warning(f"Description is shorter than minimum length ({min_len})")
            return ValidationResult(
                True, cleaned_desc, "Description is short"
            )  # Return as valid, but with reason

        if len(cleaned_desc) > max_len:
            return ValidationResult(
                False, None, f"Description is longer than maximum length ({max_len})"
            )

        return ValidationResult(True, cleaned_desc)  # Valid description


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    if vec1.shape != vec2.shape:
        logger.warning(f"Vector dimension mismatch: {vec1.shape} vs {vec2.shape}")
        return 0.0
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)


# --- Command Line Interface ---
def parse_arguments():
    """
    Parse command-line arguments.

    Defines the command-line interface for the script, allowing the user to
    specify input, configuration, and output files.

    Arguments:
        -i, --input: Path to the input JSON file containing job descriptions (default: "job_descriptions.json").
        -c, --config: Path to the configuration file (YAML format) (default: "config.yaml").
        -o, --output: Path to the output Excel file where results will be saved (default: "results.xlsx").

    Returns:
        An argparse.Namespace object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description="ATS Keyword Optimizer")
    parser.add_argument(
        "-i", "--input", default="job_descriptions.json", help="Input JSON file"
    )
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
        summary.to_excel(
            writer, sheet_name="Summary"
        )  # Save summary to "Summary" sheet
        details.to_excel(
            writer, sheet_name="Detailed Scores"
        )  # Save details to "Detailed Scores" sheet
    print(f"Analysis complete. Results saved to {output_file}")


def load_job_data(input_file: str) -> Dict:
    """
    Load and validate job data from a JSON file.

    Args:
        input_file: Path to the input JSON file.

    Returns:
        A dictionary of job titles and descriptions.

    Expected JSON format:
    {
        "Job Title 1": "Job Description 1",
        "Job Title 2": "Job Description 2",
        ...
    }

    Raises:
        SystemExit: If the input file is not found or contains invalid JSON.
    """
    try:
        with open(input_file, encoding="utf-8") as f:  # Specify UTF-8 encoding
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
    if sys.version_info < (3, 8):
        logger.error("Requires Python 3.8+")
        sys.exit(1)
    args = parse_arguments()
    try:
        run_analysis(args)
    except ConfigError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(78)  # Using exit code 78 (EX_CONFIG) for config errors
    except InputValidationError as e:
        logger.error(f"Input validation error: {e}")
        sys.exit(77)  # Using exit code 77 (EX_DATAERR) for input errors
    except MemoryError as e:
        logger.error(f"Memory error: {e}")
        sys.exit(70)  # Exit code 70 (EX_SOFTWARE) for system errors
    except TimeoutError as e:
        logger.error(f"Timeout error: {e}")
        sys.exit(73)
    except Exception as e:
        logger.exception(f"Unhandled exception: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()  # Run the main function if the script is executed directly

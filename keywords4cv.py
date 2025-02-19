import argparse
import hashlib
import json
import logging
import re
import sys
from collections import OrderedDict
from typing import Dict, List, Set, Tuple
from multiprocessing import Process, Queue

import nltk
import numpy as np
import pandas as pd
import spacy
import yaml
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from functools import lru_cache
import platform
import psutil


# --- Custom Exceptions ---
class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass

class InputValidationError(Exception):
    """Custom exception for input validation errors."""
    pass

class EnvironmentError(Exception):
    """Custom exception for environment errors (e.g., Python version)."""
    pass

class TimeoutError(Exception):
    """Custom exception for analysis timeout."""
    pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("ats_optimizer.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# --- NLTK Resource Management ---
NLTK_RESOURCES = [
    "corpora/wordnet",
    "corpora/averaged_perceptron_tagger",
    "tokenizers/punkt",
]

def ensure_nltk_resources():
    """Ensure required NLTK resources are available."""
    for resource in NLTK_RESOURCES:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource.split("/")[1], quiet=True)

# Load spaCy model efficiently
def load_spacy_model(model_name="en_core_web_sm"):
    """Load spaCy model with error handling and optimizations."""
    try:
        nlp = spacy.load(model_name, disable=["parser", "ner", "lemmatizer"])
        nlp.add_pipe("lemmatizer", config={"mode": "rule"})  # Faster than default
        return nlp
    except OSError:
        try:
            spacy.cli.download(model_name)
            return load_spacy_model(model_name)
        except Exception:
            logger.exception(f"Failed to initialize spaCy model '{model_name}'")
            sys.exit(1)

# Initialize NLP components (will be loaded later with config)
nlp = None

class EnhancedTextPreprocessor:
    """Optimized text preprocessing with memoization and batch processing."""

    def __init__(self, config: Dict):
        """Initialize the preprocessor with a configuration dictionary."""
        self.config = config
        self.stop_words = self._load_stop_words()
        self.lemmatizer = WordNetLemmatizer()
        self.regex_patterns = {
            "url": re.compile(r"http\S+|www\.\S+"),
            "email": re.compile(r"\S+@\S+"),
            "special_chars": re.compile(r"[^\w\s-]"),
            "whitespace": re.compile(r"\s+"),
        }
        self._cache = OrderedDict()  # Use OrderedDict for LRU cache
        self._CACHE_SIZE = config.get("cache_size", 1000)
        self.config_hash = self._calculate_config_hash()

    def _calculate_config_hash(self) -> str:
        """Calculates a hash of the relevant configuration parts."""
        relevant_config = {
            "stop_words": self.config.get("stop_words", []),
            "stop_words_add": self.config.get("stop_words_add", []),
            "stop_words_exclude": self.config.get("stop_words_exclude", []),
            # Add other relevant config options here
        }
        config_str = json.dumps(relevant_config, sort_keys=True)
        return hashlib.sha256(config_str.encode("utf-8")).hexdigest()

    def _load_stop_words(self) -> Set[str]:
        """Load and validate stop words from config."""
        stop_words = set(self.config.get("stop_words", []))
        stop_words.update(self.config.get("stop_words_add", []))
        stop_words.difference_update(self.config.get("stop_words_exclude", []))

        if len(stop_words) < 50:
            logger.warning("Stop words list seems unusually small. Verify config.")

        return stop_words

    def preprocess(self, text: str) -> str:
        """Preprocess single text with caching (LRU)."""
        current_hash = self._calculate_config_hash()
        if current_hash != self.config_hash:
            self._cache.clear()
            self.config_hash = current_hash

        if text in self._cache:
            self._cache.move_to_end(text)  # Mark as recently used
            return self._cache[text]

        cleaned = text.lower()
        cleaned = self.regex_patterns["url"].sub("", cleaned)
        cleaned = self.regex_patterns["email"].sub("", cleaned)
        cleaned = self.regex_patterns["special_chars"].sub(" ", cleaned)
        cleaned = self.regex_patterns["whitespace"].sub(" ", cleaned).strip()

        if len(self._cache) >= self._CACHE_SIZE:
            self._cache.popitem(last=False)  # Remove least recently used
        self._cache[text] = cleaned
        return cleaned

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """Batch preprocessing."""
        return [self.preprocess(text) for text in texts]

    def tokenize_batch(self, texts: List[str]) -> List[List[str]]:
        """Batch tokenization with spaCy and optimized lemmatization."""
        tokens_list = []
        try:
            from multiprocessing import cpu_count

            n_process = min(cpu_count(), 4)  # Limit parallel processes
        except ImportError:
            n_process = 1

        for doc in nlp.pipe(texts, batch_size=50, n_process=n_process):
            tokens = []
            for token in doc:
                if (
                    token.text in self.stop_words
                    or len(token.text) <= 1
                    or token.text.isnumeric()
                ):
                    continue
                lemma = token.lemma_.lower().strip()
                tokens.append(lemma)
            tokens_list.append(tokens)
        return tokens_list

class AdvancedKeywordExtractor:
    """Enhanced keyword extraction with phrase detection and semantic analysis."""

    def __init__(self, config: Dict):
        """Initialize the keyword extractor."""
        self.config = config
        self.preprocessor = EnhancedTextPreprocessor(config)
        self.whitelist = self._create_expanded_whitelist()
        self.ngram_range = tuple(config.get("ngram_range", [1, 3]))
        self.whitelist_ngram_range = tuple(config.get("whitelist_ngram_range", [1, 3]))

    def _create_expanded_whitelist(self) -> Set[str]:
        """Create whitelist with multi-word phrases and synonyms."""
        base_skills = self.config.get("skills_whitelist", [])
        processed = set()

        # Batch process skills
        cleaned_skills = self.preprocessor.preprocess_batch(base_skills)
        tokenized = self.preprocessor.tokenize_batch(cleaned_skills)

        # Generate n-grams and synonyms
        for tokens in tokenized:
            for n in range(
                self.whitelist_ngram_range[0], self.whitelist_ngram_range[1] + 1
            ):
                processed.update(self._generate_ngrams(tokens, n))

        # Add synonyms from previous implementation
        synonyms = self._generate_synonyms(base_skills)
        processed.update(synonyms)

        return processed

    def _generate_synonyms(self, skills: List[str]) -> Set[str]:
        """Generate semantic synonyms using WordNet and spaCy."""
        synonyms = set()
        for skill in skills:
            if not skill.strip():
                logger.warning("Skipping empty skill in whitelist")
                continue
            doc = nlp(skill)
            lemmatized = " ".join([token.lemma_ for token in doc]).lower()
            if lemmatized != skill.lower():
                synonyms.add(lemmatized)

            for token in doc:
                if token.text.strip():  # Check if token text is not empty
                    synonyms.update(
                        lemma.name().replace("_", " ").lower()
                        for syn in wordnet.synsets(token.text)
                        for lemma in syn.lemmas()
                        if lemma.name().replace("_", " ").lower() != token.text and lemma.name().replace("_", " ").lower() != lemmatized
                    )
        return synonyms


    def _generate_ngrams(self, tokens: List[str], n: int) -> Set[str]:
        """Generate n-grams from a list of tokens."""
        filtered_tokens = [token for token in tokens if token.strip()] # Filter out empty strings
        return {" ".join(filtered_tokens[i : i + n]) for i in range(len(filtered_tokens) - n + 1)}

    def extract_keywords(self, texts: List[str]) -> List[List[str]]:  # Changed return type
        cleaned = self.preprocessor.preprocess_batch(texts)
        tokenized = self.preprocessor.tokenize_batch(cleaned)

        all_keywords = []
        min_n, max_n = self.ngram_range

        for tokens in tokenized:
            keywords = []  # Changed from set to list
            # Whitelist matching using configured range
            wl_min, wl_max = self.whitelist_ngram_range
            for n in range(wl_min, wl_max + 1):
                for ngram in self._generate_ngrams(tokens, n):
                    if ngram in self.whitelist:
                        keywords.append(ngram)  # Changed from add to append

            # Configurable n-gram extraction
            for n in range(min_n, max_n + 1):
                keywords.extend(self._generate_ngrams(tokens, n))

            all_keywords.append(keywords)

        return all_keywords

class ATSOptimizer:
    """Main analysis class with enhanced scoring and validation."""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the ATS Optimizer with a configuration file path."""
        # Check Python version first
        if sys.version_info < (3, 8):
            raise EnvironmentError("Requires Python 3.8+")

        self.config = self._load_config(config_path)
        self.keyword_extractor = AdvancedKeywordExtractor(self.config)
        self._init_categories()
        self._validate_config()

        # Check spaCy version (warning only)
        if spacy.__version__ < "3.0.0":  # Lowered to a more reasonable version
            logger.warning("spaCy version <3.0 may have compatibility issues")

    def _load_config(self, config_path: str) -> Dict:
        """Load and validate configuration, with backup system."""
        CONFIG_BACKUPS = [config_path, "config_backup.yaml"]

        for cfg_file in CONFIG_BACKUPS:
            try:
                with open(cfg_file) as f:
                    config = yaml.safe_load(f)

                    # Set default weightings if missing
                    config.setdefault(
                        "weighting",
                        {
                            "tfidf_weight": 0.7,
                            "frequency_weight": 0.3,
                            "whitelist_boost": 1.5,
                        },
                    )
                    config.setdefault("spacy_model", "en_core_web_sm")
                    config.setdefault("cache_size", 1000)
                    config.setdefault("whitelist_ngram_range", [1, 3])
                    config.setdefault("max_desc_length", 100000)  # ~100KB
                    config.setdefault("timeout", 600) # Default timeout
                    return config
            except FileNotFoundError:
                continue  # Try the next backup file
            except yaml.YAMLError as e:
                logger.error(f"Error parsing YAML in {cfg_file}: {e}")
                raise ConfigError(f"Error parsing YAML in {cfg_file}: {e}")
            except Exception as e:
                logger.error(f"Unexpected config error: {str(e)}")
                raise ConfigError(f"Unexpected config error: {str(e)}")
        raise ConfigError(f"No valid config found in: {CONFIG_BACKUPS}")

    def _init_categories(self):
        """Initialize category vectors and metadata."""
        self.categories = self.config.get("keyword_categories", {})
        self.category_vectors = {}

        for category, terms in self.categories.items():
            vectors = [self._get_term_vector(term) for term in terms if self._get_term_vector(term).any()]
            if vectors: # Check if vectors is not empty after filtering
                self.category_vectors[category] = {
                    "centroid": np.mean(vectors, axis=0),
                    "terms": terms,
                }
            else:
                logger.warning(f"Category {category} has no valid terms with vectors. Cannot calculate centroid.") # Log warning
                self.category_vectors[category] = { # Still initialize with terms, but no centroid
                    "centroid": None, # Mark centroid as None
                    "terms": terms,
                }

    def _validate_config(self):
        """Validate critical configuration parameters using a schema."""
        CONFIG_SCHEMA = {
            "skills_whitelist": (list, True),
            "stop_words": (list, True),
            "weighting": (dict, False),
            "ngram_range": (list, False),
            "spacy_model": (str, False),
            "cache_size": (int, False),
            "whitelist_ngram_range": (list, False),
            "keyword_categories": (dict, False),
            "max_desc_length": (int, False),
            "min_desc_length": (int, False),
            "min_jobs": (int, False),
            "similarity_threshold": (float, False),
            "timeout": (int, False)
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

    def _calculate_scores(self, dtm, feature_names, keyword_sets, job_descriptions):
        """Helper function to calculate keyword scores."""
        results = []
        for idx, title in enumerate(job_descriptions):
            row = dtm[idx]
            keywords = keyword_sets[idx]

            for col in row.nonzero()[1]:
                term = feature_names[col]
                tfidf = row[0, col]
                freq = keywords.count(term)

                score = (
                    self.config["weighting"]["tfidf_weight"] * tfidf
                    + self.config["weighting"]["frequency_weight"] * np.log1p(freq)
                )

                if term in self.keyword_extractor.whitelist:
                    score *= self.config["weighting"]["whitelist_boost"]

                results.append({
                    "Keyword": term,
                    "Job Title": title,
                    "Score": score,
                    "TF-IDF": tfidf,
                    "Frequency": freq,
                    "Category": self._categorize_term(term),
                    "In Whitelist": term in self.keyword_extractor.whitelist,
                })
        return results

    def _create_tfidf_matrix(self, texts, keyword_sets):
        """Creates and fits the TF-IDF vectorizer."""
        vectorizer = TfidfVectorizer(
            ngram_range=self.keyword_extractor.ngram_range,
            lowercase=False,
            tokenizer=lambda x: x,
            preprocessor=lambda x: x,
            max_features=5000,
            dtype=np.float32,
        )
        dtm = vectorizer.fit_transform([" ".join(kw) for kw in keyword_sets])

        if len(vectorizer.get_feature_names_out()) == 5000:
            logger.warning("TF-IDF vocabulary truncated to 5000 features.  Consider increasing max_features.")
        return dtm, vectorizer.get_feature_names_out()

    def analyze_jobs(self, job_descriptions: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Main analysis method with enhanced scoring and memory safety."""
        self._validate_input(job_descriptions)

        # Resource check
        max_mem = psutil.virtual_memory().available * 0.8
        estimated_size = sum(len(d) for d in job_descriptions.values()) * 10
        if estimated_size > max_mem:
            raise MemoryError(f"Insufficient memory: need {estimated_size//1e6}MB, have {max_mem//1e6}MB")

        result_queue = Queue()

        def _run_analysis(queue):
            try:
                texts = list(job_descriptions.values())
                keyword_sets = self.keyword_extractor.extract_keywords(texts)

                # Create document-term matrix and calculate scores
                dtm, feature_names = self._create_tfidf_matrix(texts, keyword_sets)
                results = self._calculate_scores(dtm, feature_names, keyword_sets, job_descriptions)


                # Create summary dataframes
                df = pd.DataFrame(results)
                summary = df.groupby("Keyword").agg({
                    "Score": ["sum", "mean"],
                    "Job Title": "nunique"
                }).rename(columns={
                    "Score": {"sum": "Total_Score", "mean": "Avg_Score"},
                    "Job Title": {"nunique": "Job_Count"}
                }).sort_values("Total_Score", ascending=False)

                pivot = df.pivot_table(
                    values="Score",
                    index="Keyword",
                    columns="Job Title",
                    aggfunc="sum",
                    fill_value=0,
                )

                queue.put((summary, pivot))
            except Exception as e:
                queue.put(e)

        # Run analysis with timeout
        p = Process(target=_run_analysis, args=(result_queue,))
        p.start()
        p.join(timeout=self.config.get("timeout", 600))

        if p.is_alive():
            p.terminate()
            p.join()
            raise TimeoutError("Analysis timed out")

        result = result_queue.get()
        if isinstance(result, Exception):
            raise result

        return result

    def _categorize_term(self, term: str) -> str:
        """Categorize term using hybrid approach."""
        # First try direct matches
        for category, data in self.category_vectors.items():
            if any(keyword.lower() in term.lower() for keyword in data["terms"]):
                return category

        # Then semantic similarity
        return self._semantic_categorization(term)

    @lru_cache(maxsize=5000)  # Cache term vectors
    def _get_term_vector(self, term: str) -> np.ndarray:
        """Gets the vector for a term, with error handling."""
        try:
            return nlp(term).vector
        except Exception as e:
            logger.warning(f"Vectorization failed for '{term}': {str(e)}")
            return np.array([]) # Return empty array

    def _semantic_categorization(self, term: str) -> str:
        """Categorize using spaCy embeddings, with fallback."""
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

    def _validate_input(self, jobs: Dict):
        """Validate job descriptions input, including length and control chars."""
        self._validate_input_type(jobs)
        self._validate_min_jobs(jobs)
        for title, desc in jobs.items():
            self._validate_job_title(title)
            self._validate_description_length(title, desc)
            self._validate_control_characters(title, desc)

    def _validate_input_type(self, jobs: Dict):
        if not isinstance(jobs, dict):
            raise InputValidationError("Input must be a dictionary")

    def _validate_min_jobs(self, jobs: Dict):
        min_jobs = self.config.get("min_jobs", 2)
        if len(jobs) < min_jobs:
            raise InputValidationError(f"At least {min_jobs} job descriptions required")

    def _validate_job_title(self, title: str):
        if not isinstance(title, str) or not title.strip():
            raise InputValidationError(f"Invalid job title: {title}")

    def _validate_description_length(self, title: str, desc: str):
        min_length = self.config.get("min_desc_length", 50)
        max_length = self.config.get("max_desc_length", 100000)
        if len(desc.split()) < min_length:
            logger.warning(f"Short description for {title} - may affect results")
        if len(desc) > max_length:
            raise InputValidationError(
                f"Description too long for {title} ({len(desc)} chars)"
            )

    def _validate_control_characters(self, title: str, desc: str):
        # Check for non-printable Unicode characters
        if re.search(r"[\x00-\x1F\x7F-\x9F]", desc):
            raise InputValidationError(
                f"Invalid control characters in description: {title}"
            )

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Safe cosine similarity calculation."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)

# --- Command Line Interface ---
def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="ATS Keyword Optimizer")
    parser.add_argument("-i", "--input", required=True, help="Input JSON file")
    parser.add_argument("-c", "--config", default="config.yaml", help="Config file")
    parser.add_argument("-o", "--output", default="results.xlsx", help="Output file")
    return parser.parse_args()

def initialize_analyzer(config_path: str):
    """Initializes the analyzer, including NLTK resources and spaCy model."""
    ensure_nltk_resources()
    global nlp
    nlp = load_spacy_model()  # Load default, will be overridden if config specifies
    return ATSOptimizer(config_path)

def save_results(summary: pd.DataFrame, details: pd.DataFrame, output_file: str):
    """Saves the analysis results to an Excel file."""
    with pd.ExcelWriter(output_file) as writer:
        summary.to_excel(writer, sheet_name="Summary")
        details.to_excel(writer, sheet_name="Detailed Scores")
    print(f"Analysis complete. Results saved to {output_file}")

def load_job_data(input_file: str) -> Dict:
    """Loads and validates the job data from a JSON file."""
    try:
        with open(input_file) as f:
            jobs = json.load(f)
        return jobs
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {input_file}: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        logger.error(f"Input file not found {input_file}: {e}")
        sys.exit(1)

def main():
    """Main function to run the ATS keyword analysis."""
    args = parse_arguments()

    try:
        analyzer = initialize_analyzer(args.config)
        jobs = load_job_data(args.input)
        summary, details = analyzer.analyze_jobs(jobs)
        save_results(summary, details, args.output)

    except ConfigError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except InputValidationError as e:
        logger.error(f"Input validation error: {e}")
        sys.exit(1)
    except MemoryError as e:  # Catch MemoryError specifically
        logger.error(f"Memory error: {e}")
        sys.exit(1)
    except TimeoutError as e:
        logger.error(f"Timeout error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(
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
    main()
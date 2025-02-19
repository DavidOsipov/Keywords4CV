import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import nltk
import numpy as np
import pandas as pd
import spacy
import yaml
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Custom Exceptions ---
class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass

class InputValidationError(Exception):
    """Custom exception for input validation errors."""
    pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("ats_optimizer.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- NLTK Resource Management ---
NLTK_RESOURCES = ['corpora/wordnet', 'corpora/averaged_perceptron_tagger', 'tokenizers/punkt']

def ensure_nltk_resources():
    """Ensure required NLTK resources are available."""
    for resource in NLTK_RESOURCES:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource.split('/')[1], quiet=True)

# Load spaCy model efficiently
def load_spacy_model():
    """Load spaCy model with error handling and optimizations."""
    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
        nlp.add_pipe("lemmatizer", config={"mode": "rule"})  # Faster than default
        return nlp
    except OSError:
        try:
            spacy.cli.download("en_core_web_sm")
            return load_spacy_model()
        except Exception as e:
            logger.exception("Failed to initialize spaCy model")
            sys.exit(1)

# Initialize NLP components
ensure_nltk_resources()
nlp = load_spacy_model()

class EnhancedTextPreprocessor:
    """Optimized text preprocessing with memoization and batch processing."""

    _CACHE_SIZE = 1000  # LRU cache size for preprocessing results

    def __init__(self, config: Dict):
        """Initialize the preprocessor with a configuration dictionary.

        Args:
            config (Dict): Configuration dictionary containing stop words and regex patterns.
        """
        self.config = config
        self.stop_words = self._load_stop_words()
        self.lemmatizer = WordNetLemmatizer()
        self.regex_patterns = {
            "url": re.compile(r"http\S+|www\.\S+"),
            "email": re.compile(r"\S+@\S+"),
            "special_chars": re.compile(r"[^\w\s-]"),
            "whitespace": re.compile(r"\s+")
        }
        self._cache = {}

    def _load_stop_words(self) -> Set[str]:
        """Load and validate stop words from config.

        Returns:
            Set[str]: A set of stop words.
        """
        stop_words = set(self.config.get("stop_words", []))
        stop_words.update(self.config.get("stop_words_add", []))
        stop_words.difference_update(self.config.get("stop_words_exclude", []))

        if len(stop_words) < 50:
            logger.warning("Stop words list seems unusually small. Verify config.")

        return stop_words

    def preprocess(self, text: str) -> str:
        """Preprocess single text with caching.

        Args:
            text (str): The input text.

        Returns:
            str: The preprocessed text.
        """
        if text in self._cache:
            return self._cache[text]

        cleaned = text.lower()
        cleaned = self.regex_patterns["url"].sub("", cleaned)
        cleaned = self.regex_patterns["email"].sub("", cleaned)
        cleaned = self.regex_patterns["special_chars"].sub(" ", cleaned)
        cleaned = self.regex_patterns["whitespace"].sub(" ", cleaned).strip()

        # Manage cache size
        if len(self._cache) >= self._CACHE_SIZE:
            self._cache.pop(next(iter(self._cache)))
        self._cache[text] = cleaned

        return cleaned

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """Batch preprocessing with parallel processing.

        Args:
            texts (List[str]): A list of input texts.

        Returns:
            List[str]: A list of preprocessed texts.
        """
        return [self.preprocess(text) for text in texts]

    def tokenize_batch(self, texts: List[str]) -> List[List[str]]:
        """Batch tokenization with spaCy and optimized lemmatization.

        Args:
            texts (List[str]): A list of input texts.

        Returns:
            List[List[str]]: A list of lists of tokens.
        """
        tokens_list = []
        for doc in nlp.pipe(texts, batch_size=50, n_process=-1):
            tokens = []
            for token in doc:
                if (token.text in self.stop_words or
                    len(token.text) <= 1 or
                    token.text.isnumeric()):
                    continue
                lemma = token.lemma_.lower().strip()
                tokens.append(lemma)
            tokens_list.append(tokens)
        return tokens_list

class AdvancedKeywordExtractor:
    """Enhanced keyword extraction with phrase detection and semantic analysis."""

    def __init__(self, config: Dict):
        """Initialize the keyword extractor with a configuration dictionary.

        Args:
            config (Dict): Configuration dictionary containing skills whitelist and ngram range.
        """
        self.config = config
        self.preprocessor = EnhancedTextPreprocessor(config)
        self.whitelist = self._create_expanded_whitelist()
        self.ngram_range = tuple(config.get("ngram_range", [1, 3]))
        self.entity_types = set(config.get("allowed_entity_types", []))

    def _create_expanded_whitelist(self) -> Set[str]:
        """Create whitelist with multi-word phrases and synonyms.

        Returns:
            Set[str]: A set of whitelisted keywords.
        """
        base_skills = self.config.get("skills_whitelist", [])
        processed = set()

        # Batch process skills
        cleaned_skills = self.preprocessor.preprocess_batch(base_skills)
        tokenized = self.preprocessor.tokenize_batch(cleaned_skills)

        # Generate n-grams and synonyms
        for tokens in tokenized:
            for n in range(1, 4):
                processed.update(self._generate_ngrams(tokens, n))

        # Add synonyms from previous implementation
        synonyms = self._generate_synonyms(base_skills)
        processed.update(synonyms)

        return processed

    def _generate_synonyms(self, skills: List[str]) -> Set[str]:
        """Generate semantic synonyms using WordNet and spaCy.

        Args:
            skills (List[str]): A list of skills.

        Returns:
            Set[str]: A set of synonyms.
        """
        synonyms = set()
        for skill in skills:
            doc = nlp(skill)
            # Get lemma-based variations
            lemmatized = " ".join([token.lemma_ for token in doc]).lower()
            if lemmatized != skill.lower():
                synonyms.add(lemmatized)

            # WordNet synonyms
            for token in doc:
                for syn in wordnet.synsets(token.text):
                    for lemma in syn.lemmas():
                        synonym = lemma.name().replace("_", " ").lower()
                        if synonym != token.text and synonym != lemmatized:
                            synonyms.add(synonym)
        return synonyms

    def _generate_ngrams(self, tokens: List[str], n: int) -> Set[str]:
        """Generate n-grams from a list of tokens.

        Args:
            tokens (List[str]): A list of tokens.
            n (int): The length of the n-grams.

        Returns:
            Set[str]: A set of n-grams.
        """
        return {" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)}

    def extract_keywords(self, texts: List[str]) -> List[Set[str]]:
        """Extract keywords with configurable n-gram range.

        Args:
            texts (List[str]): A list of input texts.

        Returns:
            List[Set[str]]: A list of sets of keywords.
        """
        cleaned = self.preprocessor.preprocess_batch(texts)
        tokenized = self.preprocessor.tokenize_batch(cleaned)

        all_keywords = []
        min_n, max_n = self.ngram_range

        for tokens in tokenized:
            keywords = set()
            # Whitelist matching
            for n in range(1, 4):
                for ngram in self._generate_ngrams(tokens, n):
                    if ngram in self.whitelist:
                        keywords.add(ngram)

            # Configurable n-gram extraction
            for n in range(min_n, max_n+1):
                keywords.update(self._generate_ngrams(tokens, n))

            all_keywords.append(keywords)

        return all_keywords

class ATSOptimizer:
    """Main analysis class with enhanced scoring and validation."""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the ATS Optimizer with a configuration file path.

        Args:
            config_path (str): Path to the configuration file. Defaults to "config.yaml".
        """
        self.config = self._load_config(config_path)
        self.keyword_extractor = AdvancedKeywordExtractor(self.config)
        self._init_categories()
        self._validate_config()

    def _load_config(self, config_path: str) -> Dict:
        """Load and validate configuration.

        Args:
            config_path (str): Path to the configuration file.

        Returns:
            Dict: The configuration dictionary.

        Raises:
            ConfigError: If the configuration file is invalid or missing.
        """
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)

            # Set default weightings if missing
            config.setdefault("weighting", {
                "tfidf_weight": 0.7,
                "frequency_weight": 0.3,
                "whitelist_boost": 1.5
            })

            return config
        except Exception as e:
            logger.error(f"Config error: {str(e)}")
            raise ConfigError(f"Invalid configuration: {str(e)}")

    def _init_categories(self):
        """Initialize category vectors and metadata."""
        self.categories = self.config.get("keyword_categories", {})
        self.category_vectors = {}

        for category, terms in self.categories.items():
            vectors = [nlp(term).vector for term in terms if nlp(term).vector_norm]
            if vectors:
                self.category_vectors[category] = {
                    "centroid": np.mean(vectors, axis=0),
                    "terms": terms
                }
            else:
                logger.warning("Category %s has no valid terms with vectors.", category)


    def _validate_config(self):
        """Validate critical configuration parameters.

        Raises:
            ConfigError: If required configuration keys are missing.
        """
        required_keys = ["skills_whitelist", "stop_words"]
        for key in required_keys:
            if not self.config.get(key):
                raise ConfigError(f"Missing required config key: {key}")

        if len(self.config["skills_whitelist"]) < 10:
            logger.warning("Skills whitelist seems small. Consider expanding it.")

    def analyze_jobs(self, job_descriptions: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Main analysis method with enhanced scoring.

        Args:
            job_descriptions (Dict): A dictionary of job descriptions, where keys are job titles and values are job description texts.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two pandas DataFrames:
                - The first DataFrame is a summary of keywords and their scores.
                - The second DataFrame is a pivot table of keyword scores for each job title.
        """
        self._validate_input(job_descriptions)
        texts = list(job_descriptions.values())

        # Extract keywords in batch
        keyword_sets = self.keyword_extractor.extract_keywords(texts)

        # Create document-term matrix
        vectorizer = TfidfVectorizer(
            ngram_range=self.keyword_extractor.ngram_range,
            lowercase=False,
            tokenizer=lambda x: x,
            preprocessor=lambda x: x
        )
        dtm = vectorizer.fit_transform([" ".join(ks) for ks in keyword_sets])

        # Calculate scores
        results = []
        feature_names = vectorizer.get_feature_names_out()

        for idx, title in enumerate(job_descriptions):
            row = dtm[idx]
            for col in row.nonzero()[1]:
                term = feature_names[col]
                tfidf = row[0, col]
                freq = keyword_sets[idx].count(term)

                # Enhanced scoring formula
                score = (
                    self.config["weighting"]["tfidf_weight"] * tfidf +
                    self.config["weighting"]["frequency_weight"] * np.log1p(freq)
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
                    "In Whitelist": term in self.keyword_extractor.whitelist
                })

        # Create dataframes
        df = pd.DataFrame(results)
        summary = df.groupby("Keyword").agg(
            Total_Score=("Score", "sum"),
            Avg_Score=("Score", "mean"),
            Job_Count=("Job Title", "nunique")
        ).sort_values("Total_Score", ascending=False)

        pivot = df.pivot_table(
            values="Score",
            index="Keyword",
            columns="Job Title",
            aggfunc="sum",
            fill_value=0
        )

        return summary, pivot

    def _categorize_term(self, term: str) -> str:
        """Categorize term using hybrid approach.

        Args:
            term (str): The term to categorize.

        Returns:
            str: The category of the term.
        """
        # First try direct matches
        for category, data in self.category_vectors.items():
            if any(keyword.lower() in term.lower() for keyword in data["terms"]):
                return category

        # Then semantic similarity
        return self._semantic_categorization(term)

    def _semantic_categorization(self, term: str) -> str:
        """Categorize using spaCy embeddings.

        Args:
            term (str): The term to categorize.

        Returns:
            str: The category of the term.
        """
        term_vec = nlp(term).vector
        if not term_vec.any():
            return "Other"

        best_score = self.config.get("similarity_threshold", 0.6)
        best_category = "Other"

        for category, data in self.category_vectors.items():
            similarity = cosine_similarity(term_vec, data["centroid"])
            if similarity > best_score:
                best_score = similarity
                best_category = category

        return best_category

    def _validate_input(self, jobs: Dict):
        """Validate job descriptions input.

        Args:
            jobs (Dict): A dictionary of job descriptions.

        Raises:
            InputValidationError: If the input is invalid.
        """
        if not isinstance(jobs, dict):
            raise InputValidationError("Input must be a dictionary")

        min_jobs = self.config.get("min_jobs", 2)
        if len(jobs) < min_jobs:
            raise InputValidationError(f"At least {min_jobs} job descriptions required")

        for title, desc in jobs.items():
            if not isinstance(title, str) or not title.strip():
                raise InputValidationError(f"Invalid job title: {title}")
            if len(desc.split()) < self.config.get("min_desc_length", 50):
                logger.warning(f"Short description for {title} - may affect results")

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Safe cosine similarity calculation.

    Args:
        vec1 (np.ndarray): The first vector.
        vec2 (np.ndarray): The second vector.

    Returns:
        float: The cosine similarity between the two vectors.
    """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)

# --- Command Line Interface ---
def parse_arguments():
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="ATS Keyword Optimizer")
    parser.add_argument("-i", "--input", required=True, help="Input JSON file")
    parser.add_argument("-c", "--config", default="config.yaml", help="Config file")
    parser.add_argument("-o", "--output", default="results.xlsx", help="Output file")
    return parser.parse_args()

def main():
    """Main function to run the ATS keyword analysis."""
    args = parse_arguments()

    try:
        analyzer = ATSOptimizer(args.config)
        with open(args.input) as f:
            jobs = json.load(f)

        summary, details = analyzer.analyze_jobs(jobs)

        # Save results
        with pd.ExcelWriter(args.output) as writer:
            summary.to_excel(writer, sheet_name="Summary")
            details.to_excel(writer, sheet_name="Detailed Scores")

        print(f"Analysis complete. Results saved to {args.output}")

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
import re
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import yaml
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Download NLTK Resources ---
def download_nltk_resources():
    """Downloads necessary NLTK resources if not already present."""
    resources = ['punkt', 'wordnet', 'averaged_perceptron_tagger']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}' if resource == 'wordnet' else f'taggers/{resource}')
        except LookupError:
            logging.info(f"Downloading NLTK resource: {resource}")
            nltk.download(resource, quiet=True)

download_nltk_resources()  # Call the download function at the start

# --- Load spaCy Model ---
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser"])  # Disable parser for speed
except OSError:
    logger.warning("Downloading en_core_web_sm model...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["parser"])


# --- Helper Functions ---
def get_synonyms(word: str, pos: str = None) -> Set[str]:
    """Gets synonyms of a word from WordNet, optionally filtering by POS."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        if pos is None or syn.pos() == pos:  # Optional POS filter
            for lemma in syn.lemmas():
                synonym = lemma.name().replace("_", " ").lower()
                synonyms.add(synonym)
    return synonyms

# --- Classes ---
class TextPreprocessor:
    """Handles text preprocessing tasks."""

    def __init__(self, config: Dict):
        """
        Initializes the TextPreprocessor with configuration.

        Args:
            config (Dict): Configuration dictionary containing stop words.
        """
        self.config = config
        self.stop_words = self._load_stop_words()
        self.lemmatizer = WordNetLemmatizer()

    def _load_stop_words(self) -> Set[str]:
        """Loads and configures stop words from the configuration."""
        stop_words = set(self.config['stop_words'])
        stop_words.update(self.config.get('stop_words_add', []))
        stop_words.difference_update(self.config.get('stop_words_exclude', []))
        return stop_words

    def preprocess_text(self, text: str) -> str:
        """
        Cleans and standardizes input text.

        Removes newlines, tabs, URLs, email addresses, and punctuation (except hyphens),
        and converts text to lowercase.

        Args:
            text (str): The input text to preprocess.

        Returns:
            str: The preprocessed text.
        """
        try:
            text = text.lower()
            text = re.sub(r'[\n\t]+', ' ', text)          # Remove newlines and tabs
            text = re.sub(r'http\S+|www.\S+', '', text)   # Remove URLs
            text = re.sub(r'\S+@\S+', '', text)           # Remove email addresses
            text = re.sub(r'[^\w\s-]', ' ', text)          # Remove punctuation except hyphens
            text = re.sub(r'\s+', ' ', text).strip()      # Remove extra whitespace
            return text
        except Exception as e:
            logger.error(f"Error in text preprocessing: {e}")
            raise

    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        """
        Tokenizes, lemmatizes, and filters text using spaCy and WordNetLemmatizer.

        Filters out stop words, short words, and numeric tokens. Lemmatizes nouns, adjectives,
        proper nouns, and verbs.

        Args:
            text (str): The input text to tokenize and lemmatize.

        Returns:
            List[str]: A list of lemmatized tokens.
        """
        doc = nlp(text)
        tokens = []
        for token in doc:
            if token.text in self.stop_words or len(token.text) <= 1 or token.text.isnumeric():
                continue # Skip stop words, short words, and numbers

            if token.pos_ in {"NOUN", "ADJ", "PROPN", "VERB"}:
                lemma = self.lemmatizer.lemmatize(token.text, pos='v' if token.pos_ == 'VERB' else 'n')
                tokens.append(lemma)
        return tokens



class KeywordExtractor:
    """Extracts keywords from preprocessed text using various methods."""

    def __init__(self, config: Dict):
        """
        Initializes the KeywordExtractor with configuration.

        Args:
            config (Dict): Configuration dictionary containing skills whitelist and allowed entity types.
        """
        self.config = config
        self.skills_whitelist = set(self.config['skills_whitelist'])
        self.preprocessor = TextPreprocessor(self.config)
        self.allowed_entity_types = set(self.config.get('allowed_entity_types', ["ORG", "PRODUCT"])) # Configurable entity types

    def extract_ngrams(self, tokens: List[str], n: int) -> List[str]:
        """
        Generates n-grams from a list of tokens.

        Args:
            tokens (List[str]): List of tokens.
            n (int): The degree of n-gram (e.g., 2 for bigrams, 3 for trigrams).

        Returns:
            List[str]: List of n-grams.
        """
        return [' '.join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

    def get_keywords(self, text: str) -> Counter:
        """
        Extracts keywords from text using multiple methods:

        - Whitelisted skills
        - Lemmatized tokens
        - Noun chunks
        - Named Entities (NER)
        - Bigrams and Trigrams

        Args:
            text (str): The input text to extract keywords from.

        Returns:
            Counter: A Counter object containing keywords and their frequencies.
        """
        try:
            processed_text = self.preprocessor.preprocess_text(text)
            tokens = self.preprocessor.tokenize_and_lemmatize(processed_text)
            keywords = Counter()

            # 1. Add whitelisted skills directly if present in the text
            self._extract_whitelisted_skills(processed_text, keywords)

            # 2. Add lemmatized tokens (unigrams)
            keywords.update(tokens)

            # 3. Extract and add noun chunks
            self._extract_noun_chunks(processed_text, keywords)

            # 4. Extract and add Named Entities (NER) of allowed types
            self._extract_named_entities(processed_text, keywords)

            # 5. Generate and add filtered bigrams and trigrams
            self._extract_ngrams_keywords(tokens, keywords)

            return keywords

        except Exception as e:
            logger.error(f"Error in keyword extraction: {e}")
            raise

    def _extract_whitelisted_skills(self, processed_text: str, keywords: Counter):
        """Extracts whitelisted skills from processed text and updates keywords."""
        for skill in self.skills_whitelist:
            if skill in processed_text:
                keywords[skill] += 1

    def _extract_noun_chunks(self, processed_text: str, keywords: Counter):
        """Extracts noun chunks from processed text and updates keywords."""
        doc = nlp(processed_text)  # Process with spaCy *after* preprocessing
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.lower()
            if any(word in self.skills_whitelist or word not in self.preprocessor.stop_words for word in chunk_text.split()):
                keywords[chunk_text] += 1

    def _extract_named_entities(self, processed_text: str, keywords: Counter):
        """Extracts named entities from processed text and updates keywords."""
        doc = nlp(processed_text)
        for ent in doc.ents:
            if ent.label_ in self.allowed_entity_types:
                entity_text = self.preprocessor.preprocess_text(ent.text.lower())
                if entity_text:
                    keywords[entity_text] += 1

    def _extract_ngrams_keywords(self, tokens: List[str], keywords: Counter):
        """Extracts filtered n-grams (bigrams and trigrams) and updates keywords."""
        tokens_text = [t for t in tokens]
        for n in [2, 3]:
            for ngram in self.extract_ngrams(tokens_text, n):
                if any(word in self.skills_whitelist for word in ngram.split()) or all(word not in self.preprocessor.stop_words for word in ngram.split()):
                    keywords[ngram] += 1


class JobDescriptionAnalyzer:
    """Analyzes job descriptions to extract keywords and compute TF-IDF."""

    def __init__(self, config: Dict):
        """
        Initializes the JobDescriptionAnalyzer with configuration.

        Args:
            config (Dict): Configuration dictionary containing keyword categories and other settings.
        """
        self.config = config
        self.extractor = KeywordExtractor(self.config)
        self.keyword_categories = self.config.get('keyword_categories', {})

    def analyze_job_descriptions(self, job_descriptions: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Analyzes a dictionary of job descriptions, calculates TF-IDF, and generates summary and pivot tables.

        Args:
            job_descriptions (Dict[str, str]): A dictionary where keys are job titles and values are job descriptions.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two pandas DataFrames:
                - summary_stats: DataFrame summarizing keyword TF-IDF across all jobs.
                - pivot_df: DataFrame pivoting TF-IDF scores with keywords as rows and job titles as columns.
        """
        self._validate_job_descriptions_input_type(job_descriptions)
        self._validate_job_descriptions_length(job_descriptions)
        self._validate_job_description_contents(job_descriptions)

        job_texts = list(job_descriptions.values())

        vectorizer = TfidfVectorizer(
            analyzer='word',
            tokenizer=self.extractor.get_keywords,
            preprocessor=None, # Preprocessing handled within get_keywords
            ngram_range=(1, 3), # Consider unigrams, bigrams, and trigrams directly
            sublinear_tf=True,
            use_idf=True
        )

        try:
            tfidf_matrix = vectorizer.fit_transform(job_texts)
            feature_names = vectorizer.get_feature_names_out()
            keyword_data = self._extract_keyword_data(job_descriptions, tfidf_matrix, feature_names)

            df = pd.DataFrame(keyword_data)
            if df.empty:
                logger.warning("No keywords were extracted.")
                return pd.DataFrame(), pd.DataFrame()

            pivot_df = self._create_pivot_table(df)
            summary_stats = self._create_summary_stats(df)

            return summary_stats, pivot_df

        except Exception as e:
            logger.error(f"Error in job description analysis: {e}")
            raise

    def _validate_job_descriptions_input_type(self, job_descriptions: Dict[str, str]):
        """Validates if job_descriptions is a dictionary."""
        if not isinstance(job_descriptions, dict):
            raise TypeError("job_descriptions must be a dictionary.")

    def _validate_job_descriptions_length(self, job_descriptions: Dict[str, str]):
        """Validates the length of the job_descriptions dictionary."""
        if not (2 <= len(job_descriptions) <= 100):
            raise ValueError("job_descriptions must contain between 2 and 100 entries.")

    def _validate_job_description_contents(self, job_descriptions: Dict[str, str]):
        """Validates the content of each job description."""
        for title, description in job_descriptions.items():
            if not isinstance(title, str):
                raise TypeError(f"Job title '{title}' must be a string.")
            if not isinstance(description, str):
                raise TypeError(f"Job description for '{title}' must be a string.")
            if not description.strip():
                raise ValueError(f"Job description for '{title}' is empty.")
            if len(description.split()) < 10:
                logger.warning(f"Job description for '{title}' is very short. Results may be unreliable.")


    def _extract_keyword_data(self, job_descriptions: Dict[str, str], tfidf_matrix, feature_names) -> List[Dict]:
        """Extracts keyword data from TF-IDF matrix and feature names."""
        keyword_data = []
        for i, job_title in enumerate(job_descriptions):
            doc_vector = tfidf_matrix[i].tocoo()
            for j, score in zip(doc_vector.col, doc_vector.data):
                keyword = feature_names[j]
                category = self._categorize_keyword(keyword)
                keyword_data.append({
                    'keyword': keyword,
                    'job_title': job_title,
                    'tfidf': score,
                    'category': category
                })
        return keyword_data

    def _categorize_keyword(self, keyword: str) -> str:
        """Categorizes a keyword based on configured keyword categories."""
        category = "Other"
        for cat, keywords_list in self.keyword_categories.items():
            if any(kw in keyword for kw in keywords_list): # Check if extracted keyword contains category keyword
                category = cat
                break
        return category

    def _create_pivot_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates a pivot table from the keyword DataFrame."""
        return df.pivot_table(index='keyword', columns='job_title', values='tfidf', aggfunc='sum', fill_value=0)

    def _create_summary_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates summary statistics from the keyword DataFrame."""
        summary_stats = df.groupby('keyword').agg(
            total_tfidf=('tfidf', 'sum'),
            job_count=('job_title', 'nunique'),
            avg_tfidf=('tfidf', 'mean')
        ).reset_index()

        summary_stats = summary_stats.sort_values('total_tfidf', ascending=False)
        summary_stats.rename(columns={
            'total_tfidf': 'Total TF-IDF',
            'job_count': 'Job Count',
            'avg_tfidf': 'Average TF-IDF'
        }, inplace=True)
        return summary_stats


class JobKeywordAnalyzer:
    """Orchestrates job keyword analysis using configuration and analyzer classes."""

    def __init__(self, config_file: str = "config.yaml"):
        """
        Initializes the JobKeywordAnalyzer by loading configuration from a YAML file.

        Args:
            config_file (str): Path to the YAML configuration file. Defaults to "config.yaml".
        """
        try:
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_file}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML in {config_file}: {e}")
            raise

        self.analyzer = JobDescriptionAnalyzer(self.config)

    def analyze_jobs(self, job_descriptions: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Analyzes job descriptions using the configured JobDescriptionAnalyzer.

        Args:
            job_descriptions (Dict[str, str]): A dictionary of job titles and descriptions.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Summary and pivot DataFrames from the analysis.
        """
        return self.analyzer.analyze_job_descriptions(job_descriptions)



def main():
    """
    Main function to execute job keyword analysis, save results, and display top keywords.

    Loads configuration, analyzes job descriptions, saves output to Excel, and prints
    the top keywords by TF-IDF.
    """

    try:
        analyzer = JobKeywordAnalyzer()

        job_descriptions = {
            "Data Science Product Owner": (
                "People deserve more from their money.  We are looking for a Data Science Product Owner "
                "to join our team and lead the development of innovative data-driven products.  "
                "Responsibilities include defining product vision, prioritizing features, and working "
                "closely with data scientists and engineers.  The ideal candidate will have strong "
                "experience in product management, a deep understanding of data science techniques (like "
                "machine learning and statistical modeling), and excellent communication skills. "
                "Proficiency in Python, SQL, and data visualization tools (Tableau, Power BI) is required. "
                "Experience with Agile development methodologies is a plus. You will be managing data products."
            ),
            "Product Manager - SaaS": (
                "We are seeking a hands-on Product Manager to drive the strategy and execution of our "
                "SaaS platform.  This role requires a proven track record of successfully launching and "
                "scaling SaaS products.  Key responsibilities include conducting market research, gathering "
                "customer feedback, defining product requirements, and collaborating with engineering, "
                "design, and marketing teams.  Strong analytical and problem-solving skills are essential.  "
                "Experience with Agile methodologies (Scrum, Kanban) and product management tools (Jira, "
                "Confluence) is required.  Excellent communication, negotiation, and leadership skills are a must. "
                "Experience with data analysis is beneficial."
            ),
            "Data Analyst": (
                "Join our growing data analytics team! We need a Data Analyst proficient in SQL, Excel, and Python.  "
                "You will be responsible for analyzing large datasets, identifying trends, and creating reports "
                "to support business decisions. Experience with data visualization and machine learning is a plus. "
                "Strong communication and collaboration skills are important.  Experience managing databases is desirable."
            ),
            "Software Engineer (Backend)": (
                "We're hiring a talented Software Engineer to develop and maintain our backend systems.  "
                "This role requires strong proficiency in Java and experience with cloud platforms (AWS, Azure, or GCP). "
                "Experience with database technologies (SQL, NoSQL) and API development (REST, GraphQL) is essential.  "
                "Familiarity with DevOps practices (CI/CD, Docker, Kubernetes) is a plus. You must be able to deliver results."
            ),
            "Frontend Developer": (
                "We are looking for a skilled Frontend Developer to build user-friendly interfaces for our web applications.  "
                "The ideal candidate will have strong experience with JavaScript, React, and HTML/CSS.  "
                "Experience with testing frameworks and responsive design principles is required.  "
                "Good communication and teamwork skills are essential.  Experience with Node.js is a plus."
            )
        }
        summary_df, pivot_df = analyzer.analyze_jobs(job_descriptions)

        if summary_df.empty:
            print("No keywords were extracted.")
            return

        # Create output directory
        output_dir = Path(analyzer.config['output_dir'])
        output_dir.mkdir(exist_ok=True)

        # Save results to Excel
        output_file = output_dir / "job_keywords_analysis.xlsx"
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='Keyword Summary', index=False)
            pivot_df.to_excel(writer, sheet_name='Job Specific Details')

        print("\nAnalysis complete! Results saved to:", output_file)
        print("\nTop Keywords (Sorted by Total TF-IDF):")
        print(summary_df[['keyword', 'Total TF-IDF']].head(20))


    except Exception: # Removed unused variable 'e'
        logger.exception("An error occurred during main execution:") # Use exception for full traceback



if __name__ == "__main__":
    main()